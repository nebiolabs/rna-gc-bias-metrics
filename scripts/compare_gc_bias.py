#!/usr/bin/env -S uv run --script
# /// script
# requires-python = '>=3.12'
# dependencies = ['polars>=1.37.1', 'plotly>=6.0']
# ///
'''Compare GC bias profiles produced by two versions of this tool.

Runs `calculate_gc_coverage` at two git refs over the same set of BAMs and
renders one interactive HTML plot: colour identifies the BAM, line dash
identifies the run, so a per-sample difference between versions (or between
parameter sets) is readable at a glance.

    scripts/compare_gc_bias.py 0.2.0 WORKTREE \
        --fasta transcripts.fa --bam sample.bam \
        --args-b '--min_transcript_cpm 1'

Either ref may be WORKTREE, meaning the current checkout including uncommitted
tracked changes. Passing the same ref twice with different --args-a/--args-b
turns this into a parameter sweep.
'''

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

WORKTREE_REF = 'WORKTREE'

# Fixed categorical slot order. Hues are assigned in this order and never
# cycled -- past eight series hue stops carrying identity, so we refuse instead.
SERIES_COLORS = {
    'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
              '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    'dark': ['#3987e5', '#d95926', '#199e70', '#c98500',
             '#d55181', '#008300', '#9085e9', '#e66767'],
}

CHROME = {
    'light': {'surface': '#fcfcfb', 'paper': '#f9f9f7', 'ink': '#0b0b0b',
              'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7'},
    'dark': {'surface': '#1a1a19', 'paper': '#0d0d0d', 'ink': '#ffffff',
             'muted': '#898781', 'grid': '#2c2c2a', 'axis': '#383835'},
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DASHES = ['solid', 'dash']
PROFILE_COLUMNS = ['gc_fraction', 'mean_normalized_depth', 'transcriptome_bin_count']


def die(message):
    print(f'error: {message}', file=sys.stderr)
    sys.exit(1)


def note(message):
    print(message, file=sys.stderr)


def git(repo, *args, check=True):
    '''Run git against `repo` and return stripped stdout.'''
    result = subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        die(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_ref(repo, ref):
    '''Resolve a ref to a commit SHA that can be checked out into a worktree.

    WORKTREE goes through `git stash create`, which builds a commit object from
    the current working tree without touching the working tree or the stash list.
    '''
    if ref.upper() == WORKTREE_REF:
        if git(repo, 'ls-files', '--others', '--exclude-standard'):
            note(f'warning: {WORKTREE_REF} does not capture untracked files')
        sha = git(repo, 'stash', 'create')
        if sha:
            return sha
        note(f'{WORKTREE_REF}: no uncommitted changes to tracked files, using HEAD')
        return git(repo, 'rev-parse', 'HEAD')

    sha = git(repo, 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}', check=False)
    if not sha:
        die(f"'{ref}' is not a valid git ref in {repo}")
    return sha


def ensure_worktree(repo, sha, root):
    '''Check `sha` out into a cached worktree under `root`, reusing it if present.'''
    worktree = root / sha[:12]
    if worktree.exists():
        if git(worktree, 'rev-parse', 'HEAD', check=False) == sha:
            note(f'reusing worktree {worktree}')
            return worktree
        git(repo, 'worktree', 'remove', '--force', str(worktree))

    root.mkdir(parents=True, exist_ok=True)
    git(repo, 'worktree', 'add', '--detach', str(worktree), sha)
    return worktree


def detect_package(worktree):
    '''Find the package directory holding calculate_gc_coverage.

    The package was renamed gc_profile -> rna_gc_bias_metrics at tag 0.1.1, so
    the module path has to come from the checkout rather than a constant.
    '''
    matches = sorted(worktree.glob('src/*/calculate_gc_coverage.py'))
    if not matches:
        die(f'no src/*/calculate_gc_coverage.py found in {worktree}')
    if len(matches) > 1:
        names = ', '.join(match.parent.name for match in matches)
        die(f'ambiguous package in {worktree}: {names}')
    return matches[0].parent.name


def run_tool(worktree, package, fasta, bam, extra_args, tsv):
    '''Invoke calculate_gc_coverage inside `worktree` via uv.'''
    cmd = ['uv', 'run', 'python', '-m', f'{package}.calculate_gc_coverage',
           str(fasta), str(bam), '-o', str(tsv), *extra_args]
    note(f'+ (cd {worktree} && {shlex.join(cmd)})')

    # An activated venv from the caller's shell would otherwise fight uv over
    # which environment this worktree's project should use.
    env = {k: v for k, v in os.environ.items()
           if k not in ('VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'CONDA_PREFIX')}

    if subprocess.run(cmd, cwd=worktree, env=env).returncode != 0:
        die(f'{package}.calculate_gc_coverage failed on {bam.name} in {worktree}')


def load_profile(tsv, bam_name, run_label):
    frame = pl.read_csv(tsv, separator='\t')
    missing = [column for column in PROFILE_COLUMNS if column not in frame.columns]
    if missing:
        die(f"{tsv} is missing column(s): {', '.join(missing)}")
    return frame.select(
        pl.col('gc_fraction').cast(pl.Float64, strict=False),
        pl.col('mean_normalized_depth').cast(pl.Float64, strict=False),
        pl.col('transcriptome_bin_count').cast(pl.Int64, strict=False),
        bam=pl.lit(bam_name),
        run=pl.lit(run_label),
    ).sort('gc_fraction')


def build_figure(profiles, bams, runs, theme, title):
    palette = SERIES_COLORS[theme]
    chrome = CHROME[theme]

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.06,
    )

    for bam_index, bam in enumerate(bams):
        color = palette[bam_index]
        for run_index, run in enumerate(runs):
            frame = profiles.filter(
                (pl.col('bam') == bam) & (pl.col('run') == run)
            )
            if frame.is_empty():
                continue

            gc = frame['gc_fraction'].to_list()
            counts = frame['transcriptome_bin_count'].to_list()
            style = dict(color=color, width=2, dash=DASHES[run_index])

            figure.add_trace(go.Scatter(
                x=gc, y=frame['mean_normalized_depth'].to_list(),
                name=run, legendgroup=bam, legendgrouptitle_text=bam,
                mode='lines', line=style,
                connectgaps=False, customdata=counts,
                hovertemplate='%{y:.3f}  (n=%{customdata})<extra>%{fullData.name}</extra>',
            ), row=1, col=1)

            figure.add_trace(go.Scatter(
                x=gc, y=counts,
                name=run, legendgroup=bam, showlegend=False,
                mode='lines', line=style,
                connectgaps=False,
                hovertemplate='%{y}<extra>%{fullData.name}</extra>',
            ), row=2, col=1)

    # 1.0 is the within-transcript mean, i.e. the no-bias line.
    figure.add_hline(
        y=1, row=1, col=1, layer='below',
        line=dict(color=chrome['muted'], width=1, dash='dot'),
    )

    figure.update_layout(
        template='none',
        title=dict(text=title, font=dict(size=17, color=chrome['ink'])),
        plot_bgcolor=chrome['surface'],
        paper_bgcolor=chrome['paper'],
        font=dict(family=FONT_FAMILY, color=chrome['ink'], size=13),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=chrome['surface'], bordercolor=chrome['axis'],
            font=dict(family=FONT_FAMILY, color=chrome['ink']),
        ),
        legend=dict(groupclick='toggleitem', bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=80, r=40, t=80, b=60),
    )

    axis = dict(
        showgrid=True, gridcolor=chrome['grid'], gridwidth=1,
        linecolor=chrome['axis'], zeroline=False,
        tickfont=dict(color=chrome['muted']),
        title_font=dict(color=chrome['ink']),
    )
    figure.update_xaxes(
        **axis, showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor=chrome['muted'], spikethickness=1, spikedash='dot',
    )
    figure.update_yaxes(**axis)
    figure.update_xaxes(tickformat='.0%', title_text='GC fraction', row=2, col=1)
    figure.update_yaxes(title_text='Mean normalized depth', row=1, col=1)
    figure.update_yaxes(title_text='Transcriptome bin count', row=2, col=1)
    return figure


def write_report(figure, path, theme):
    '''Write a standalone HTML page.

    plotly's own write_html leaves the surrounding page unstyled, so in dark
    mode the body shows through white below the plot.
    '''
    paper = CHROME[theme]['paper']
    fragment = figure.to_html(
        include_plotlyjs=True, full_html=False, default_height='100%'
    )
    path.write_text(
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>GC bias comparison</title>\n'
        f'<style>html, body {{ margin: 0; height: 100%; background: {paper}; }}</style>\n'
        f'</head>\n<body>\n{fragment}\n</body>\n</html>\n'
    )


def slugify(text):
    return ''.join(char if char.isalnum() else '_' for char in text).strip('_')


def default_label(ref, extra_args):
    return f"{ref} ({' '.join(extra_args)})" if extra_args else ref


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare GC bias profiles between two versions of this tool.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Either ref may be WORKTREE, meaning the current checkout including\n'
            'uncommitted changes to tracked files.'
        ),
    )
    parser.add_argument('ref_a', help=f'First git ref (tag, branch, SHA, or {WORKTREE_REF}).')
    parser.add_argument('ref_b', help=f'Second git ref (tag, branch, SHA, or {WORKTREE_REF}).')
    parser.add_argument(
        '--fasta', type=Path, required=True,
        help='Transcript FASTA, indexed. Must be named *.fa, since the tool '
             'derives the index path as <fasta>.with_suffix(".fa.fai").'
    )
    parser.add_argument(
        '--bam', type=Path, action='append', required=True, metavar='BAM',
        help='BAM to profile. Repeat for multiple BAMs (max 8).'
    )
    parser.add_argument('--args-a', default='', help='Extra CLI arguments for ref_a.')
    parser.add_argument('--args-b', default='', help='Extra CLI arguments for ref_b.')
    parser.add_argument('--label-a', help='Legend label for ref_a (default: the ref plus its extra args).')
    parser.add_argument('--label-b', help='Legend label for ref_b (default: the ref plus its extra args).')
    parser.add_argument(
        '--outdir', type=Path, default=Path('gc_bias_comparison'),
        help='Directory for the TSVs, the manifest and the HTML (default: gc_bias_comparison).'
    )
    parser.add_argument('--theme', choices=['light', 'dark'], default='light')
    parser.add_argument('--repo', type=Path, default=Path('.'), help='Repository to take the refs from.')
    parser.add_argument(
        '--worktree-dir', type=Path,
        help='Where to cache the per-ref worktrees (default: under the system temp dir).'
    )
    parser.add_argument('--force', action='store_true', help='Re-run even when the TSV already exists.')
    parser.add_argument(
        '--clean-worktrees', action='store_true',
        help='Remove the worktrees on exit instead of keeping them for reuse.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if shutil.which('uv') is None:
        die('uv is not on PATH; it is needed to build an environment per ref')
    if shutil.which('git') is None:
        die('git is not on PATH')

    fasta = args.fasta.resolve()
    if not fasta.exists():
        die(f'FASTA not found: {fasta}')
    faidx = fasta.with_suffix('.fa.fai')
    if not faidx.exists():
        die(f'FASTA index not found: {faidx} (the tool derives this path from the FASTA name)')

    bams = [bam.resolve() for bam in args.bam]
    for bam in bams:
        if not bam.exists():
            die(f'BAM not found: {bam}')

    bam_names = [bam.stem for bam in bams]
    duplicates = {name for name in bam_names if bam_names.count(name) > 1}
    if duplicates:
        die(f"BAM file stems must be unique; repeated: {', '.join(sorted(duplicates))}")
    if len(bams) > len(SERIES_COLORS[args.theme]):
        die(f'at most {len(SERIES_COLORS[args.theme])} BAMs per plot; split the run')

    repo = Path(git(args.repo, 'rev-parse', '--show-toplevel'))
    git(repo, 'worktree', 'prune')

    extra = [shlex.split(args.args_a), shlex.split(args.args_b)]
    refs = [args.ref_a, args.ref_b]
    labels = [
        args.label_a or default_label(refs[0], extra[0]),
        args.label_b or default_label(refs[1], extra[1]),
    ]
    if labels[0] == labels[1]:
        labels = [f'{labels[0]} [A]', f'{labels[1]} [B]']

    shas = [resolve_ref(repo, ref) for ref in refs]
    if shas[0] == shas[1] and extra[0] == extra[1]:
        die(f'both refs resolve to {shas[0][:12]} with identical arguments; nothing to compare')

    outdir = args.outdir.resolve()
    tsv_dir = outdir / 'tsv'
    tsv_dir.mkdir(parents=True, exist_ok=True)

    worktree_root = args.worktree_dir.resolve() if args.worktree_dir else (
        Path(tempfile.gettempdir()) / 'rna_gc_bias_worktrees' / repo.name
    )

    worktrees = []
    frames = []
    manifest = []
    for index, (ref, sha, label, extra_args) in enumerate(zip(refs, shas, labels, extra)):
        worktree = ensure_worktree(repo, sha, worktree_root)
        worktrees.append(worktree)
        package = detect_package(worktree)
        note(f'{label}: {sha[:12]} -> {package} at {worktree}')

        run_slug = f'{"ab"[index]}_{slugify(label)}'
        for bam, bam_name in zip(bams, bam_names):
            tsv = tsv_dir / f'{run_slug}__{bam_name}.tsv'
            if tsv.exists() and not args.force:
                note(f'reusing {tsv} (pass --force to re-run)')
            else:
                run_tool(worktree, package, fasta, bam, extra_args, tsv)
            frames.append(load_profile(tsv, bam_name, label))
            manifest.append({
                'run': label, 'ref': ref, 'sha': sha, 'package': package,
                'extra_args': ' '.join(extra_args), 'bam': str(bam), 'tsv': str(tsv),
            })

    pl.DataFrame(manifest).write_csv(outdir / 'runs.tsv', separator='\t')

    title = (
        f'GC bias — {labels[0]} (solid) vs {labels[1]} (dashed)'
        f'<br><span style="font-size:13px">{len(bams)} BAM(s), colour by sample</span>'
    )
    figure = build_figure(pl.concat(frames), bam_names, labels, args.theme, title)

    html = outdir / 'gc_bias.html'
    write_report(figure, html, args.theme)

    if args.clean_worktrees:
        for worktree in dict.fromkeys(worktrees):
            git(repo, 'worktree', 'remove', '--force', str(worktree))
    else:
        note(f'worktrees kept for reuse: {worktree_root}')

    note(f'profiles:  {tsv_dir}')
    note(f'manifest:  {outdir / "runs.tsv"}')
    note(f'plot:      {html}')


if __name__ == '__main__':
    main()
