# Agent Notes for rna_gc_bias_metrics

## Project layout

`rna_gc_bias_metrics` is a pure-Python package built with the `uv_build` PEP 517
backend. BAM files are loaded into Polars via `polars-bio` (`pb.scan_bam`).

- `pyproject.toml` — package definition. Plain PyPI dependencies only
  (`polars`, `polars-bio`, `numpy`); no `[tool.uv.sources]`.
- `src/rna_gc_bias_metrics/` — Python source. Use
  `from rna_gc_bias_metrics.calculate_gc_coverage import ...`, not the old
  flat-layout `from calculate_gc_coverage import ...`. The src layout was
  introduced in commit `f87d09c`; tests and any new callers must use the
  package-qualified import.
- `test/test_calculate_gc_coverage.py` — pytest suite.
- `pyproject.toml` + `uv.lock` — uv workflow.
- `pixi.toml` + `pixi.lock` — pixi workflow.

## BAM loading (`load_bam` in `calculate_gc_coverage.py`)

`polars-bio` is the BAM reader. A thin adapter inside `load_bam` keeps the
rest of the pipeline (and the tests) working on SAM-style column names/dtypes:

- `pb.scan_bam(path, use_zero_based=False)` — `use_zero_based=False` keeps
  `POS`/`MPOS` 1-based, matching the SAM spec.
- `polars-bio` has no read-time flag exclusion, so the old
  `exclude_flags=2308` (unmapped 4 + secondary 256 + supplementary 2048) is
  replicated as `.filter((pl.col('flags') & 2308) == 0)`.
- Columns are renamed `name→QNAME, flags→FLAG, chrom→RNAME, start→POS,
  cigar→CIGAR, mate_start→MPOS, template_length→ISIZE` and cast to the
  expected dtypes (`FLAG→UInt16`, `RNAME→Categorical`; the rest already match).
- There is **no downsampling** — `polars-bio` does not provide it and the
  feature was dropped along with the `--downsample` CLI flag.

## Reference names (`_normalize_rname`)

The FASTA, the `.fa.fai` and the BAM must agree on `rname`/`RNAME` for the joins
and the shared Enum category set to line up, but they do not agree natively: a
FASTA header keeps its full description, while aligners (the SAM spec forbids
whitespace in `RNAME`) and `samtools faidx` both keep only the leading word.

All three loaders therefore push their name through `_normalize_rname`, which
truncates at the first whitespace.

## Daily workflow

| Task | Command |
|---|---|
| Sync uv env | `uv sync` (the `dev` dependency-group syncs by default) |
| Run tests (uv) | `uv run pytest test/ -v` |
| Sync pixi env | `pixi install -e dev` |
| Run tests (pixi) | `pixi run test` |
| Interactive pixi shell | `pixi shell -e dev` |

Tests must pass under both workflows. They lock different ecosystems
(`uv.lock` is PyPI-only, `pixi.lock` is conda + PyPI) so divergence is
possible — re-run both after touching dependency declarations.

## Things to keep in sync

When a runtime dependency is added or removed, mirror it manually in:

1. `pyproject.toml` `[project.dependencies]` — uv / PyPI side.
2. `pixi.toml` — `[dependencies]` for conda-available packages,
   `[pypi-dependencies]` for PyPI-only ones (e.g. `polars-bio`).

There is no automated check yet.

## To do

- **Raise the memory ceiling for production runs.** `_load_fasta` used to keep only
  the first sequence line of each record, so a line-wrapped reference parsed to a
  fraction of its sequence. Parsing whole records takes peak RSS on the T2T
  transcriptome from ~2.4 GB to ~10.4 GB, so Nextflow `memory` directives sized
  against the old figure need raising before the next production rerun.

  The growth is mostly amplification, not the extra sequence itself. `get_bin_gc`
  explodes to one row per bin while carrying the full `seq` string, slicing
  `bin_seq` out only afterwards, so every transcript's whole sequence is
  duplicated once per bin. The factor is length-weighted --
  `mean_length * (1 + CV**2) / fixed_length_bin_bp`, ~37x on a transcriptome-like
  length distribution -- and truncated sequences yielded a single bin per
  transcript, hence a factor of 1. Measured at a tenth of T2T scale: 39 MB of
  sequence becomes 1.46 GB after the explode, against ~40 MB for the `bin_seq`
  slices themselves.

  Chunking each sequence into bin-sized pieces before the explode
  (`pl.col('seq').str.extract_all('.{1,N}')` -> `List[String]`) makes every base
  appear exactly once, so the explode duplicates nothing: 1.92 GB -> 0.50 GB peak
  at that scale, with identical `gc_frac`. `main()`'s `pl.collect_all` also holds
  this stage and `calculate_gc_pct_frequency_across_full_transcriptome` in memory
  concurrently. Keep any of this as its own change.
