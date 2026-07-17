# rna-gc-bias-metrics

Measure GC-content bias in RNA-seq coverage.

Given a transcriptome FASTA and a BAM of reads aligned to it, `gc-profile`
computes per-bin GC content and coverage depth, then summarizes how normalized
coverage varies with GC fraction. This reveals whether a library is
over- or under-covering GC-rich (or GC-poor) regions — a common artifact of
library preparation and sequencing chemistry.

## How it works

For each transcript the pipeline:

1. Reads the BAM (via [polars-bio](https://github.com/biodatageeks/polars-bio))
   and drops unmapped, secondary, and supplementary alignments.
2. Expands each read's CIGAR into the reference positions it covers, and
   deduplicates the overlap between paired-end mates so shared bases are counted
   once.
3. Collapses coverage into a per-range bedgraph, then redistributes it into
   fixed-length bins (default 100 bp), splitting partial-bin coverage
   proportionally.
4. Computes each bin's GC fraction from the transcript sequence and normalizes
   its depth against the transcript's mean bin depth (so `1.0` is the
   transcript average).
5. Aggregates mean normalized depth and bin counts per rounded GC fraction
   across the whole input.

## Requirements

- Python ≥ 3.12
- [numpy](https://numpy.org/), [polars](https://pola.rs/),
  [polars-bio](https://github.com/biodatageeks/polars-bio)

Dependencies are managed with either [uv](https://docs.astral.sh/uv/) (PyPI) or
[pixi](https://pixi.sh/) (conda + PyPI). Pick whichever you already use.

## Installation

```bash
git clone git@github.com:nebiolabs/rna-gc-bias-metrics.git
cd rna-gc-bias-metrics

# with uv
uv sync

# or with pixi
pixi install
```

## Input requirements

- **Transcriptome FASTA** with **single-line (unwrapped)** sequences — one line
  of sequence per record. Wrapped FASTA is not parsed correctly.
- A FASTA index named `<name>.fa.fai` alongside it (created with
  `samtools faidx`).
- A **BAM aligned to that transcriptome** (each `RNAME` is a transcript ID). BAM
  reference names must match the FASTA names, or the run aborts with an error.

To prepare a wrapped FASTA:

```bash
seqkit seq -w 0 transcripts.fa > transcripts.unwrapped.fa   # unwrap sequences
samtools faidx transcripts.unwrapped.fa                      # create .fa.fai
```

## Usage

The tool runs as a module. Under uv, prefix commands with `uv run`; under pixi,
use `pixi run`.

```bash
uv run python -m gc_profile.calculate_gc_coverage \
    transcripts.fa \
    reads.bam \
    --report_bin_count_for_full_transcriptome \
    -o gc_profile.tsv
```

### Arguments

| Argument | Description |
|---|---|
| `fp_fasta` | Path to the transcriptome FASTA (indexed; see input requirements). |
| `fp_bam` | Path to the BAM of reads aligned to the transcriptome. |
| `-o`, `--outfp` | Output file path (default: stdout). |
| `--fixed_length_bin_bp` | Bin size in base pairs (default: `100`). |
| `--report_bin_count_for_full_transcriptome` | Also report, per GC fraction, the number of bins across *every* transcript in the FASTA (covered or not) — the background GC distribution to compare coverage against. |

### Output

A tab-separated table, one row per rounded GC fraction:

| Column | Meaning |
|---|---|
| `gc_fraction` | GC fraction of the bin, rounded to two decimals (0–1). |
| `mean_normalized_depth` | Mean within-transcript-normalized depth across covered bins at this GC fraction. `1.0` = transcript average; `> 1` enriched, `< 1` depleted. |
| `transcriptome_bin_count` | Number of covered bins at this GC fraction. |
| `transcriptome_bin_count_all_transcripts` | Bins at this GC fraction across all transcripts (only with `--report_bin_count_for_full_transcriptome`). |

Example (run against the bundled test fixtures):

```
gc_fraction  mean_normalized_depth  transcriptome_bin_count  transcriptome_bin_count_all_transcripts
0.47         0.9900990099009903     1                        1
0.59         1.0099009900990097     1                        3
0.65         1.7894736842105265     1                        3
0.68         0.28947368421052655    1                        1
0.70         0.21052631578947384    1                        1
0.77         1.7105263157894732     1                        1
```

(GC fractions with no covered bins have empty depth/count columns but still
appear when the full-transcriptome background is reported.)

## Python API

The pipeline steps are importable for use in scripts or notebooks:

```python
from gc_profile.calculate_gc_coverage import (
    load_sequences,
    calculate_gc_coverage,
    calculate_gc_pct_coverage,
)

sequences, faidx = load_sequences("transcripts.fa", "transcripts.fa.fai")
bin_cov_with_gc = calculate_gc_coverage("reads.bam", sequences, faidx, fixed_length_bin_bp=100)

# mean normalized depth per rounded GC fraction
per_gc_coverage = calculate_gc_pct_coverage(bin_cov_with_gc).collect()
```

`calculate_gc_coverage` returns a per-bin table (`rname`, `bin_start`,
`depth_fractional`, `gc_frac`, `depth_normalized`, `gc_frac_rounded`) as a Polars
`LazyFrame`; call `.collect()` to materialize it. `load_bam`, `expand_cigar`,
`bam_to_bedgraph`, `get_binned_coverage`, and `get_bin_gc` expose the individual
stages.

## Notes and limitations

- Coverage is deduplicated across paired-end mates. Fully mateless reads
  (single-end or mate-unmapped) and mate pairs that start at the same position
  are not yet handled, and read-through ("dovetail") mate tails past the mate's
  end are dropped.
- The bedgraph `end` coordinate carries a known off-by-one that can leak a small
  amount of coverage into the adjacent bin.

## Development

```bash
uv run pytest test/ -v      # with uv
pixi run test               # with pixi
```

Tests should pass under both workflows. See [AGENTS.md](AGENTS.md) for project
layout and details on keeping the two dependency ecosystems in sync.

## License

Distributed under the GNU Affero General Public License v3.0. See
[LICENSE.txt](LICENSE.txt).
