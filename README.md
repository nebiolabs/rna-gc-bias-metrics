# rna-gc-bias-metrics

Measure GC-content bias in RNA-seq coverage.

Given a transcriptome FASTA and a BAM of reads aligned to it, `rna-gc-bias-metrics`
computes per-bin GC content and coverage depth, then summarizes how normalized
coverage varies with GC fraction. This reveals whether a library is
over- or under-covering GC-rich (or GC-poor) regions — a common artifact of
library preparation and sequencing chemistry.

## How it works

For each transcript the pipeline:

1. Reads the BAM (via [polars-bio](https://github.com/biodatageeks/polars-bio))
   and drops unmapped, secondary, and supplementary alignments, as well as paired
   reads that are not properly paired (mate-unmapped or discordant mates). Single-end
   reads and proper pairs are kept.
2. Expands each read's CIGAR into the reference positions it covers, and
   deduplicates the overlap between properly paired mates so shared bases are
   counted once.
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

- **Transcriptome FASTA**, either line-wrapped (as `samtools faidx`, GENCODE and
  Ensembl emit by default) or with single-line sequences. Both parse identically.
- A FASTA index named `<name>.fa.fai` alongside it (created with
  `samtools faidx`).
- A **BAM aligned to that transcriptome** (each `RNAME` is a transcript ID). BAM
  reference names must match the FASTA names, or the run aborts with an error.

Every reference name — in the FASTA, the `.fa.fai` and the BAM — is taken as the
header up to the **first whitespace**, so descriptive headers need no cleanup:

```
>ENST00000227525.8 cdna chromosome:GRCh38:12:6534517:6538371:1 gene:ENSG00000111640.15
```

is matched against a BAM `RNAME` of `ENST00000227525.8`. This mirrors the
aligners (the SAM spec forbids whitespace in `RNAME`, so only the leading word
survives alignment) and `samtools faidx`, which truncates the same way.

To index a FASTA:

```bash
samtools faidx transcripts.fa    # creates transcripts.fa.fai
```

## Usage

The tool runs as a module. Under uv, prefix commands with `uv run`; under pixi,
use `pixi run`.

```bash
uv run python -m rna_gc_bias_metrics.calculate_gc_coverage \
    transcripts.fa \
    reads.bam \
    --report_bin_count_for_full_transcriptome \
    -o gc_bias_profile.tsv
```

### Arguments

| Argument | Description |
|---|---|
| `fp_fasta` | Path to the transcriptome FASTA (indexed; see input requirements). |
| `fp_bam` | Path to the BAM of reads aligned to the transcriptome. |
| `-o`, `--outfp` | Output file path (default: stdout). |
| `--fixed_length_bin_bp` | Bin size in base pairs (default: `100`). |
| `--min_transcript_read_count` | Drop transcripts with fewer than this many fragments before calculating GC bias (default: no filter). |
| `--min_transcript_cpm` | Drop transcripts below this many fragments per million before calculating GC bias (default: no filter). |
| `--report_bin_count_for_full_transcriptome` | Also report, per GC fraction, the number of bins across *every* transcript in the FASTA, including transcripts with no coverage at all — the background GC distribution to compare coverage against. |

### Transcript depth filters

Because each transcript's bins are normalized against that transcript's own mean
bin depth, a transcript covered by one or two reads contributes a single wildly
enriched bin and a long tail of zeros — noise at whatever GC fractions it happens
to span. `--min_transcript_read_count` and `--min_transcript_cpm` drop such
transcripts before the profile is built. Both are optional; given together, a
transcript must clear both. Thresholds are inclusive, so
`--min_transcript_read_count 5` drops transcripts with 4 fragments or fewer.

Depth is counted in **fragments**, not alignment records: a proper pair counts
once, and so does a single-end read. The CPM denominator is the total number of
fragments the tool retains — mapped, non-secondary, non-supplementary, and either
single-end or properly paired — so CPM sums to 1,000,000 across transcripts. How
many transcripts survived is reported on stderr; if none do, the run fails with an
error naming the thresholds and the deepest transcript observed.

### Output

A tab-separated table, one row per rounded GC fraction. Every transcript with any
coverage contributes *all* of its bins, so bins no read reached count as zero depth
rather than being dropped — that is what makes a systematically uncovered GC range
visible as a depleted signal instead of an absent one.

A transcript excluded by a depth filter leaves the table entirely, zero bins
included, lowering `mean_normalized_depth` contributions and `transcriptome_bin_count`
alike. `transcriptome_bin_count_all_transcripts` is measured on the FASTA alone and
stays the whole-transcriptome background regardless of filtering.

| Column | Meaning |
|---|---|
| `gc_fraction` | GC fraction of the bin, rounded to two decimals (0–1). |
| `mean_normalized_depth` | Mean within-transcript-normalized depth at this GC fraction, over every bin of every transcript that has coverage somewhere. `1.0` = transcript average; `> 1` enriched, `< 1` depleted; `0.0` = no read reached any bin at this GC fraction. |
| `transcriptome_bin_count` | Number of bins at this GC fraction, counted over transcripts that have coverage somewhere — i.e. bins interrogated, not bins that got reads. |
| `transcriptome_bin_count_all_transcripts` | Bins at this GC fraction across *all* transcripts in the FASTA, whether or not they have any coverage (only with `--report_bin_count_for_full_transcriptome`). Always `>=` `transcriptome_bin_count`. |

Example (run against the bundled test fixtures — two of the four transcripts have
coverage, contributing 30 bins between them):

```
gc_fraction  mean_normalized_depth  transcriptome_bin_count  transcriptome_bin_count_all_transcripts
0.24         0.0                    1                        1
0.25                                                         1
0.28         0.0                    1                        1
0.33         0.0                    1                        2
0.36                                                         1
0.43         0.0                    1                        1
0.46                                                         1
0.47         10.888888888888891     1                        1
0.48         0.0                    1                        2
0.49                                                         2
0.5          0.0                    1                        1
0.51         0.0                    1                        1
0.52                                                         1
0.53         0.0                    3                        5
0.55                                                         1
0.56         0.0                    3                        4
0.57                                                         1
0.58         0.0                    1                        5
0.59         5.5555555555555545     2                        3
0.6          0.0                    1                        2
0.61         0.0                    1                        3
0.62         0.0                    2                        3
0.64                                                         2
0.65         1.208888888888889      3                        3
0.66                                                         2
0.67         0.0                    2                        3
0.68         0.5333333333333337     1                        1
0.7          0.37333333333333474    1                        1
0.71         0.0                    1                        2
0.77         3.466666666666665      1                        1
```

(GC fractions occurring only in transcripts with no coverage at all have empty
depth/count columns, but still appear when the full-transcriptome background is
reported.)

## Python API

The pipeline steps are importable for use in scripts or notebooks:

```python
from rna_gc_bias_metrics.calculate_gc_coverage import (
    load_sequences,
    calculate_gc_coverage,
    calculate_gc_pct_coverage,
)

sequences, faidx = load_sequences("transcripts.fa", "transcripts.fa.fai")
bin_cov_with_gc = calculate_gc_coverage(
    "reads.bam", sequences, faidx, fixed_length_bin_bp=100,
    min_transcript_read_count=None, min_transcript_cpm=None,
)

# mean normalized depth per rounded GC fraction
per_gc_coverage = calculate_gc_pct_coverage(bin_cov_with_gc).collect()
```

`calculate_gc_coverage` returns a per-bin table (`rname`, `bin_start`,
`depth_fractional`, `gc_frac`, `depth_normalized`, `gc_frac_rounded`) as a Polars
`LazyFrame`; call `.collect()` to materialize it. It holds every bin of every
transcript that has coverage somewhere, with `depth_fractional` 0 for bins no read
reached. `load_bam`, `expand_cigar`, `bam_to_bedgraph`, `get_binned_coverage`, and
`get_bin_gc` expose the individual stages — note that `get_binned_coverage` alone
emits only bins that received coverage; the zero bins are filled in by `get_bin_gc`.
`get_transcript_fragment_counts` and `filter_transcripts_by_depth` implement the
depth filters and can be applied to a `load_bam` frame directly.

## Notes and limitations

- Coverage is deduplicated across the overlap of properly paired mates so shared
  bases are counted once. Paired reads that are not properly paired (mate-unmapped or discordant
  mates) are dropped.
  Read-through ("dovetail") mate tails past the mate's end are dropped.

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
