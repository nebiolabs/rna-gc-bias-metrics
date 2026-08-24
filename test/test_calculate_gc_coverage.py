import os
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from rna_gc_bias_metrics.calculate_gc_coverage import (
    load_sequences,
    load_bam,
    expand_cigar,
    bam_to_bedgraph,
    get_binned_coverage,
    get_bin_gc,
    calculate_gc_coverage,
    calculate_gc_pct_coverage,
    calculate_gc_pct_frequency,
)

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
BAM = os.path.join(FIXTURES, 'bam', 'test.bam')
FASTA = os.path.join(FIXTURES, 'fasta', 'test_gencode_v33.fa')
FAIDX = os.path.join(FIXTURES, 'fasta', 'test_gencode_v33.fa.fai')

# Same four transcripts and byte-identical sequences as FASTA, but with Ensembl
# style descriptions on the headers -- for TestReferenceNameNormalization.
DESC_FASTA = os.path.join(FIXTURES, 'fasta', 'test_gencode_v33_with_descriptions.fa')
DESC_FAIDX = os.path.join(FIXTURES, 'fasta', 'test_gencode_v33_with_descriptions.fa.fai')

TRANSCRIPTS = ['ENST00000227525.8', 'ENST00000536171.1', 'ENST00000540280.1', 'ENST00000438571.5']
TRANSCRIPT_LENGTHS = {'ENST00000227525.8': 2129, 'ENST00000536171.1': 1959, 'ENST00000540280.1': 724, 'ENST00000438571.5': 792}
BIN_BP = 100

# Shared reference for the exact bin-coordinate tests (see TestExactBinCoordinates):
# 20bp = two 10bp homopolymer blocks (bin 0 all-A/0% GC, bin 1 all-G/100% GC).
EXACT_COORD_FASTA = os.path.join(FIXTURES, 'fasta', 'test_exact_coord.fa')
EXACT_COORD_FAIDX = os.path.join(FIXTURES, 'fasta', 'test_exact_coord.fa.fai')
EXACT_COORD_BIN_BP = 10

# One single-read BAM per scenario, both against the shared reference above:
# END_COORD_BAM's read ends at a bin boundary; START_COORD_BAM's read starts at
# the first G (start of bin 1).
END_COORD_BAM = os.path.join(FIXTURES, 'bam', 'test_exact_end_coord.bam')
START_COORD_BAM = os.path.join(FIXTURES, 'bam', 'test_exact_start_coord.bam')

# Overlapping pair on the shared exact-coord reference (chrDemo): left mate spans
# 1-based 1-10, right mate 6-15, overlap 6-10 -- for TestExactBinCoordinates.
PAIR_OVERLAP_BAM = os.path.join(FIXTURES, 'bam', 'test_pair_overlap.bam')

# 120bp reference (chrPair) carrying one read pair per dedup scenario at a
# distinct locus -- for TestPairDeduplication.
PAIR_DEDUP_FASTA = os.path.join(FIXTURES, 'fasta', 'test_pair_dedup.fa')
PAIR_DEDUP_FAIDX = os.path.join(FIXTURES, 'fasta', 'test_pair_dedup.fa.fai')
PAIR_DEDUP_BAM = os.path.join(FIXTURES, 'bam', 'test_pair_dedup.bam')


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sequences_and_faidx():
    return load_sequences(FASTA, FAIDX)


@pytest.fixture(scope="module")
def sequences(sequences_and_faidx):
    return sequences_and_faidx[0]


@pytest.fixture(scope="module")
def faidx(sequences_and_faidx):
    return sequences_and_faidx[1]


@pytest.fixture(scope="module")
def transcript_categories(faidx):
    return faidx.collect_schema()['rname'].categories


@pytest.fixture(scope="module")
def bam_df(transcript_categories):
    return load_bam(BAM, transcript_categories=transcript_categories)


@pytest.fixture(scope="module")
def expanded(bam_df):
    return expand_cigar(bam_df)


@pytest.fixture(scope="module")
def bg(expanded):
    return bam_to_bedgraph(expanded)


@pytest.fixture(scope="module")
def bin_cov(bg, faidx):
    return get_binned_coverage(bg, BIN_BP, faidx)


@pytest.fixture(scope="module")
def bin_cov_with_gc(bin_cov, sequences):
    return get_bin_gc(bin_cov, BIN_BP, sequences)


# ---------------------------------------------------------------------------
# 1. load_sequences
# ---------------------------------------------------------------------------
class TestLoadSequences:
    def test_faidx_shape(self, faidx):
        assert faidx.collect().shape == (4, 2)

    def test_faidx_columns_and_dtypes(self, faidx):
        df = faidx.collect()
        assert df.columns == ['rname', 'length']
        assert df['rname'].dtype == pl.Enum(TRANSCRIPTS)
        assert df['length'].dtype == pl.UInt32

    def test_faidx_transcript_names(self, faidx):
        names = faidx.collect()['rname'].cast(pl.String).to_list()
        assert names == TRANSCRIPTS

    def test_faidx_transcript_lengths(self, faidx):
        df = faidx.collect()
        for row in df.iter_rows(named=True):
            assert row['length'] == TRANSCRIPT_LENGTHS[row['rname']]

    def test_sequences_shape(self, sequences):
        assert sequences.collect().shape == (4, 2)

    def test_sequences_columns_and_dtypes(self, sequences):
        df = sequences.collect()
        assert df.columns == ['rname', 'seq']
        assert df['rname'].dtype == pl.Enum(TRANSCRIPTS)
        assert df['seq'].dtype == pl.String

    def test_sequence_lengths_match_faidx(self, sequences):
        df = sequences.collect()
        for row in df.iter_rows(named=True):
            assert len(row['seq']) == TRANSCRIPT_LENGTHS[row['rname']]


# ---------------------------------------------------------------------------
# 3. load_bam
# ---------------------------------------------------------------------------
class TestLoadBam:
    def test_shape(self, bam_df):
        assert bam_df.collect().shape == (4, 10)

    def test_columns(self, bam_df):
        assert set(bam_df.collect().columns) == set(
            ['QNAME', 'FLAG', 'RNAME', 'POS', 'CIGAR', 'MPOS', 'ISIZE', 'FPAIRED', 'FREVERSE', 'side']
        )

    def test_rnames(self, bam_df):
        rnames = bam_df.collect()['RNAME'].cast(pl.String).sort().to_list()
        assert rnames == ['ENST00000227525.8', 'ENST00000227525.8', 'ENST00000438571.5', 'ENST00000438571.5']

    def test_positions(self, bam_df):
        pos = bam_df.collect().sort('RNAME', 'POS')['POS'].to_list()
        assert pos == [1451, 1475, 36, 233]

    def test_side_counts(self, bam_df):
        counts = bam_df.collect()['side'].value_counts().sort('side')
        assert counts['count'].to_list() == [2, 2]
        assert counts['side'].to_list() == ['R1', 'R2']

    def test_freverse_counts(self, bam_df):
        counts = bam_df.collect()['FREVERSE'].value_counts().sort('FREVERSE')
        assert counts['count'].to_list() == [2, 2]

    def test_faidx_transcripts_absent_from_bam_do_not_raise(self, bam_df, transcript_categories):
        """The check asserts BAM ⊆ faidx only -- do not make it symmetric."""
        bam_names = set(bam_df.collect()['RNAME'].cast(pl.String))
        assert bam_names < set(transcript_categories)


# ---------------------------------------------------------------------------
# 4. expand_cigar
# ---------------------------------------------------------------------------
class TestExpandCigar:
    def test_shape(self, expanded):
        assert expanded.collect().shape[0] == 4

    def test_has_expected_columns(self, expanded):
        cols = expanded.collect().columns
        for c in ['cigar_start', 'cigar_end', 'cigar_end_dedup', 'is_left_mate']:
            assert c in cols

    def test_mate_overlap_dedup(self, expanded):
        """First read (POS=1451, left mate) should be trimmed from cigar_end=1526 to cigar_end_dedup=1475 (MPOS)."""
        df = expanded.collect().sort('RNAME', 'POS')
        row = df.row(0, named=True)
        assert row['POS'] == 1451
        assert row['is_left_mate'] is True
        assert row['cigar_end'] == 1526
        assert row['cigar_end_dedup'] == 1475

    def test_non_overlapping_reads_unchanged(self, expanded):
        """Reads that don't overlap their mate should have cigar_end == cigar_end_dedup."""
        df = expanded.collect().sort('RNAME', 'POS')
        for i in [1, 2, 3]:
            row = df.row(i, named=True)
            assert row['cigar_end'] == row['cigar_end_dedup']


# ---------------------------------------------------------------------------
# 5. bam_to_bedgraph
# ---------------------------------------------------------------------------
class TestBamToBedgraph:
    def test_shape(self, bg):
        assert bg.collect().shape == (4, 4)

    def test_columns(self, bg):
        assert bg.collect().columns == ['rname', 'start', 'end', 'depth']

    def test_exact_values(self, bg):
        df = bg.collect().sort('rname', 'start')
        expected = pl.DataFrame({
            'rname': ['ENST00000227525.8', 'ENST00000227525.8', 'ENST00000438571.5', 'ENST00000438571.5'],
            'start': [1450, 1474, 35, 232],
            'end':   [1474, 1549, 110, 307],
            'depth': [1, 1, 1, 1],
        }).cast({
            'rname': pl.Enum(TRANSCRIPTS),
            'start': pl.UInt32,
            'end': pl.UInt32,
            'depth': pl.UInt32,
        }).sort('rname', 'start')
        assert_frame_equal(df, expected)


# ---------------------------------------------------------------------------
# 6. get_binned_coverage
# ---------------------------------------------------------------------------
class TestGetBinnedCoverage:
    def test_shape(self, bin_cov):
        assert bin_cov.collect().shape == (6, 3)

    def test_columns(self, bin_cov):
        assert bin_cov.collect().columns == ['rname', 'bin_start', 'depth_fractional']

    def test_exact_values(self, bin_cov):
        df = bin_cov.collect().sort('rname', 'bin_start')
        rnames = df['rname'].cast(pl.String).to_list()
        assert rnames == [
            'ENST00000227525.8', 'ENST00000227525.8',
            'ENST00000438571.5', 'ENST00000438571.5', 'ENST00000438571.5', 'ENST00000438571.5',
        ]
        assert df['bin_start'].to_list() == [1400.0, 1500.0, 0.0, 100.0, 200.0, 300.0]
        expected = [0.5, 0.49, 0.65, 0.1, 0.68, 0.07]
        for actual, exp in zip(df['depth_fractional'].to_list(), expected):
            assert abs(actual - exp) < 1e-10

    def test_mismatched_rname_dtype_raises(self, faidx):
        bg_categorical = bam_to_bedgraph(expand_cigar(load_bam(BAM)))
        with pytest.raises(ValueError, match='dtype mismatch'):
            get_binned_coverage(bg_categorical, BIN_BP, faidx)


# ---------------------------------------------------------------------------
# 7. get_bin_gc
# ---------------------------------------------------------------------------
#  Per-bin GC fraction of the two covered transcripts, in sorted order (the enum
#  category order puts ENST00000227525.8 first). GC depends only on the sequence
#  and the bin boundaries, never on coverage, so this covers all 30 emitted bins.
GC_FRAC = [
    #  ENST00000227525.8, 2129bp -> 22 bins, the last a 29bp remainder
    0.62, 0.53, 0.56, 0.56, 0.62, 0.65, 0.65, 0.71, 0.67, 0.61, 0.56,
    0.58, 0.53, 0.60, 0.59, 0.47, 0.53, 0.59, 0.67, 0.50, 0.33, 7 / 29,
    #  ENST00000438571.5, 792bp -> 8 bins, the last a 92bp remainder
    0.77, 0.68, 0.65, 0.70, 0.48, 0.51, 0.28, 40 / 92,
]

#  The bins that carry coverage in the test.bam fixture, in sorted order. Every
#  other bin of these two transcripts is emitted with zero depth; the remaining
#  two transcripts have no coverage at all and are absent entirely.
#  Only the depth columns need this split -- an assertion about a
#  coverage-independent column (gc_frac, gc_frac_rounded) belongs on all 30 bins.
COVERED_BINS = [
    ('ENST00000227525.8', 1400), ('ENST00000227525.8', 1500),
    ('ENST00000438571.5', 0), ('ENST00000438571.5', 100),
    ('ENST00000438571.5', 200), ('ENST00000438571.5', 300),
]


def _split_on_coverage(df):
    """(rows that received coverage, rows that did not), each in sorted order."""
    keys = {f'{rname}:{bin_start}' for rname, bin_start in COVERED_BINS}
    is_covered = pl.concat_str(
        pl.col('rname').cast(pl.String), pl.col('bin_start'), separator=':'
    ).is_in(keys)
    df = df.sort('rname', 'bin_start')
    return df.filter(is_covered), df.filter(~is_covered)


class TestGetBinGc:
    def test_shape(self, bin_cov_with_gc):
        """All 22 bins of the 2129bp transcript and all 8 of the 792bp one."""
        assert bin_cov_with_gc.collect().shape == (30, 6)

    def test_columns(self, bin_cov_with_gc):
        assert bin_cov_with_gc.collect().columns == [
            'rname', 'bin_start', 'depth_fractional', 'gc_frac', 'depth_normalized', 'gc_frac_rounded'
        ]

    def test_gc_frac_values(self, bin_cov_with_gc):
        df = bin_cov_with_gc.collect().sort('rname', 'bin_start')
        assert df['gc_frac'].to_list() == GC_FRAC

    def test_gc_frac_rounded_values(self, bin_cov_with_gc):
        """Only the two remainder bins have a GC fraction that is not already 2dp,
        so they are the only ones that exercise the rounding at all: 7/29 -> 0.24
        and 40/92 -> 0.43."""
        df = bin_cov_with_gc.collect().sort('rname', 'bin_start')
        assert df['gc_frac_rounded'].to_list() == [
            0.62, 0.53, 0.56, 0.56, 0.62, 0.65, 0.65, 0.71, 0.67, 0.61, 0.56,
            0.58, 0.53, 0.60, 0.59, 0.47, 0.53, 0.59, 0.67, 0.50, 0.33, 0.24,
            0.77, 0.68, 0.65, 0.70, 0.48, 0.51, 0.28, 0.43,
        ]

    def test_depth_normalized_values(self, bin_cov_with_gc):
        """Normalized against the transcript mean over *all* bins, so a covered
        bin's value scales with how much of its transcript went uncovered: x11 for
        ENST00000227525.8 (2 of 22 bins covered), x2 for ENST00000438571.5 (4 of 8).
        """
        covered, uncovered = _split_on_coverage(bin_cov_with_gc.collect())
        expected = [11.111111, 10.888889, 3.466667, 0.533333, 3.626667, 0.373333]
        for actual, exp in zip(covered['depth_normalized'].to_list(), expected):
            assert abs(actual - exp) < 1e-4
        #  Pins the remaining 24 bins, so both depth columns are asserted in full
        assert (uncovered['depth_normalized'] == 0.0).all()
        assert (uncovered['depth_fractional'] == 0.0).all()

    def test_uncovered_bins_are_emitted_as_zero(self, bin_cov_with_gc):
        """A bin with no reads is reported with zero depth rather than dropped, even
        when other bins of the same transcript are covered. ENST00000438571.5 is
        covered only at its head (bins 0-300), while ENST00000227525.8 is covered
        only in the middle (1400, 1500) and so must emit zeros on both sides."""
        df = bin_cov_with_gc.collect()
        for rname, expected_bins in [
            ('ENST00000227525.8', list(range(0, 2200, 100))),
            ('ENST00000438571.5', list(range(0, 800, 100))),
        ]:
            transcript = df.filter(pl.col('rname').cast(pl.String) == rname).sort('bin_start')
            assert transcript['bin_start'].to_list() == expected_bins
            assert transcript['gc_frac'].null_count() == 0

    def test_transcripts_without_any_coverage_are_excluded(self, bin_cov_with_gc):
        """Zero-coverage transcripts stay out; their depth_normalized would be 0/0."""
        rnames = bin_cov_with_gc.collect()['rname'].cast(pl.String).unique().to_list()
        assert sorted(rnames) == ['ENST00000227525.8', 'ENST00000438571.5']


# ---------------------------------------------------------------------------
# 8. calculate_gc_coverage (end-to-end)
# ---------------------------------------------------------------------------
class TestCalculateGcCoverage:
    def test_shape(self, sequences, faidx):
        result = calculate_gc_coverage(BAM, sequences, faidx, fixed_length_bin_bp=BIN_BP).collect()
        assert result.shape == (30, 6)

    def test_columns(self, sequences, faidx):
        result = calculate_gc_coverage(BAM, sequences, faidx, fixed_length_bin_bp=BIN_BP).collect()
        assert result.columns == [
            'rname', 'bin_start', 'depth_fractional', 'gc_frac', 'depth_normalized', 'gc_frac_rounded'
        ]

    def test_matches_stepwise(self, sequences, faidx, bin_cov_with_gc):
        """End-to-end result should match the stepwise pipeline."""
        e2e = calculate_gc_coverage(BAM, sequences, faidx, fixed_length_bin_bp=BIN_BP).collect()
        stepwise = bin_cov_with_gc.collect()
        assert_frame_equal(
            e2e.sort('rname', 'bin_start'),
            stepwise.sort('rname', 'bin_start')
        )


# ---------------------------------------------------------------------------
# 9. calculate_gc_pct_coverage
# ---------------------------------------------------------------------------
#  Every rounded GC fraction present across the 30 bins, in sorted order.
GC_FRACTIONS = [
    0.24, 0.28, 0.33, 0.43, 0.47, 0.48, 0.50, 0.51, 0.53, 0.56, 0.58,
    0.59, 0.60, 0.61, 0.62, 0.65, 0.67, 0.68, 0.70, 0.71, 0.77,
]


class TestCalculateGcPctCoverage:
    def test_shape(self, bin_cov_with_gc):
        result = calculate_gc_pct_coverage(bin_cov_with_gc).collect()
        assert result.shape == (21, 2)

    def test_columns(self, bin_cov_with_gc):
        result = calculate_gc_pct_coverage(bin_cov_with_gc).collect()
        assert result.columns == ['gc_frac_rounded', 'depth_normalized']

    def test_exact_values(self, bin_cov_with_gc):
        """GC fractions reached by no read average to zero rather than vanishing.
        0.59 and 0.65 each mix a covered bin with uncovered ones, so they average
        below the covered bin's own normalized depth."""
        result = calculate_gc_pct_coverage(bin_cov_with_gc).collect().sort('gc_frac_rounded')
        assert result['gc_frac_rounded'].to_list() == GC_FRACTIONS
        expected_depths = [
            0.0, 0.0, 0.0, 0.0, 10.888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            5.555556, 0.0, 0.0, 0.0, 1.208889, 0.0, 0.533333, 0.373333, 0.0, 3.466667,
        ]
        for actual, exp in zip(result['depth_normalized'].to_list(), expected_depths):
            assert abs(actual - exp) < 1e-4


# ---------------------------------------------------------------------------
# 10. calculate_gc_pct_frequency
# ---------------------------------------------------------------------------
class TestCalculateGcPctFrequency:
    def test_shape(self, bin_cov_with_gc):
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect()
        assert result.shape == (21, 2)

    def test_columns(self, bin_cov_with_gc):
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect()
        assert result.columns == ['gc_frac_rounded', 'count']

    def test_exact_counts(self, bin_cov_with_gc):
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect().sort('gc_frac_rounded')
        assert result['gc_frac_rounded'].to_list() == GC_FRACTIONS
        assert result['count'].to_list() == [
            1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 2, 1, 1, 2, 3, 2, 1, 1, 1, 1
        ]

    def test_counts_every_bin_of_the_covered_transcripts(self, bin_cov_with_gc):
        """The counts are bins interrogated, not bins that happened to get reads."""
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect()
        assert result['count'].sum() == 30


# ---------------------------------------------------------------------------
# 11. Exact bin-coordinate handling
# ---------------------------------------------------------------------------
class TestExactBinCoordinates:
    """Exact placement of a read's coverage relative to bin boundaries, on a
    purpose-built reference where bin 0 is 0% GC ("AAAAAAAAAA") and bin 1 is 100%
    GC ("GGGGGGGGGG"), with a 10 bp bin size -- so any coverage landing in the
    wrong bin surfaces as spurious GC signal. Two single-read fixtures are used:
    one at 1-based POS=1 covering exactly 0-based [0, 10) (all of bin 0, confirmed
    by `bedtools bamtobed`, which reports `chrDemo 0 10`), and one at POS=11 that
    starts exactly at bin 1.
    """

    @pytest.fixture(scope="class")
    def seqs_faidx(self):
        return load_sequences(EXACT_COORD_FASTA, EXACT_COORD_FAIDX)

    @pytest.fixture(scope="class")
    def pipeline(self, seqs_faidx):
        """Returns (bedgraph LazyFrame, collected+sorted get_bin_gc result)."""
        sequences, faidx = seqs_faidx
        transcript_categories = faidx.collect_schema()['rname'].categories
        bam = load_bam(END_COORD_BAM, transcript_categories=transcript_categories)
        bg = bam_to_bedgraph(expand_cigar(bam))
        bin_cov = get_binned_coverage(bg, EXACT_COORD_BIN_BP, faidx)
        bin_cov_with_gc = get_bin_gc(bin_cov, EXACT_COORD_BIN_BP, sequences).collect().sort('bin_start')
        return bg, bin_cov_with_gc

    def test_read_fills_its_own_bin(self, pipeline):
        """Bin 0 -- the read's true span, 0% GC -- receives the read's full depth."""
        _, bin_cov_with_gc = pipeline
        row = bin_cov_with_gc.filter(pl.col('bin_start') == 0).row(0, named=True)
        assert row['gc_frac'] == 0.0
        assert abs(row['depth_fractional'] - 1.0) < 1e-12

    def test_bedgraph_end_is_zero_based_exclusive(self, pipeline):
        """bedtools bamtobed reports the read as 0-based [0, 10), so the bedgraph
        end must be 10."""
        bg, _ = pipeline
        row = bg.filter(pl.col('start') == 0).collect().row(0, named=True)
        assert row['end'] == 10

    def test_no_coverage_leaks_past_read_span(self, pipeline):
        """The read lies entirely in bin 0, so the 100% GC bin at bin_start 10 is
        reported with zero depth -- present (chrDemo has coverage elsewhere) but
        carrying none of it."""
        _, bin_cov_with_gc = pipeline
        row = bin_cov_with_gc.filter(pl.col('bin_start') == 10).row(0, named=True)
        assert row['gc_frac'] == 1.0
        assert row['depth_fractional'] == 0.0

    @pytest.fixture(scope="class")
    def start_coord_bin_cov_with_gc(self, seqs_faidx):
        """get_bin_gc result for a single 10bp read starting at the first G (bin 1)."""
        sequences, faidx = seqs_faidx
        transcript_categories = faidx.collect_schema()['rname'].categories
        bam = load_bam(START_COORD_BAM, transcript_categories=transcript_categories)
        bin_cov = get_binned_coverage(bam_to_bedgraph(expand_cigar(bam)), EXACT_COORD_BIN_BP, faidx)
        return get_bin_gc(bin_cov, EXACT_COORD_BIN_BP, sequences).collect().sort('bin_start')

    def test_read_starting_at_bin_boundary_fills_that_bin(self, start_coord_bin_cov_with_gc):
        """A 10bp read starting at the first G (1-based POS=11 = start of bin 1)
        fully covers bin 1 -- the 100% GC bin at bin_start 10 -- with the read's
        full depth."""
        row = start_coord_bin_cov_with_gc.filter(pl.col('bin_start') == 10).row(0, named=True)
        assert row['gc_frac'] == 1.0
        assert abs(row['depth_fractional'] - 1.0) < 1e-12

    @pytest.fixture(scope="class")
    def overlap_pair(self, seqs_faidx):
        """(expand_cigar, get_bin_gc) for an overlapping pair on chrDemo."""
        sequences, faidx = seqs_faidx
        transcript_categories = faidx.collect_schema()['rname'].categories
        bam = load_bam(PAIR_OVERLAP_BAM, transcript_categories=transcript_categories)
        expanded = expand_cigar(bam).collect()
        bin_cov = get_binned_coverage(bam_to_bedgraph(expand_cigar(bam)), EXACT_COORD_BIN_BP, faidx)
        bin_cov_with_gc = get_bin_gc(bin_cov, EXACT_COORD_BIN_BP, sequences).collect()
        return expanded, bin_cov_with_gc

    def test_overlapping_pair_left_mate_trimmed(self, overlap_pair):
        """The left mate's run is trimmed to the mate start (cigar_end_dedup == MPOS)
        while the right mate is kept whole, so the overlap is ceded to one mate."""
        expanded, _ = overlap_pair
        left = expanded.filter(pl.col('is_left_mate')).row(0, named=True)
        assert left['cigar_end'] == 11
        assert left['cigar_end_dedup'] == left['MPOS'] == 6
        right = expanded.filter(~pl.col('is_left_mate')).row(0, named=True)
        assert right['cigar_end_dedup'] == right['cigar_end'] == 16

    def test_overlapping_pair_counts_each_base_once(self, overlap_pair):
        """The pair covers 15 distinct reference bases (1-based 1-15), so the total
        deduplicated coverage must equal 15 bases == 1.5 bins at 10bp."""
        _, bin_cov_with_gc = overlap_pair
        assert bin_cov_with_gc['depth_fractional'].sum() == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 12. Paired-end deduplication (structural + defect cases)
# ---------------------------------------------------------------------------
class TestPairDeduplication:
    """Mate-overlap dedup in expand_cigar across CIGAR shapes and edge cases, on a
    120bp reference (chrPair) carrying one read pair per scenario at a distinct
    locus (see the test_pair_dedup fixture)."""

    @pytest.fixture(scope="class")
    def seqs_faidx(self):
        return load_sequences(PAIR_DEDUP_FASTA, PAIR_DEDUP_FAIDX)

    @pytest.fixture(scope="class")
    def dedup_expanded(self, seqs_faidx):
        _, faidx = seqs_faidx
        transcript_categories = faidx.collect_schema()['rname'].categories
        bam = load_bam(PAIR_DEDUP_BAM, transcript_categories=transcript_categories)
        return expand_cigar(bam).collect()

    @pytest.fixture(scope="class")
    def dedup_bg(self, seqs_faidx):
        _, faidx = seqs_faidx
        transcript_categories = faidx.collect_schema()['rname'].categories
        bam = load_bam(PAIR_DEDUP_BAM, transcript_categories=transcript_categories)
        return bam_to_bedgraph(expand_cigar(bam)).collect()

    def test_left_run_beginning_at_mate_start_is_dropped(self, dedup_expanded):
        """A left-mate CIGAR run beginning at or after the mate start is dropped (the
        right mate is assumed to cover that region): the 5M5D5M left mate keeps only
        its first run, and the run past MPOS is gone."""
        left = dedup_expanded.filter((pl.col('QNAME') == 'multirun') & pl.col('is_left_mate'))
        assert left.height == 1
        assert left['cigar_start'].to_list() == [1]

    def test_left_mate_ending_at_mate_start_not_trimmed(self, dedup_expanded):
        """A left mate whose run ends exactly at the mate start (cigar_end == MPOS) is
        not trimmed; the cap applies only when cigar_end strictly exceeds MPOS."""
        left = dedup_expanded.filter(
            (pl.col('QNAME') == 'boundary') & pl.col('is_left_mate')
        ).row(0, named=True)
        assert left['cigar_end_dedup'] == left['cigar_end'] == 51

    def test_equal_start_pair_not_double_counted(self, dedup_bg):
        """Fully-overlapping mates that start at the same position must count each
        base once (depth 1)."""
        equalstart = dedup_bg.filter(pl.col('start') == 70).row(0, named=True)
        assert equalstart['depth'] == 1

    @pytest.mark.xfail(strict=True, reason=(
        "Dovetail/read-through (left mate extends past the right mate's end): capping "
        "the left mate at MPOS discards its tail beyond the right mate. This is an "
        "accepted limitation -- such pairs are rare after adapter trimming and it is "
        "not planned for a fix; strict xfail flags any accidental behavior change."
    ))
    def test_dovetail_left_tail_retained(self, dedup_bg):
        """The left mate reaches 1-based reference base 110; its coverage past the
        right mate's end must be retained (max bedgraph end reaches 110)."""
        # Bound the upper end to the dovetail locus: the single-end read sits at
        # bedgraph start 114 and would otherwise be pulled into a bare start>=90
        # slice, making its end (119) the max instead of the dovetail tail.
        dovetail = dedup_bg.filter((pl.col('start') >= 90) & (pl.col('start') < 114))
        assert dovetail['end'].max() == 110

    def test_single_end_read_retained(self, dedup_expanded):
        """A single-end read (no mate, null MPOS) contributes its own coverage rather
        than being dropped, and is never treated as a left mate to trim."""
        singleend = dedup_expanded.filter(pl.col('QNAME') == 'singleend')
        assert singleend.height == 1
        row = singleend.row(0, named=True)
        assert row['is_left_mate'] is False
        assert row['cigar_end'] == row['cigar_end_dedup']  # untrimmed

    def test_single_end_read_full_depth(self, dedup_bg):
        """The single-end read (1-based 115-119) covers its full 5bp span at depth 1."""
        singleend = dedup_bg.filter(pl.col('start') == 114).row(0, named=True)
        assert singleend['end'] == 119
        assert singleend['depth'] == 1

    def test_nonproper_paired_mates_dropped(self, dedup_expanded):
        """Paired reads that are not properly paired -- a mate-unmapped read and a
        discordant (non-proper) mapped read -- are dropped in load_bam and never reach
        the expanded coverage, since they have no reliable mate overlap to deduplicate."""
        dropped = dedup_expanded.filter(
            pl.col('QNAME').is_in(['nonproper_paired_munmap', 'nonproper_paired_discordant'])
        )
        assert dropped.height == 0


# ---------------------------------------------------------------------------
# 13. Reference name normalization
# ---------------------------------------------------------------------------
class TestReferenceNameNormalization:
    """A reference name is the FASTA header up to the first whitespace.

    Aligners cannot store more than that (the SAM spec forbids whitespace in
    RNAME) and `samtools faidx` truncates identically, so a FASTA carrying
    descriptive headers must resolve to the same names as the BAM and .fai it
    is paired with. Without the truncation the descriptive header is not a
    member of the faidx-derived Enum and loading fails outright.

    DESC_FASTA holds the same four transcripts and byte-identical sequences as
    FASTA, so every result below must equal its plain-header counterpart.
    """

    @pytest.fixture(scope="class")
    def desc_sequences_and_faidx(self):
        return load_sequences(DESC_FASTA, DESC_FAIDX)

    def test_fasta_names_drop_description(self, desc_sequences_and_faidx):
        """The description after the first whitespace is not part of the name."""
        sequences = desc_sequences_and_faidx[0].collect()
        assert sequences['rname'].cast(pl.String).to_list() == TRANSCRIPTS

    def test_tab_terminates_name(self, desc_sequences_and_faidx):
        """Any whitespace ends the name, not just a space -- ENST00000540280.1's
        header is tab-separated from its description."""
        sequences = desc_sequences_and_faidx[0].collect()
        assert 'ENST00000540280.1' in sequences['rname'].cast(pl.String).to_list()

    def test_sequences_unaffected_by_truncation(self, desc_sequences_and_faidx):
        """Only the header is trimmed; sequence content is untouched."""
        sequences = desc_sequences_and_faidx[0].collect()
        for row in sequences.iter_rows(named=True):
            assert len(row['seq']) == TRANSCRIPT_LENGTHS[row['rname']]

    def test_faidx_matches_plain_header_fixture(self, desc_sequences_and_faidx, faidx):
        assert_frame_equal(desc_sequences_and_faidx[1].collect(), faidx.collect())

    def test_fasta_matches_plain_header_fixture(self, desc_sequences_and_faidx, sequences):
        assert_frame_equal(desc_sequences_and_faidx[0].collect(), sequences.collect())

    def test_end_to_end_matches_plain_header_fixture(self, desc_sequences_and_faidx, bin_cov_with_gc):
        desc_sequences, desc_faidx = desc_sequences_and_faidx
        result = calculate_gc_coverage(
            BAM, desc_sequences, desc_faidx, fixed_length_bin_bp=BIN_BP
        ).collect()
        assert_frame_equal(
            result.sort('rname', 'bin_start'),
            bin_cov_with_gc.collect().sort('rname', 'bin_start')
        )
