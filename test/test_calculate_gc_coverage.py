import os
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from gc_profile.calculate_gc_coverage import (
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

TRANSCRIPTS = ['ENST00000227525.8', 'ENST00000536171.1', 'ENST00000540280.1', 'ENST00000438571.5']
TRANSCRIPT_LENGTHS = {'ENST00000227525.8': 2129, 'ENST00000536171.1': 1959, 'ENST00000540280.1': 724, 'ENST00000438571.5': 792}
BIN_BP = 100


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
    return faidx.select(pl.col('rname').cat.get_categories()).collect()


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
        assert bam_df.collect().shape == (4, 9)

    def test_columns(self, bam_df):
        assert set(bam_df.collect().columns) == set(
            ['QNAME', 'FLAG', 'RNAME', 'POS', 'CIGAR', 'MPOS', 'ISIZE', 'FREVERSE', 'side']
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
            'end':   [1475, 1550, 111, 308],
            'depth': [1, 1, 1, 1],
        }).cast({
            'rname': pl.Enum(TRANSCRIPTS),
            'start': pl.Int64,
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
        expected = [0.51, 0.5, 0.65, 0.11, 0.68, 0.08]
        for actual, exp in zip(df['depth_fractional'].to_list(), expected):
            assert abs(actual - exp) < 1e-10


# ---------------------------------------------------------------------------
# 7. get_bin_gc
# ---------------------------------------------------------------------------
class TestGetBinGc:
    def test_shape(self, bin_cov_with_gc):
        assert bin_cov_with_gc.collect().shape == (6, 6)

    def test_columns(self, bin_cov_with_gc):
        assert bin_cov_with_gc.collect().columns == [
            'rname', 'bin_start', 'depth_fractional', 'gc_frac', 'depth_normalized', 'gc_frac_rounded'
        ]

    def test_gc_frac_values(self, bin_cov_with_gc):
        df = bin_cov_with_gc.collect().sort('rname', 'bin_start')
        assert df['gc_frac'].to_list() == [0.59, 0.47, 0.77, 0.68, 0.65, 0.70]

    def test_depth_normalized_values(self, bin_cov_with_gc):
        df = bin_cov_with_gc.collect().sort('rname', 'bin_start')
        values = df['depth_normalized'].to_list()
        expected = [1.009901, 0.990099, 1.710526, 0.289474, 1.789474, 0.210526]
        for actual, exp in zip(values, expected):
            assert abs(actual - exp) < 1e-4

    def test_gc_frac_rounded_values(self, bin_cov_with_gc):
        df = bin_cov_with_gc.collect().sort('rname', 'bin_start')
        assert df['gc_frac_rounded'].to_list() == [0.59, 0.47, 0.77, 0.68, 0.65, 0.70]


# ---------------------------------------------------------------------------
# 8. calculate_gc_coverage (end-to-end)
# ---------------------------------------------------------------------------
class TestCalculateGcCoverage:
    def test_shape(self, sequences, faidx):
        result = calculate_gc_coverage(BAM, sequences, faidx, fixed_length_bin_bp=BIN_BP).collect()
        assert result.shape == (6, 6)

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
class TestCalculateGcPctCoverage:
    def test_shape(self, bin_cov_with_gc):
        result = calculate_gc_pct_coverage(bin_cov_with_gc).collect()
        assert result.shape == (6, 2)

    def test_columns(self, bin_cov_with_gc):
        result = calculate_gc_pct_coverage(bin_cov_with_gc).collect()
        assert result.columns == ['gc_frac_rounded', 'depth_normalized']

    def test_exact_values(self, bin_cov_with_gc):
        result = calculate_gc_pct_coverage(bin_cov_with_gc).collect().sort('gc_frac_rounded')
        assert result['gc_frac_rounded'].to_list() == [0.47, 0.59, 0.65, 0.68, 0.70, 0.77]
        expected_depths = [0.990099, 1.009901, 1.789474, 0.289474, 0.210526, 1.710526]
        for actual, exp in zip(result['depth_normalized'].to_list(), expected_depths):
            assert abs(actual - exp) < 1e-4


# ---------------------------------------------------------------------------
# 10. calculate_gc_pct_frequency
# ---------------------------------------------------------------------------
class TestCalculateGcPctFrequency:
    def test_shape(self, bin_cov_with_gc):
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect()
        assert result.shape == (6, 2)

    def test_columns(self, bin_cov_with_gc):
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect()
        assert result.columns == ['gc_frac_rounded', 'count']

    def test_all_counts_are_one(self, bin_cov_with_gc):
        result = calculate_gc_pct_frequency(bin_cov_with_gc).collect().sort('gc_frac_rounded')
        assert result['gc_frac_rounded'].to_list() == [0.47, 0.59, 0.65, 0.68, 0.70, 0.77]
        assert result['count'].to_list() == [1, 1, 1, 1, 1, 1]
