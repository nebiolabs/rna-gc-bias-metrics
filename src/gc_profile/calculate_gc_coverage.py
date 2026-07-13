
import sys
from pathlib import Path
from os.path import join, splitext
import argparse
import numpy as np

import polars as pl

import polars_bio as pb


def _load_faidx(fp_faidx):
    '''Read a .fa.fai index into a lazy frame of transcript names and lengths.

    `rname` is cast to an Enum over the file's own name order so downstream
    joins and category alignment stay consistent.

    Example output:
        rname             | length
        ENST00000227525.8 | 2129
        ENST00000536171.1 | 1959
    '''
    faidx = pl.read_csv(
        fp_faidx,
        separator = '\t',
        has_header = False,
        new_columns = ['rname','length'],
        schema_overrides = {'length':pl.UInt32}
    ).select(
        pl.col('rname','length')
    )
    
    faidx = faidx.with_columns(
        pl.col('rname').cast(pl.Enum(categories=faidx.select('rname')))
    ).lazy()

    return faidx


def _load_fasta(fp_fasta, transcript_categories=None):
    '''Read an unwrapped (single-line-sequence) FASTA into a lazy frame.

    When `transcript_categories` is given, `rname` is cast to a matching Enum.

    Example output:
        rname             | seq
        ENST00000227525.8 | ATCCCGCCTTGCGCATGCGG…
        ENST00000536171.1 | CCTGGCAGACCCAGTCATGG…
    '''
    if transcript_categories is not None:
        schema_overrides = {'rname': pl.Enum(transcript_categories)}
    else:
        schema_overrides = None
    
    # Assumes single-line (unwrapped) sequences: each record parses to
    # [rname, seq, ''] where the trailing field is the newline before the next
    # '>'. truncate_ragged_lines drops that trailing field (polars >=1.34 errors
    # on it otherwise instead of silently dropping it).
    sequences = pl.scan_csv(
        fp_fasta,
        separator='\n',
        eol_char='>',
        new_columns = ['rname','seq'],
        has_header=False,
        schema_overrides = schema_overrides,
        infer_schema_length=1000000,
        truncate_ragged_lines=True
    ).select(
        pl.col('rname','seq')
    ).drop_nulls()

    return sequences


def load_sequences(fp_fasta, fp_faidx):
    '''Load the faidx and FASTA, aligning `rname` categories, and return both.'''
    faidx = _load_faidx(fp_faidx)
    transcript_categories = faidx.select(pl.col('rname').cat.get_categories()).collect()
    sequences = _load_fasta(fp_fasta, transcript_categories=transcript_categories)
    return sequences, faidx


def load_bam(fp_bam, transcript_categories=None):
    '''Read a BAM into a lazy frame with SAM-style columns, flags, and R1/R2 side.

    Scans `fp_bam` via polars-bio, drops unmapped/secondary/supplementary reads
    (exclude_flags=2308), renames columns to the SAM spec, and adds a `FREVERSE`
    boolean plus an R1/R2 `side` label. When `transcript_categories` is given,
    `RNAME` is validated against it (raising if the BAM references sequences
    absent from the FASTA index) and cast to a matching Enum.

    Example output:
        QNAME         | FLAG | RNAME             | POS  | MPOS | CIGAR | ISIZE | FREVERSE | side
        M05473…:12904 | 675  | ENST00000227525.8 | 1451 | 1475 | 75M   | 99    | false    | R2
        M05473…:12904 | 595  | ENST00000227525.8 | 1475 | 1451 | 75M   | -99   | true     | R1
    '''
    # from dict(pysam.SAM_FLAGS.__members__), but not worth rest of pysam dependency
    sam_flags = {
        "FPAIRED":1,
        "FPROPER_PAIR":2,
        "FUNMAP":4,
        "FMUNMAP":8,
        "FREVERSE":16,
        "FMREVERSE":32,
        "FREAD1":64,
        "FREAD2":128,
        "FSECONDARY":256,
        "FQCFAIL":512,
        "FDUP":1024,
        "FSUPPLEMENTARY":2048
    }

    sam_flags = {np.log2(v).astype(int):k for k,v in sam_flags.items()}

    def get_flag_expressions(flags_to_use):
        return [
            (pl.col('FLAG') & 2**i == 2**i).alias(flagname)
            for i, flagname in sam_flags.items()
            if flagname in flags_to_use
        ]

    usecols = ['QNAME','FLAG','RNAME','POS','MPOS','CIGAR','ISIZE']
    flags_to_use = ['FREAD1', 'FREAD2', 'FREVERSE']
    # polars-bio has no read-time flag exclusion, so replicate bam_utils'
    # exclude_flags=2308 (unmapped 4 + secondary 256 + supplementary 2048) as a filter.
    # use_zero_based=False keeps POS/MPOS 1-based to match the SAM spec (and bam_utils).
    bam_df = pb.scan_bam(str(fp_bam), use_zero_based=False)
    bam_df = bam_df.filter((pl.col('flags') & 2308) == 0)
    bam_df = bam_df.rename({
        'name': 'QNAME', 'flags': 'FLAG', 'chrom': 'RNAME', 'start': 'POS',
        'cigar': 'CIGAR', 'mate_start': 'MPOS', 'template_length': 'ISIZE',
    }).select(
        *usecols
    ).cast({
        'FLAG': pl.UInt16,        # polars-bio emits UInt32
        'RNAME': pl.Categorical,  # polars-bio emits String
    })
    # Materialize the BAM read eagerly, then continue lazily on the in-memory
    # frame. bam_utils.import_bam was itself eager, so this preserves behavior;
    # it also sidesteps a polars-bio defect where each execution re-registers a
    # file-backed BAM table in a shared context and panics ("table already
    # exists") when the same BAM backs coexisting/concurrently-collected
    # LazyFrames (e.g. the pl.collect_all in main()).
    bam_df = bam_df.collect().lazy()
    bam_df = bam_df.with_columns(
        get_flag_expressions(flags_to_use)
    )
    bam_df = bam_df.with_columns(
        side = pl.when(pl.col('FREAD1')).then(pl.lit('R1')).otherwise(pl.lit('R2'))
    ).drop('FREAD1', 'FREAD2')

    if transcript_categories is not None:
        # Check that all transcripts in bam are present in faidx
        categories = transcript_categories.to_series()
        bam_rnames = (
            bam_df.select(pl.col('RNAME').cast(pl.String))
            .unique()
            .collect()
            .to_series()
        )
        missing = bam_rnames.filter(~bam_rnames.is_in(categories.implode())).sort()
        if missing.len() > 0:
            preview = ', '.join(missing.head(10).to_list())
            raise ValueError(
                f"{missing.len()} BAM reference name(s) are absent from the FASTA "
                f"index: {preview}{', ...' if missing.len() > 10 else ''}. "
                "The BAM and FASTA appear to reference different sequence names "
                "(e.g. different assembly/annotation or transcript versioning)."
            )
        bam_df = bam_df.with_columns(
            pl.col('RNAME').cast(pl.Enum(categories=transcript_categories))
        )
    #TO DO: will we ever use this without transcript_categories?

    # if bam_df.select('QNAME','side').collect().is_duplicated().sum() > 0:
    #     raise ValueError('Expects one alignment per side per pair')

    return bam_df


def expand_cigar(bam_df):
    '''Explode each alignment's CIGAR into one row per reference-consuming match run.

    Keeps only match ops (M/X/=), computes each run's reference span
    (`cigar_start`/`cigar_end`), and deduplicates paired-end overlap by capping
    the left mate's end at the mate start (`cigar_end_dedup`) and dropping runs
    that fall entirely past it. Input alignment columns are carried through.

    Example output:
        QNAME         | FLAG | RNAME             | POS  | MPOS
        M05473…:12904 | 675  | ENST00000227525.8 | 1451 | 1475
        M05473…:12904 | 595  | ENST00000227525.8 | 1475 | 1451

        ISIZE | FREVERSE | side | cigar_length | cigar_pos_offset
        99    | false    | R2   | 75           | 0
        -99   | true     | R1   | 75           | 0

        is_left_mate | cigar_start | cigar_end | cigar_end_dedup
        true         | 1451        | 1526      | 1475
        false        | 1475        | 1550      | 1550
    '''
    expanded_cigar_df = bam_df.with_row_index(
    ).with_columns(
        pl.col('CIGAR').str.extract_all(
            r'\d+[MX=DN]'    # all ops that "consume reference" in SAM spec
        ).alias('cigar_part')
    ).explode(
        'cigar_part'
    ).with_columns(
        pl.col('cigar_part').str.extract_groups(
            r'(\d+)([MX=DN])'
        ).struct.rename_fields(
            ['cigar_length','cigar_op']
        ).struct.unnest()
    ).cast(
        {'cigar_length':pl.UInt32, 'cigar_op':pl.Enum(['M','X','=','D','N'])}
    ).with_columns(
        cigar_pos_offset = pl.col('cigar_length').cum_sum().shift().fill_null(0).over('index')
    ).filter(    # filter for "match" ops
        pl.col('cigar_op').is_in(['M','X','='])
    ).drop(    # TO DO: allow option to exclude mismatch ("X") ops or only keep match ("=") ops
        ['CIGAR','cigar_part', 'index', 'cigar_op']
    )

    # Remove double counted bases
    expanded_cigar_df = expanded_cigar_df.with_columns(
        is_left_mate = pl.col('POS') < pl.col('MPOS'),
        cigar_start  = pl.col('POS') + pl.col('cigar_pos_offset'),
        cigar_end    = pl.col('POS') + pl.col('cigar_pos_offset') + pl.col('cigar_length')
    ).filter(   # First, remove leftmate cigar parts completely overlapping rightmate
        ~pl.col('is_left_mate') | (pl.col('cigar_start') < pl.col('MPOS'))
    ).with_columns(
        cigar_end_dedup = pl.when(
            pl.col('is_left_mate') & (pl.col('cigar_end') > pl.col('MPOS'))
        ).then(
            pl.col('MPOS')
        ).otherwise(
            pl.col('cigar_end')
        )
    )

    return expanded_cigar_df


def bam_to_bedgraph(expanded_cigar_df):
    '''Collapse CIGAR match runs into a per-range coverage bedgraph.

    Groups runs by (rname, start, end) and counts them as `depth`. `start` is
    shifted to 0-based (BED convention); note `end` is left as the 1-based
    one-past-end coordinate (a known off-by-one — see the exact-end-coord tests).

    Example output:
        rname             | start | end  | depth
        ENST00000227525.8 | 1450  | 1475 | 1
        ENST00000227525.8 | 1474  | 1550 | 1
    '''
    #  BED files are 0 indexed and start-inclusive/end-exclusive
    #  SAM files are 1 indexed
    #  NOTE: bams are actually 0 indexed, while sam files are 1 indexed
    #    polars-bio is read with use_zero_based=False, so POS follows the SAM
    #    spec (and Noodles) and uses 1-based coordinates.
    bed_bam_offset = pl.lit(-1).cast(pl.Int32)

    bg = expanded_cigar_df.select(    # Create bedgraph
        pl.col('RNAME').alias('rname'),
        pl.col('cigar_start').alias('start') + bed_bam_offset,
        pl.col('cigar_end_dedup').alias('end')
    ).group_by(
        'rname','start','end'
    ).agg(
        pl.len().alias('depth')
    ).sort('rname','start')

    return bg


def get_binned_coverage(bg, fixed_length_bin_bp, faidx):
    '''Redistribute bedgraph coverage into fixed-length (bp) bins per transcript.

    Splits each coverage range across the bins it overlaps, weighting partial
    bins by their covered fraction, then sums `depth_fractional` per
    (rname, bin_start). `fixed_length_bin_bp` sets the bin size and `faidx`
    supplies transcript lengths.

    Example output:
        rname             | bin_start | depth_fractional
        ENST00000227525.8 | 1400.0    | 0.51
        ENST00000227525.8 | 1500.0    | 0.5
    '''
    binning_type = 'fixed_length'   # One of ['fixed_length', 'fixed_n_bins']

    if binning_type == 'fixed_length':
        bin_cov = bg.with_columns(
            start_bin_exact = pl.col('start') / fixed_length_bin_bp,
            end_bin_exact = pl.col('end') / fixed_length_bin_bp
        )
    elif binning_type == 'fixed_n_bins':
        raise BaseException('not yet implemented')

    # Join transcript lengths, for both determining last bin boundary and calculating fixed_n_bins bin size
    bin_cov = bin_cov.join(
        faidx,
        on='rname',
        how='inner'
    )
    # TO DO CHECK THIS!! Looking at the end_bin code, we may actually want this to remain exclusive
    # convert "end" field from exclusive to inclusive
    # bin_cov = bin_cov.with_columns(
    #     end = pl.col('end') - 1,
    #     LENGTH = pl.col('LENGTH') - 1
    #     )

    # Calculate bin membership and fractions for each region
    bin_cov = bin_cov.with_columns(
        start_bin = pl.col('start_bin_exact').floor(),
        end_bin = pl.col('end_bin_exact').floor()
    ).with_columns(
        center_bin_ct = pl.max_horizontal(
            pl.lit(0),
            pl.col('end_bin') - 1 - pl.col('start_bin')
        )
    ).with_columns(
        # Handles region only spanning one single bin (i.e. end_bin_exact < start_bin+1)
        start_bin_frc = pl.min_horizontal(
            'end_bin_exact', pl.col('start_bin')+1
        ) - pl.col('start_bin_exact'),
        end_bin_frc = (
            pl.col('end_bin_exact')
            - (pl.min_horizontal('end_bin_exact', pl.col('start_bin')+1))
            - pl.col('center_bin_ct')
        )
    )

    # Expand to one row per bin
    bin_cov = bin_cov.with_row_index(   # Generate unique (and identical within-group) indices for each region
        name = 'region_id'
    ).select(
        pl.col(
            'region_id','rname','depth','start_bin','end_bin','start_bin_frc','end_bin_frc'
        ).repeat_by(
            pl.col('center_bin_ct') + 1 + pl.when(pl.col('end_bin_frc') != 0).then(1).otherwise(0)
        ).explode()
    ).with_columns(
        bin_id = pl.int_range(pl.len()).over('region_id') + pl.col('start_bin')
    ).with_columns(
        depth_fractional = pl.when(
            pl.col('bin_id') == pl.col('start_bin')
        ).then(
            'start_bin_frc'
        ).when(
            pl.col('bin_id') == pl.col('end_bin')
        ).then(
            'end_bin_frc'
        ).otherwise(
            1
        ) * pl.col('depth')
    )


    # Add bin start coordinate
    if binning_type == 'fixed_length':
        bin_cov = bin_cov.with_columns(
            bin_start = pl.col('bin_id') * fixed_length_bin_bp
        )
    elif binning_type == 'fixed_n_bins':
        raise BaseException('not yet implemented')


    # Sum counts for each bin across all regions overlapping it
    bin_cov = bin_cov.group_by(
        'rname','bin_start'
    ).agg(
        pl.col('depth_fractional').sum()
    ).sort('rname','bin_start')

    return bin_cov


def get_bin_gc(bin_cov, fixed_length_bin_bp, sequences):
    '''Attach per-bin GC fraction and within-transcript normalized depth.

    Computes each bin's GC fraction from the transcript sequence, joins it onto
    the binned coverage, and adds `depth_normalized` (bin depth over the
    transcript mean bin depth) and `gc_frac_rounded` (GC rounded to 2 dp).

    Example output:
        rname             | bin_start | depth_fractional | gc_frac | depth_normalized | gc_frac_rounded
        ENST00000227525.8 | 1400      | 0.51             | 0.59    | 1.009901         | 0.59
        ENST00000227525.8 | 1500      | 0.5              | 0.47    | 0.990099         | 0.47
    '''
    bin_gc = sequences.join(
        bin_cov.select(pl.col('rname')).unique(),
        on='rname'
    ).with_columns(
        pl.col('seq').str.len_chars().alias('length')
    ).select(
        pl.exclude('length'),
        pl.int_ranges(pl.col('length'), step=fixed_length_bin_bp).alias('bin_start')
    ).explode(
        'bin_start'
    ).cast(
        {'seq':pl.String}
    ).select(
        pl.col('rname','bin_start'),
        bin_seq = pl.col('seq').str.slice(pl.col('bin_start'), fixed_length_bin_bp)
    ).select(
        pl.col('rname','bin_start'),
        gc_frac = pl.col('bin_seq').str.count_matches('[GC]') / pl.col('bin_seq').str.len_chars()
    )

    bin_cov_with_gc = bin_cov.cast(
        {'bin_start':pl.UInt32}
    ).join(
        bin_gc,
        on=['rname', 'bin_start']
    ).with_columns(
        depth_normalized = pl.col('depth_fractional') / pl.col('depth_fractional').mean().over('rname'),
        gc_frac_rounded = pl.col('gc_frac').round(2)
    )

    return bin_cov_with_gc


def plot(bin_cov_with_gc):
    '''Build a stacked hvplot layout of GC-bias coverage and bin count vs GC.

    Returns a HoloViews layout: mean `depth_normalized` over `gc_frac_rounded`
    above a line of per-GC bin counts. Intended for interactive/notebook use
    (not part of the CLI path).
    '''
    lineplot = bin_cov_with_gc.group_by(
        'gc_frac_rounded'
    ).agg(
        pl.col('depth_normalized').mean()
    ).sort('gc_frac_rounded').hvplot.line(
        x='gc_frac_rounded',
        y='depth_normalized'
    )

    histplot = bin_cov_with_gc.group_by(
        'gc_frac_rounded'
    ).agg(
        pl.len().alias('count')
    ).hvplot.line(
        x='gc_frac_rounded',
        y='count',
        height=100
    )

    return (lineplot + histplot).cols(1)


def calculate_gc_coverage(fp_bam, sequences, faidx, fixed_length_bin_bp=100):
    '''Run the full BAM→binned GC-vs-coverage pipeline for one BAM.'''
    transcript_categories = faidx.select(pl.col('rname').cat.get_categories()).collect()
    bam_df = load_bam(fp_bam, transcript_categories=transcript_categories)
    expanded_cigar_df = expand_cigar(bam_df)
    bg = bam_to_bedgraph(expanded_cigar_df)
    bin_cov = get_binned_coverage(bg, fixed_length_bin_bp, faidx)
    bin_cov_with_gc = get_bin_gc(bin_cov, fixed_length_bin_bp, sequences)

    return bin_cov_with_gc


def calculate_gc_pct_coverage(bin_cov_with_gc):
    '''Average normalized depth per rounded GC fraction across all bins.

    Example output:
        gc_frac_rounded | depth_normalized
        0.47            | 0.990099
        0.59            | 1.009901
    '''
    return bin_cov_with_gc.group_by(
        'gc_frac_rounded'
    ).agg(
        pl.col('depth_normalized').mean()
    ).sort('gc_frac_rounded')


def calculate_gc_pct_frequency(bin_cov_with_gc):
    '''Count how many covered bins fall in each rounded GC fraction.

    Example output:
        gc_frac_rounded | count
        0.47            | 1
        0.59            | 1
    '''
    return bin_cov_with_gc.group_by(
        'gc_frac_rounded'
    ).agg(
        pl.len().alias('count')
    ).sort('gc_frac_rounded')


def calculate_gc_pct_frequency_across_full_transcriptome(sequences, fixed_length_bin_bp):
    '''Count bins per rounded GC fraction across every transcript's full sequence.

    Example output:
        gc_frac_rounded | count_bins_full_transcriptome
        0.24            | 1
        0.25            | 1
    '''
    return sequences.with_columns(
        pl.int_ranges(pl.col('seq').str.len_chars(), step=fixed_length_bin_bp).alias('start')
    ).explode(
        'start'
    ).select(
        pl.col('seq').str.slice(pl.col('start'), fixed_length_bin_bp).alias('bin_seq')
    ).select(
        ( pl.col('bin_seq').str.count_matches('[GC]') / pl.col('bin_seq').str.len_chars() )
        .round(2)
        .alias('gc_frac_rounded')
    ).select(
        pl.col('gc_frac_rounded').value_counts()
    ).unnest(
        'gc_frac_rounded'
    ).rename({
        'count':'count_bins_full_transcriptome'
    }).sort('gc_frac_rounded')


def main():
    parser = argparse.ArgumentParser(
        description='Calculate GC content and coverage from BAM file.'
    )
    parser.add_argument(
        'fp_fasta',
        type=Path,
        help='Path to Fasta file containing transcript sequences. Must be indexed.'
    )
    parser.add_argument(
        'fp_bam',
        type=Path,
        help='Path to the BAM file.'
    )
    parser.add_argument(
        '--outfp', '-o',
        type=Path,
        default=Path('/dev/stdout'),
        help='Path to the output file. Defaults to /dev/stdout'
    )
    parser.add_argument(
        '--fixed_length_bin_bp',
        type=int,
        default=100,
        help='Length of each bin in base pairs (default: 100).'
    )
    parser.add_argument(
        '--report_bin_count_for_full_transcriptome',
        action='store_true',
        help='Whether to calculate and report the frequency of each GC percentage bin across the full transcriptome'
    )
    args = parser.parse_args()

    fp_faidx = args.fp_fasta.with_suffix('.fa.fai')
    if not fp_faidx.exists():
        raise FileNotFoundError(f"FASTA index file not found: {fp_faidx}")

    sequences, faidx = load_sequences(args.fp_fasta, fp_faidx)

    bin_cov_with_gc = calculate_gc_coverage(
        args.fp_bam,
        sequences,
        faidx,
        fixed_length_bin_bp=args.fixed_length_bin_bp,
    )

    per_gc_pct_coverage = calculate_gc_pct_coverage(bin_cov_with_gc)

    gc_pct_frequency = calculate_gc_pct_frequency(bin_cov_with_gc)

    if args.report_bin_count_for_full_transcriptome:
        gc_pct_frequency_all_transcripts = \
            calculate_gc_pct_frequency_across_full_transcriptome(sequences, args.fixed_length_bin_bp)

        per_gc_pct_coverage, gc_pct_frequency, gc_pct_frequency_all_transcripts = pl.collect_all(
            [per_gc_pct_coverage, gc_pct_frequency, gc_pct_frequency_all_transcripts]
        )
    else:
        per_gc_pct_coverage, gc_pct_frequency = pl.collect_all(
            [per_gc_pct_coverage, gc_pct_frequency]
        )

    output = per_gc_pct_coverage.join(
        gc_pct_frequency,
        on='gc_frac_rounded',
        how='left'
    )
    
    if args.report_bin_count_for_full_transcriptome:
        output = output.join(
            gc_pct_frequency_all_transcripts,
            on='gc_frac_rounded',
            how='right'
        ).rename({
            'count_bins_full_transcriptome': 'transcriptome_bin_count_all_transcripts'
        })
    
    output = output.rename({
        'depth_normalized': 'mean_normalized_depth',
        'count': 'transcriptome_bin_count',
        'gc_frac_rounded': 'gc_fraction'
    }).select(
        pl.col('gc_fraction','mean_normalized_depth','transcriptome_bin_count','transcriptome_bin_count_all_transcripts')
    )
    output.write_csv(args.outfp, separator='\t')

if __name__ == "__main__":
    main()
