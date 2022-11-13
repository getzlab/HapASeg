#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import scipy.stats as s

###################
# these functions take the raw coverage and callstats files and filter them according
# to each method's specifications
##################

# read in callstats and trim useless columns
def load_callstats(callstats_file):
    CS = pd.read_csv(callstats_file, sep='\t', low_memory=False, comment="#")
    CS = CS.loc[:, ['contig', 'position', 'ref_allele', 'alt_allele', 'total_reads', 'map_Q0_reads', 't_ref_count', 't_alt_count', 'n_ref_count', 'n_alt_count']]
    CS = CS.rename({'contig':'chr', 'position':'pos', 'ref_allele':'ref', 'alt_allele':'alt', 'map_Q0_reads':'mapq0_reads', 't_ref_count': 't_refcount', 't_alt_count': 't_altcount', 'n_ref_count': 'n_refcount', 'n_alt_count':'n_altcount'}, axis=1)
    CS = CS.astype({ "chr" : str, "pos" : np.uint32, "total_reads" : np.uint32, "mapq0_reads" : np.uint32, "t_refcount" : np.uint32, "t_altcount" : np.uint32, "n_refcount" : np.uint32, "n_altcount" : np.uint32 })
    return CS

# HapASeg

# For hapaseg this is step is only done here for speed of benchmarking.
# callstats filtering is usually done as part of the hetpulldown in the workflow
def hapaseg_cs_filtering(callstats_file = None,
                         sample_name = None,
                         outdir='./'):

    CS = load_callstats(callstats_file)
    
    frac_mapq0 = CS["mapq0_reads"]/CS["total_reads"] 
    mapq_pass_idx = frac_mapq0 <= 0.05
    tumor_total_reads = CS["total_reads"] - CS.loc[:, ["n_refcount", "n_altcount"]].sum(1)
    frac_prefiltered = 1 - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)/tumor_total_reads
    prefilter_pass_idx = frac_prefiltered <= 0.1
    CS = CS.loc[mapq_pass_idx & prefilter_pass_idx]

    # filter based on het site posterior odds 
    A = CS["t_altcount"].values[:, None]
    B = CS["t_refcount"].values[:, None]
    CS["log_pod"] = np.abs(s.beta.logsf(0.5, A + 1, B + 1) - s.beta.logcdf(0.5, A + 1, B + 1))
    CS['bdens'] = s.beta.cdf(0.6, A + 1, B + 1) - s.beta.cdf(0.4, A + 1, B + 1)
    good_pod_idx = CS["log_pod"] < 2.15
    good_bdens_idx = CS['bdens'] > 0.7
    good_idx = good_pod_idx & good_bdens_idx
    CS = CS.loc[good_idx]

    # posterior odds are not influenced by coverage, so still need to filter out poorly covered snps
    CS = CS.loc[CS.total_reads > 8]
    V = CS[['chr', 'pos', 't_refcount', 't_altcount']]
    V.to_csv(os.path.join(outdir, f'./hapaseg_{sample_name}_hetsites_cs_filtered.tsv'), sep = '\t', index=False)
    D = CS[['chr', 'pos', 'total_reads']]
    D.to_csv(os.path.join(outdir, f'./hapaseg_{sample_name}_hetsites_depth.tsv'), sep = '\t', header=None,index=False)


# Facets

def facets_cs_filtering(callstats_file = None,
                        db_snp_file = None,
                        sample_name = None,
                        outdir='./'):
    
    CS = load_callstats(callstats_file)
    
    # be nice and filter based on frac mapq0 reads and mutect filtered reads
    frac_mapq0 = CS["mapq0_reads"]/CS["total_reads"] 
    mapq_pass_idx = frac_mapq0 <= 0.05
    tumor_total_reads = CS["total_reads"] - CS.loc[:, ["n_refcount", "n_altcount"]].sum(1)
    frac_prefiltered = 1 - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)/tumor_total_reads
    prefilter_pass_idx = frac_prefiltered <= 0.1
    CS = CS.loc[mapq_pass_idx & prefilter_pass_idx]

    # filter on common dbSNP polymorphisms
    # can be downloaded from https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz
    common_snps = pd.read_csv(db_snp_file, sep='\t', comment='#', usecols=[0,1], low_memory=False, names=['chr', 'pos'])
    common_snps['chr'] = common_snps['chr'].apply(lambda x : 'chr' + x)

    # filter for common snps
    CS = CS.merge(common_snps, on = ['chr', 'pos'], how='inner')
    V = CS[['chr', 'pos', 't_refcount', 't_altcount', 'total_reads']]
    V.to_csv(os.path.join(outdir, f'facets_{sample_name}_cs_variant_filtered.tsv'), sep='\t', index=False)

    D = CS[['chr', 'pos', 'total_reads']]
    D.to_csv(os.path.join(outdir, f'facets_{sample_name}_cs_variant_depths.tsv'), sep='\t', header=None, index=False)

# ASCAT

def ascat_cs_filtering(callstats_file = None,
                       sample_name = None,
                       ascat_loci_list = None,
                       outdir = './'):

    CS = load_callstats(callstats_file)

    frac_mapq0 = CS["mapq0_reads"]/CS["total_reads"] 
    mapq_pass_idx = frac_mapq0 <= 0.05
    tumor_total_reads = CS["total_reads"] - CS.loc[:, ["n_refcount", "n_altcount"]].sum(1)
    frac_prefiltered = 1 - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)/tumor_total_reads
    prefilter_pass_idx = frac_prefiltered <= 0.1
    CS = CS.loc[mapq_pass_idx & prefilter_pass_idx]

    # filter on ascat WGS loci list
    # can be downloaded from https://www.dropbox.com/s/80cq0qgao8l1inj/G1000_loci_hg38.zip
    common_snps = pd.read_csv(ascat_loci_list, sep='\t', low_memory=False, names=['chr', 'pos'])
    common_snps['chr'] = common_snps['chr'].apply(lambda x : 'chr' + x)

    # filter for common snps
    CS = CS.merge(common_snps, on = ['chr', 'pos'], how='inner')
    V = CS[['chr', 'pos', 't_refcount', 't_altcount', 'total_reads']]
    V.to_csv(os.path.join(outdir, f'ascat_{sample_name}_cs_variant_filtered.tsv'), sep='\t', index=False)

    D = CS[['chr', 'pos', 'total_reads']]
    D.to_csv(os.path.join(outdir, f'ascat_{sample_name}_cs_variant_depths.tsv'), sep='\t', header=None, index=False)
    
def parse_args():
    parser = argparse.ArgumentParser(description = "preprocess callstats file for benchmarking methods use")
    parser.add_argument("--callstats", required = True, help="path to mutect call stats file")
    parser.add_argument("--sample_name", required = True, help="name of sample for file naming")
    parser.add_argument("--outdir", default="./", help="directory in which to save output files")

    subparsers = parser.add_subparsers(dest="command")
    
    # hapaseg -- no additional args
    hapaseg_cs = subparsers.add_parser("hapaseg", help = "preprocess callstats file for hapaseg")
    
    # facets
    facets_cs = subparsers.add_parser("facets", help = "preprocess callstats file for facets")
    facets_cs.add_argument("--db_snp_vcf", required=True, help="path to db_snp vcf file containing common variants")

    # ascat
    ascat_cs = subparsers.add_parser("ascat", help = "preprocess callstats file for ascat")
    ascat_cs.add_argument("--ascat_loci_list", required = True, help="ascat WGS loci list")

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    output_dir = os.path.realpath(args.outdir)

    if args.command == "hapaseg":
        print("filtering callstats for hapaseg...", flush=True)
        hapaseg_cs_filtering(callstats_file = args.callstats,
                             sample_name = args.sample_name,
                             outdir = output_dir)

    elif args.command == "ascat":
        print("filtering callstats for ascat...", flush=True)
        ascat_cs_filtering(callstats_file = args.callstats,
                           sample_name = args.sample_name, 
                           ascat_loci_list = args.ascat_loci_list,
                           outdir = output_dir)

    elif args.command == "facets":
        print("filtering callstats for facets...", flush=True)
        facets_cs_filtering(callstats_file = args.callstats,
                            sample_name = args.sample_name,
                            db_snp_file = args.db_snp_vcf,
                            outdir = output_dir)
    else:
        raise ValueError(f"could not recognize command {args.command}")

if __name__ == "__main__":
    main()
