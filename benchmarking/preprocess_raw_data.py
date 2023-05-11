#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import scipy.stats as s
import h5py
import subprocess
from capy import mut

###################
# these functions take the raw coverage and callstats files and filter them according
# to each method's specifications
##################

# read in callstats and trim useless columns
def load_callstats(callstats_file):
    callstats_trimmed = subprocess.Popen("sed '1,2d' {} | cut -f1,2,4,5,16,17,26,27,38,39".format(callstats_file), shell = True, stdout = subprocess.PIPE)
    CS = pd.read_csv(callstats_trimmed.stdout, sep = "\t",
      names = ["chr", "pos", "ref", "alt", "total_reads", "mapq0_reads", "t_refcount", "t_altcount", "n_refcount", "n_altcount"],
          dtype = { "chr" : str, "pos" : np.uint32, "total_reads" : np.uint32, "mapq0_reads" : np.uint32, "t_refcount" : np.uint32, "t_altcount" : np.uint32, "n_refcount" : np.uint32, "n_altcount" : np.uint32 }
    )
    return CS

# HapASeg

# For hapaseg this is step is only done here for speed of benchmarking.
# callstats filtering is usually done as part of the hetpulldown in the workflow
def hapaseg_cs_filtering(callstats_file = None,
                         sample_name = None,
                         outdir='./',
                         dummy_normal=False #pass true to skip het site filtering
                        ):

    CS = load_callstats(callstats_file)
    
    frac_mapq0 = CS["mapq0_reads"]/CS["total_reads"] 
    mapq_pass_idx = frac_mapq0 <= 0.05
    tumor_total_reads = CS["total_reads"] - CS.loc[:, ["n_refcount", "n_altcount"]].sum(1)
    frac_prefiltered = 1 - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)/tumor_total_reads
    prefilter_pass_idx = frac_prefiltered <= 0.1
    CS = CS.loc[mapq_pass_idx & prefilter_pass_idx]
    ample_cov = CS.loc[:, ["t_refcount", "t_altcount"]].sum(1) > 8
    
    if not dummy_normal:
    # filter based on het site posterior odds 
        A = CS["t_altcount"].values[:, None]
        B = CS["t_refcount"].values[:, None]
        CS["log_pod"] = np.abs(s.beta.logsf(0.5, A + 1, B + 1) - s.beta.logcdf(0.5, A + 1, B + 1))
        CS['bdens'] = s.beta.cdf(0.6, A + 1, B + 1) - s.beta.cdf(0.4, A + 1, B + 1)
        good_pod_idx = CS["log_pod"] < 2.15
        good_bdens_idx = CS['bdens'] > 0.7
        CS['bden_hom'] =  1 - s.beta.cdf(0.95, A + 1, B + 1)
        good_idx = good_pod_idx & good_bdens_idx & ample_cov

    else:
        good_idx = ample_cov

    V = CS.loc[good_idx, ['chr', 'pos', 't_refcount', 't_altcount']]
    V.to_csv(os.path.join(outdir, f'./hapaseg_{sample_name}_hetsites_cs_filtered.tsv'), sep = '\t', index=False)
    D = CS.loc[good_idx, ['chr', 'pos', 'total_reads']]
    D.to_csv(os.path.join(outdir, f'./hapaseg_{sample_name}_hetsites_depth.tsv'), sep = '\t', header=None,index=False)

    if not dummy_normal:
        # also return genotype
        hom_idx = CS['bden_hom'] > 0.7
        genotype_idxs = ample_cov & ((good_pod_idx & good_bdens_idx) | hom_idx)
        
        G = CS.loc[genotype_idxs, ['chr', 'pos', 'ref', 'alt']]
        
        alleles = np.c_[G.alt.values, G.ref.values]
        # create genotype for confidently homozygous sites
        alleles[hom_idx[genotype_idxs], 1] = alleles[hom_idx[genotype_idxs], 0]
        
        G['genotype'] =  [ref + alt for alt, ref in alleles]
        G[['chr', 'pos', 'genotype']].to_csv(os.path.join(outdir, f'./hapaseg_{sample_name}_genotype.tsv'), sep = '\t', index=False)

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
   
# GATK 

## utility methods
def convert_gatkCov_to_hapasegCov_format(hdf5_file):
    f = h5py.File(hdf5_file, 'r')
    chromosomes = np.array([chrom.decode() for chrom in f['intervals/indexed_contig_names'][:]])
    interval_data = f['intervals/transposed_index_start_end'][:].astype(int)
    cov_formatted = pd.DataFrame({'chr': chromosomes[interval_data[0]], 'start':interval_data[1], 'end':interval_data[2], 'covcorr': f['counts/values'][0].astype(int), 'mean_fraglen':0, 'sqrt_avg_fragvar':0, 'tot_reads':0, 'fail_reads':0})
    return cov_formatted

def generate_gatkCov_dummy_normal(hdf5_in, hdf5_out):
    f = h5py.File(hdf5_in, 'r')
    new_f = h5py.File(hdf5_out, 'a')
    # copy all groups over to replicate the metadata
    for k in f.keys():
        new_f.copy(f[k], k)
    # set all counts to the mean
    new_f['counts/values'][:] = new_f['counts/values'][0].mean()
    # close files
    f.close()
    new_f.close()

def convert_gatkAlleleCounts_to_Hapaseg(gatk_allelecounts_tsv_path):
    df = pd.read_csv(gatk_allelecounts_tsv_path, comment='@', sep='\t', low_memory=False)
    variant_depths = df['REF_COUNT'] + df['ALT_COUNT']
    variant_depths_df = pd.DataFrame({'CHROM': df['CONTIG'], 'POS': df['POSITION'], 'DEPTH':variant_depths})
    
    sim_normal_df = df.copy()
    non_zero_mask = (df['REF_COUNT'] + df['ALT_COUNT']) > 0
    sim_altcount = s.binom.rvs(30, df.loc[non_zero_mask,'ALT_COUNT'] / (df.loc[non_zero_mask, 'REF_COUNT'] + df.loc[non_zero_mask, 'ALT_COUNT']))
    sim_refcount = 30 - sim_altcount
    sim_normal_df.loc[non_zero_mask, 'ALT_COUNT'] = sim_altcount
    sim_normal_df.loc[non_zero_mask, 'REF_COUNT'] = sim_refcount
    return variant_depths_df, sim_normal_df

def gatk_preprocessing(gatk_fragcounts = None,
                       gatk_allelecounts = None,
                       sample_name = None,
                       outdir = './'):
    
    gatk_cov_df = convert_gatkCov_to_hapasegCov_format(gatk_fragcounts)
    gatk_cov_df.to_csv(os.path.join(outdir, f'{sample_name}_gatk_cov_counts.tsv'), sep='\t', index = False, header=False)
    generate_gatkCov_dummy_normal(gatk_fragcounts, os.path.join(outdir, f'{sample_name}_gatk_sim_normal_frag.counts.hdf5'))
    gatk_var_depth, gatk_sim_normal_allele_counts = convert_gatkAlleleCounts_to_Hapaseg(gatk_allelecounts)
    gatk_var_depth.to_csv(os.path.join(outdir, f'{sample_name}_gatk_var_depth.tsv'), sep='\t', index=False)
    gatk_sim_normal_allele_counts.to_csv(os.path.join(outdir, f'{sample_name}_gatk_sim_normal_allele_counts.tsv'), sep='\t', index=False)

def hatchet_preprocessing(totals_file_paths_txt = None,
                          thresholds_file_paths_txt = None,
                          tumor_baf_path = None,
                          sample_name = None,
                          dummy_normal=False, # pass true to interpret normal as first the tumor sample and tumor as second tumor sample
                          outdir = './' ):
    
    total_names = ['n_int_reads', 'n_pos_reads', 't_int_reads', 't_pos_reads']
    if dummy_normal:
        # throw out the first two columns
        total_names = ["dummy_int_reads", "dummy_pos_reads"] + total_names

    total_reads_paths = open(totals_file_paths_txt, 'r').read().split()
    thresholds_snps_paths = open(thresholds_file_paths_txt, 'r').read().split()

    if len(total_reads_paths) != len(thresholds_snps_paths):
        raise ValueError("number of totals and thresholds files found did not match!")

    interval_corr_lst = []
    position_corr_lst = []
    read_combined_lst = []

    print("Reading in total and threshold counts by chromosome")
    l_bname = lambda x: os.path.basename(x)
    for total_fn, threshold_fn in zip(sorted(total_reads_paths, key = l_bname), sorted(thresholds_snps_paths, key = l_bname)):
        if os.path.basename(total_fn).rstrip('total.gz') != os.path.basename(threshold_fn).rstrip('thresholds.gz'):
            raise ValueError("chromosomes in total and thesholds files did no match up")

        this_read_df = pd.read_csv(total_fn, sep=' ', header=None, names=total_names)
        if dummy_normal:
            this_read_df = this_read_df.drop(['dummy_int_reads', 'dummy_pos_reads'], axis=1)

        this_thresholds_df = pd.read_csv(threshold_fn, sep='\t', header=None, names=['threshold'])
        
        int_corr = pd.concat([this_thresholds_df['threshold'], pd.Series(np.append(this_thresholds_df.loc[1:, 'threshold'], this_thresholds_df.values[-1])), this_read_df['t_int_reads']], axis=1)
        int_corr.columns = ['start', 'end', 'covcorr']
        chrom = os.path.basename(total_fn).rstrip('total.gz')
        int_corr['contig'] = chrom
        pos_corr = pd.concat([this_thresholds_df['threshold'], this_thresholds_df['threshold'] + 1, this_read_df['t_pos_reads']], axis=1)
        pos_corr.columns = ['start', 'end', 'covcorr']
        pos_corr['contig'] = chrom

        this_read_df['contig'] = chrom
        interval_corr_lst.append(int_corr)
        position_corr_lst.append(pos_corr)
        read_combined_lst.append(this_read_df)

    interval_corr_df = pd.concat(interval_corr_lst)
    position_corr_df = pd.concat(position_corr_lst)
    read_combined_df = pd.concat(read_combined_lst)
    
    #
    int_counts_sim_fn = os.path.join(outdir, f'{sample_name}_interval_counts.for_simulation_input.txt')
    pos_counts_sim_fn = os.path.join(outdir, f'{sample_name}_position_counts.for_simulation_input.txt')
    snp_counts_sim_fn = os.path.join(outdir, f'{sample_name}_snp_counts.for_simulation_input.txt')
    read_combined_fn = os.path.join(outdir, f'{sample_name}_read_combined_df.txt')

    interval_corr_df.to_csv(int_counts_sim_fn, sep='\t', index=False, columns=['contig', 'start', 'end', 'covcorr'], header=False)
    position_corr_df.to_csv(pos_counts_sim_fn, sep='\t', index=False, columns=['contig', 'start', 'end', 'covcorr'], header=False)
    read_combined_df.to_csv(read_combined_fn, sep='\t', index=False)
 
    snp_counts_1bed = pd.read_csv(tumor_baf_path, sep='\t', header=None, names=['contig', 'pos', 'sample', 'ref_count', 'alt_count'])
    snp_counts_1bed['depth'] = snp_counts_1bed['ref_count'] + snp_counts_1bed['alt_count']
    snp_counts_1bed.to_csv(snp_counts_sim_fn, sep='\t', index=False, columns=['contig', 'pos', 'depth'])

## standard preprocessing (i.e. from real tumor/normal to method inputs    
# Facets

def facets_standard_filtering(callstats_file = None,
                        sample_name = None,
                        outdir='./'):
    
    CS = load_callstats(callstats_file)
    
    # be nice and filter based on frac mapq0 reads and mutect filtered reads
    frac_mapq0 = CS["mapq0_reads"]/CS["total_reads"] 
    mapq_pass_idx = frac_mapq0 <= 0.05
    tumor_total_reads = CS["total_reads"] - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)
    frac_prefiltered = 1 - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)/tumor_total_reads
    prefilter_pass_idx = frac_prefiltered <= 0.1
    CS = CS.loc[mapq_pass_idx & prefilter_pass_idx]

    # format to match input
    CS = CS.rename({'chr':'Chromosome', 'pos': 'Position', 'ref':'Ref', 'alt':'Alt', 'n_refcount': 'File1R', 'n_altcount': 'File1A', 't_refcount': 'File2R', 't_altcount': 'File2A'}, axis=1)
    CS.loc[:, 'Chromosome'] = mut.convert_chr(CS['Chromosome']) 
    CS.loc[:, ['File1E', 'File1D', 'File2E', 'File2D']] = 0
    CS = CS[['Chromosome', 'Position', 'Ref','Alt', 'File1R', 'File1A', 'File1E','File1D', 'File2R', 'File2A', 'File2E', 'File2D']]
    CS.to_csv(f'{sample_name}_facets_input_counts.csv', index=False)

# ASCAT

def ascat_standard_filtering(callstats_file=None,
                             sample_name=None,
                             outdir='./'):
        
    CS = load_callstats(callstats_file)
    
    # be nice and filter based on frac mapq0 reads and mutect filtered reads
    frac_mapq0 = CS["mapq0_reads"]/CS["total_reads"] 
    mapq_pass_idx = frac_mapq0 <= 0.05
    tumor_total_reads = CS["total_reads"] - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)
    frac_prefiltered = 1 - CS.loc[:, ["t_refcount", "t_altcount"]].sum(1)/tumor_total_reads
    prefilter_pass_idx = frac_prefiltered <= 0.1
    CS = CS.loc[mapq_pass_idx & prefilter_pass_idx]

    CS['tumor_depth'] = CS['t_refcount'] + CS['t_altcount']
    CS['normal_depth'] = CS['n_refcount'] + CS['n_altcount']
    
    # use conservative ascat filter for sites covered by normal > 20 counts
    CS = CS.loc[CS.normal_depth > 20]

    # calculate ascat input logR and BAF
    CS['tumorLogR'] = CS['tumor_depth'] / CS['normal_depth']
    CS['tumorLogR'] = np.log2(CS['tumorLogR'] / CS['tumorLogR'].mean())
    # ascat passes around a normalLogR file but never actually defines these values (nor are they ever used)
    # we will make a dummy file 
    CS['normalLogR'] = 0

    CS['normalBAF'] = np.nan
    CS['tumorBAF'] = np.nan
    # for unexplained reasons ascat also randomizes A and B alleles
    mask = np.random.rand(len(CS)) > 0.5
    CS.loc[mask, 'normalBAF'] = CS.loc[mask, 'n_refcount'] / CS.loc[mask, 'normal_depth']
    CS.loc[~mask, 'normalBAF'] = CS.loc[~mask, 'n_altcount'] / CS.loc[~mask, 'normal_depth']
    CS.loc[mask, 'tumorBAF'] = CS.loc[mask, 't_refcount'] / CS.loc[mask, 'tumor_depth']
    CS.loc[~mask, 'tumorBAF'] = CS.loc[~mask, 't_altcount'] / CS.loc[~mask, 'tumor_depth']

    # switch back to string chrom names 
    if 'chr' in CS.iloc[0,0]:
        CS.loc[:, 'chr'] = CS['chr'].apply(lambda x: x.lstrip('chr'))
    CS = CS.rename({'chr':'chrs'}, axis=1)
    # ascat expects the index to be in chr_pos form
    CS = CS.set_index(CS.apply(lambda x: x.chrs + '_' + str(x.pos), axis = 1))
    
    file_sample_name = '{}_ascat'.format(sample_name)
    # ascat returns 4 seperate files each with one column of data
    CS.rename({'tumorLogR':file_sample_name}, axis=1)[['chrs', 'pos', file_sample_name]].to_csv(os.path.join(outdir, f'{sample_name}_ascat_tumor_LogR.txt'), sep='\t')
    CS.rename({'normalLogR':file_sample_name}, axis=1)[['chrs', 'pos', file_sample_name]].to_csv(os.path.join(outdir, f'{sample_name}_ascat_normal_LogR.txt'), sep='\t')
    CS.rename({'tumorBAF':file_sample_name}, axis=1)[['chrs', 'pos', file_sample_name]].to_csv(os.path.join(outdir, f'{sample_name}_ascat_tumor_BAF.txt'), sep='\t')
    CS.rename({'normalBAF':file_sample_name}, axis=1)[['chrs', 'pos', file_sample_name]].to_csv(os.path.join(outdir, f'{sample_name}_ascat_normal_BAF.txt'), sep='\t')

def parse_args():
    parser = argparse.ArgumentParser(description = "preprocess callstats file for benchmarking methods use")
    parser.add_argument("--sample_name", required = True, help="name of sample for file naming")
    parser.add_argument("--outdir", default="./", help="directory in which to save output files")

    subparsers = parser.add_subparsers(dest="command")
    
    # hapaseg -- no additional args
    hapaseg_cs = subparsers.add_parser("hapaseg", help = "preprocess callstats file for hapaseg")
    hapaseg_cs.add_argument("--callstats", required = True, help="path to mutect call stats file")
    hapaseg_cs.add_argument("--dummy_normal", required=False, default=False, action='store_true', help="flag to skip normal counts hetsite filtering")
    
    # facets
    facets_cs = subparsers.add_parser("facets", help = "preprocess callstats file for facets")
    facets_cs.add_argument("--db_snp_vcf", required=True, help="path to db_snp vcf file containing common variants")
    facets_cs.add_argument("--callstats", required = True, help="path to mutect call stats file")

    # ascat
    ascat_cs = subparsers.add_parser("ascat", help = "preprocess callstats file for ascat")
    ascat_cs.add_argument("--ascat_loci_list", required = True, help="ascat WGS loci list")
    ascat_cs.add_argument("--callstats", required = True, help="path to mutect call stats file")

    # GATK
    gatk = subparsers.add_parser("gatk", help="preprocess gatk raw data")
    gatk.add_argument("--frag_counts", required=True, help="path to gatk frag counts hdf5 file")
    gatk.add_argument("--allele_counts", required=True, help="path to gatk allele counts file")

    #Hatchet
    hatchet = subparsers.add_parser("hatchet", help="preprocess hatchet raw data")
    hatchet.add_argument("--totals_file_paths", required=True, help="path to txt file containing locations of chr{}.total files")
    hatchet.add_argument("--thresholds_file_paths", required=True, help = "path to txt file containing locations of chr{}.thresholds files")
    hatchet.add_argument("--tumor_baf_path", required=True, help= "path to hatchet allelecounts results")
    hatchet.add_argument("--dummy_normal", required=False, default=False, action='store_true', help="flag to ignore normal counts and use first two tumors instead")
    
    ## standard processing
    # facets
    facets_cs = subparsers.add_parser("facets-standard", help = "preprocess standard callstats file for facets (non-simulation)")
    facets_cs.add_argument("--callstats", required = True, help="path to mutect call stats file")

    # ascat
    ascat_cs = subparsers.add_parser("ascat-standard", help = "preprocess standard callstats file for ascat (non-simultation")
    ascat_cs.add_argument("--callstats", required = True, help="path to mutect call stats file")
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
                             dummy_normal = args.dummy_normal,
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
    
    elif args.command == "gatk":
        print("processing gatk raw data for benchmarking...", flush=True)
        gatk_preprocessing(gatk_fragcounts = args.frag_counts, 
                          gatk_allelecounts = args.allele_counts,
                          sample_name = args.sample_name,
                          outdir=output_dir)
    
    elif args.command == "hatchet":
        print("processing hatchet raw data for benchmarking...", flush = True)
        hatchet_preprocessing(totals_file_paths_txt = args.totals_file_paths,
                              thresholds_file_paths_txt = args.thresholds_file_paths,
                              tumor_baf_path = args.tumor_baf_path,
                              sample_name = args.sample_name,
                              outdir = output_dir,
                              dummy_normal=args.dummy_normal)
    
    elif args.command == "ascat-standard":
        print("filtering callstats for ascat...", flush=True)
        ascat_standard_filtering(callstats_file = args.callstats,
                           sample_name = args.sample_name, 
                           outdir = output_dir)

    elif args.command == "facets-standard":
        print("filtering callstats for facets...", flush=True)
        facets_standard_filtering(callstats_file = args.callstats,
                            sample_name = args.sample_name,
                            outdir = output_dir)
    else:
        raise ValueError(f"could not recognize command {args.command}")
if __name__ == "__main__":
    main()
