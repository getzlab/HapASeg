#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import argparse
import scipy.stats as s
import h5py

from cnv_suite.simulate.cnv_profile import CNV_Profile
from capy import mut

def parse_args():
    parser = argparse.ArgumentParser(description = "generate requisite benchmarking files from simulated coverage profile")
    parser.add_argument("--sim_profile", required=True, help="path to cnv_suite coverage profile pickle")
    parser.add_argument("--output_dir", default=".", help="output directory path for file outputs")
    parser.add_argument("--purity", required = True, type=float, help="purity for simulated sample")
    parser.add_argument("--out_label", required = True, help="label for this simulated sample")
    parser.add_argument("--parallel", default=False, action='store_true', help="use multiple cores for generating files")
    subparsers = parser.add_subparsers(dest="command")
    
    ## hapaseg
    hapaseg_gen = subparsers.add_parser("hapaseg", help="generate hapaseg inputs")
    hapaseg_gen.add_argument("--normal_vcf_path", required=True, help="path to normal sample vcf file")
    hapaseg_gen.add_argument("--hetsite_depth_path", required=True, help="path to hetsite depth file")
    hapaseg_gen.add_argument("--covcollect_path", required=True, help="path to normal covcollect file")

    ## ascat
    ascat_gen = subparsers.add_parser("ascat", help="generate ascat inputs")
    ascat_gen.add_argument("--normal_vcf_path", required = True, help="path to normal sample vcf file")
    ascat_gen.add_argument("--variant_depth_path", required=True, help = "path to variant depth file")
    ascat_gen.add_argument("--filtered_variants_path", required=True, help="path to filtered variants file")
    
    ## facets
    facets_gen = subparsers.add_parser("facets", help="generate facets inputs")
    facets_gen.add_argument("--normal_vcf_path", required=True, help="path to normal sample vcf file")
    facets_gen.add_argument("--variant_depth_path", required=True, help="path to variant depth file")
    facets_gen.add_argument("--filtered_variants_path", required=True, help="path to filtered variants file")

    ## gatk
    gatk_gen = subparsers.add_parser("gatk", help="generate gatk inputs")
    gatk_gen.add_argument("--normal_vcf_path", required=True, help="path to normal sample vcf file")
    gatk_gen.add_argument("--variant_depth_path", required=True, help="path to variant depth file")
    gatk_gen.add_argument("--coverage_tsv_path", required=True, help="path to gatk coverage in covcorr format")
    gatk_gen.add_argument("--sim_normal_allelecounts_path", required=True, help="path to simulated normal allelecounts in gatk format")
    gatk_gen.add_argument("--raw_gatk_allelecounts_path", required=True, 
                            help="path to original gatk output allelecounts file from raw normal")
    gatk_gen.add_argument("--raw_gatk_coverage_path", required=True,
                            help="path to original gatk output coverage hdf5 from raw normal")
    
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = os.path.realpath(args.output_dir)
    
    if args.purity < 0 or args.purity > 1:
        raise ValueError("purity must be in [0,1]")

    if args.command == "hapaseg":
        print("generating hapsaeg simulated files...")
        generate_hapaseg_files(sim_profile_pickle = args.sim_profile,
                               purity = args.purity,
                               normal_vcf_path= args.normal_vcf_path,
                               hetsite_depth_path = args.hetsite_depth_path,
                               covcollect_path = args.covcollect_path,
                               out_dir = output_dir,
                               out_label = args.out_label,
                               parallel = args.parallel)
    elif args.command == "ascat":
        print("generating ascat simulated files...")
        generate_ascat_files(sim_profile_pickle = args.sim_profile,
                             purity = args.purity,
                             normal_vcf_path = args.normal_vcf_path,
                             variant_depth_path = args.variant_depth_path,
                             filtered_variants_path = args.filtered_variants_path,
                             out_dir = output_dir,
                             out_label = args.out_label,
                             parallel = args.parallel)
    
    elif args.command == "facets":
        print("generating facets simulated files...")
        generate_facets_files(sim_profile_pickle = args.sim_profile,
                             purity = args.purity,
                             normal_vcf_path = args.normal_vcf_path,
                             variant_depth_path = args.variant_depth_path,
                             filtered_variants_path = args.filtered_variants_path,
                             out_dir = output_dir,
                             out_label = args.out_label,
                             parallel = args.parallel)

    elif args.command == "gatk":
        print("generating gatk simulated files...")        
        generate_gatk_files(sim_profile_pickle=args.sim_profile,
                        purity = args.purity,
                        normal_vcf_path=args.normal_vcf_path,
                        variant_depth_path=args.variant_depth_path,
                        raw_gatk_allelecounts_path=args.raw_gatk_allelecounts_path,
                        sim_normal_allelecounts_path=args.sim_normal_allelecounts_path,
                        coverage_tsv_path=args.coverage_tsv_path,
                        raw_gatk_coverage_path=args.raw_gatk_coverage_path,
                        out_dir=output_dir,
                        out_label=args.out_label,
                        parallel = args.parallel)

    else:
        raise ValueError("Did not recognize command")

def convert_hapaseg_cov_to_gatk_hdf5(sim_dataframe, ref_hdf5, new_hdf5):
    f = h5py.File(ref_hdf5, 'r')
    cov_df = pd.read_csv(sim_dataframe, sep='\t')
    if len(cov_df) != f['counts/values'].shape[1]:
        raise ValueError("length of simulated coverage bin does not match normal ref file")
    new_f = h5py.File(new_hdf5, 'w')
    for k in f.keys():
        new_f.copy(f[k], k)
    new_f['counts/values'][:] = cov_df.covcorr.values.astype(float)
    new_f.close()
    f.close()


def generate_hapaseg_files(sim_profile_pickle = None,
                           purity = None,
                           normal_vcf_path = None,
                           hetsite_depth_path = None,
                           covcollect_path = None,
                           out_dir = None,
                           out_label = None,
                           parallel = False):
    
    default_profile=pd.read_pickle(sim_profile_pickle)
    
    # generate snvs
    snv_df, correct_phase_interval_trees = default_profile.generate_snvs(normal_vcf_path, hetsite_depth_path, purity, do_parallel=parallel)
    hets_df = snv_df.loc[(snv_df['NA12878'] != '1|1') &
                         (snv_df.REF.apply(lambda x : len(x)) == 1) & 
                         (snv_df.ALT.apply(lambda x : len(x)) == 1)]
    hets_df = hets_df.rename(columns={'CHROM': 'CONTIG',
                            'POS': 'POSITION',
                            'ref_count': 'REF_COUNT',
                            'alt_count': 'ALT_COUNT'})[['CONTIG', 'POSITION',
                                                      'REF_COUNT', 'ALT_COUNT']]
    hets_df.to_csv(os.path.join(out_dir, '{}_{}_hapaseg_hets.bed'.format(out_label, purity)), sep='\t', index=False)

    # generate coverage
    cov_df = default_profile.generate_coverage(purity, covcollect_path, do_parallel=parallel)
    cov_df = cov_df.rename(columns={'chrom': 'chr'})
    cov_df['chr'] = cov_df['chr'].apply(lambda x: 'chr' + str(x))
    cov_df.replace(to_replace={'chr':{'chr23':'chrX', 'chr24':'chrY'}}, inplace=True)
    cov_df[['chr', 'start', 'end',
            'covcorr', 'mean_fraglen',
            'sqrt_avg_fragvar', 'n_frags',
            'tot_reads', 'reads_flagged']].to_csv(os.path.join(out_dir, '{}_{}_hapaseg_coverage.bed'.format(out_label, purity)), sep='\t', index=False, header=False)

def generate_facets_files(sim_profile_pickle=None,
                          purity = None,
                          normal_vcf_path=None,
                          variant_depth_path=None,
                          filtered_variants_path=None,
                          out_dir=None,
                          out_label=None,
                          parallel=False):

    default_profile=pd.read_pickle(sim_profile_pickle)
    # generate snvs
    snv_df, correct_phase_interval_trees = default_profile.generate_snvs(normal_vcf_path, variant_depth_path, purity, do_parallel=parallel)
    snv_df = snv_df.loc[(snv_df.REF.apply(lambda x : len(x)) == 1) & (snv_df.ALT.apply(lambda x : len(x)) == 1)]

    snv_df = snv_df.rename({'CHROM':'chr', 'POS':'pos',
                            'ref_count':'File2R',
                            'alt_count':'File2A'}, axis=1)[['chr', 'pos',
                                                            'REF', 'ALT',
                                                            'File2R', 'File2A']]
    snv_df['chr'] = mut.convert_chr(snv_df['chr']).astype(int)

    # generate fake normal using binomial with N=30
    normal_df = pd.read_csv(filtered_variants_path, sep='\t')
    normal_df['File1A'] = s.binom.rvs(30, normal_df['t_altcount'] / normal_df['total_reads'])
    normal_df['File1R'] = 30 - normal_df['File1A']
    normal_df = normal_df.drop(['t_refcount', 't_altcount', 'total_reads'], axis=1)
    normal_df['chr'] = mut.convert_chr(normal_df['chr'])

    merged = snv_df.merge(normal_df, on=['chr', 'pos'], how='inner').drop_duplicates()
    merged.loc[:, ['File1E', 'File1D', 'File2E', 'File2D']] = 0
    merged = merged.rename({'chr':'Chromosome', 'pos': 'Position', 'REF':'Ref', 'ALT':'Alt'}, axis=1)
    merged = merged[['Chromosome', 'Position', 'Ref',
                     'Alt', 'File1R', 'File1A', 'File1E',
                     'File1D', 'File2R', 'File2A', 'File2E', 'File2D']]
    merged.to_csv(os.path.join(out_dir, '{}_{}_facets_input_counts.csv'.format(out_label, purity)), index=False)
        

# utility method for adding header back onto simulated allele count files
# inputs: og_tsv: allelecounts file with header intact (outputted by gatk CollectAlleleCounts)
#          sim_tsv: path to simulated allelecounts file witout header
def add_header_to_allelecounts(og_tsv, sim_tsv):
    header=[]
    with open(og_tsv, 'r') as f:
        fl = f.readlines()
        for l in fl:
            if l[0] == "@":
                header.append(l)
    header_str = ''.join(header)
    # gather tsv
    with open(sim_tsv, 'r') as f:
        sim_contents = f.read()
    # rewrite tsv
    with open(sim_tsv, 'w') as f:
        f.write(header_str)
        f.write(sim_contents)

# utility method for converting hapseg formatted coverage tsv into gatk fragcount hdf5 format
# inputs: sim_dataframe: path to coverage tsv
#         ref_hdf5: path to a hdf5 file generated by gatk with metadata intact
#                   and intervals/contigs matching the tsv
#         new_hdf5: path to create new hdf5 file at using covcorr from coverage tsv
#                   and metadata from ref_hdf5


def generate_gatk_files(sim_profile_pickle=None,
                        purity = None,
                        normal_vcf_path=None, # path to raw data vcf
                        variant_depth_path=None, # path to variant depths in og vcf
                        raw_gatk_allelecounts_path=None, # need original gatk allele counts output to copy its header
                        sim_normal_allelecounts_path=None, # path to simulated normal allelecounts file
                        coverage_tsv_path=None, # path to gatk coverage counts converted to hapaseg format
                        raw_gatk_coverage_path=None, # need original hdf5 file to copy metadata
                        out_dir=None,
                        out_label=None,
                        parallel=False):
                        
    default_profile = pd.read_pickle(sim_profile_pickle)   
    # generate snvs
    snv_df, correct_phase_interval_trees = default_profile.generate_snvs(normal_vcf_path, variant_depth_path, purity, do_parallel=parallel)
    # limit to SNPs
    snv_df = snv_df.loc[(snv_df.REF.apply(lambda x : len(x)) == 1) & (snv_df.ALT.apply(lambda x : len(x)) == 1)]
    # remove duplicates
    snv_df.loc[~snv_df.iloc[:, :2].duplicated(keep='first')]
    snv_df.CHROM = mut.convert_chr_back(snv_df.CHROM.astype(int))
    snv_df = snv_df.rename(columns={'CHROM': 'CONTIG',
                            'POS': 'POSITION',
                            'ref_count': 'REF_COUNT',
                            'alt_count': 'ALT_COUNT',
                            'REF':'REF_NUCLEOTIDE',
                            'ALT': 'ALT_NUCLEOTIDE'})
    sim_tumor_allelecount_out_path = os.path.join(out_dir, '{}_{}_gatk_allele.counts.tsv'.format(out_label, purity))
    snv_df[['CONTIG', 'POSITION', 'REF_COUNT', 'ALT_COUNT', 'REF_NUCLEOTIDE', 
            'ALT_NUCLEOTIDE']].to_csv(sim_tumor_allelecount_out_path, sep='\t', index=False)

    add_header_to_allelecounts(raw_gatk_allelecounts_path, sim_tumor_allelecount_out_path)

    # restrict normal counts to the tumor sites
    df = pd.read_csv(sim_normal_allelecounts_path, comment="@", sep='\t')
    df = df.merge(snv_df[['CONTIG', 'POSITION']], on = ['CONTIG', 'POSITION'], how='inner')

    norm_sim_allelecounts_out_path = os.path.join(out_dir, '{}_{}_gatk_sim_normal_allele.counts.tsv'.format(out_label, purity))
    df.to_csv(norm_sim_allelecounts_out_path, sep='\t', index=False)
 
    add_header_to_allelecounts(raw_gatk_allelecounts_path, norm_sim_allelecounts_out_path)

    # generate coverage
    sim_cov_out_path = os.path.join(out_dir, '{}_{}_gatk_sim_tumor_cov.tsv'.format(out_label, purity))
    default_profile.save_coverage_file(sim_cov_out_path, purity, coverage_tsv_path, do_parallel=parallel)

    # convert bed file into gatk hdf5 format
    convert_hapaseg_cov_to_gatk_hdf5(sim_cov_out_path, raw_gatk_coverage_path, 
                                     os.path.join(out_dir, '{}_{}_gatk_sim_tumor.frag.counts.hdf5'.format(out_label, purity)))


def generate_ascat_files(sim_profile_pickle=None,
                         purity = None,
                         normal_vcf_path=None,
                         variant_depth_path=None,
                         filtered_variants_path=None,
                         out_dir = None,
                         out_label=None,
                         parallel=False):
    
    default_profile = pd.read_pickle(sim_profile_pickle) 
     
    # generate snvs
    snv_df, correct_phase_interval_trees = default_profile.generate_snvs(normal_vcf_path, variant_depth_path, purity, do_parallel=parallel)
    snv_df = snv_df.loc[(snv_df.REF.apply(lambda x : len(x)) == 1) & (snv_df.ALT.apply(lambda x : len(x)) == 1)]

    snv_df = snv_df.rename({'CHROM':'chr', 'POS':'pos',
                            'ref_count':'t_refcount',
                            'alt_count':'t_altcount'}, axis=1)[['chr', 'pos',
                                                                'REF', 'ALT',
                                                                 't_refcount', 't_altcount',
                                                                 'adjusted_depth']]
    snv_df['chr'] = mut.convert_chr(snv_df['chr']).astype(int)
    snv_df = snv_df.loc[snv_df.adjusted_depth > 0]

    # generate fake normal using binomial with N=30
    normal_df = pd.read_csv(filtered_variants_path, sep='\t')
    normal_df['n_altcount'] = s.binom.rvs(30, normal_df['t_altcount'] / normal_df['total_reads'])
    normal_df['n_refcount'] = 30 - normal_df['n_altcount']
    normal_df = normal_df.drop(['t_refcount', 't_altcount'], axis=1)
    normal_df['chr'] = mut.convert_chr(normal_df['chr'])

    merged = snv_df.merge(normal_df, on=['chr', 'pos'], how='inner').drop_duplicates()
    # calculate ascat input logR and BAF
    merged['tumorLogR'] = merged['adjusted_depth'] / merged['total_reads']
    merged['tumorLogR'] = np.log2(merged['tumorLogR'] / merged['tumorLogR'].mean())
    # ascat passes around a normalLogR file but never actually defines these values (nor are they ever used)
    # we will make a dummy file 
    merged['normalLogR'] = 0

    merged['normalBAF'] = np.nan
    merged['tumorBAF'] = np.nan
    # for unexplained reasons ascat also randomizes A and B alleles
    mask = np.random.rand(len(merged)) > 0.5
    merged.loc[mask, 'normalBAF'] = merged.loc[mask, 'n_refcount'] / merged.loc[mask, 'total_reads']
    merged.loc[~mask, 'normalBAF'] = merged.loc[~mask, 'n_altcount'] / merged.loc[~mask, 'total_reads']
    merged.loc[mask, 'tumorBAF'] = merged.loc[mask, 't_refcount'] / merged.loc[mask, 'adjusted_depth']
    merged.loc[~mask, 'tumorBAF'] = merged.loc[~mask, 't_altcount'] / merged.loc[~mask, 'adjusted_depth']

    # switch back to string X
    merged['chr'] = merged['chr'].apply(lambda x: 'X' if x ==23 else str(x))
    merged = merged.rename({'chr':'chrs'}, axis=1)
    # ascat expects the index to be in chr_pos form
    merged = merged.set_index(merged.apply(lambda x: x.chrs + '_' + str(x.pos), axis = 1))

    # ascat returns 4 seperate files each with one column of data
    merged.rename({'tumorLogR':'sim_sample'}, axis=1)[['chrs', 'pos', 'sim_sample']].to_csv(os.path.join(out_dir, '{}_{}_ascat_tumor_LogR.txt'.format(out_label, purity)), sep='\t')
    merged.rename({'normalLogR':'sim_sample'}, axis=1)[['chrs', 'pos', 'sim_sample']].to_csv(os.path.join(out_dir, '{}_{}_ascat_normal_LogR.txt'.format(out_label, purity)), sep='\t')
    merged.rename({'tumorBAF':'sim_sample'}, axis=1)[['chrs', 'pos', 'sim_sample']].to_csv(os.path.join(out_dir, '{}_{}_ascat_tumor_BAF.txt'.format(out_label, purity)), sep='\t')
    merged.rename({'normalBAF':'sim_sample'}, axis=1)[['chrs', 'pos', 'sim_sample']].to_csv(os.path.join(out_dir, '{}_{}_ascat_normal_BAF.txt'.format(out_label, purity)), sep='\t')

            
if __name__ == "__main__":
    main()
