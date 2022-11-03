#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
from cnv_suite.compare.acr_compare import acr_compare
import numpy as np
from capy import mut, seq, plots
from hapaseg.utils import plot_chrbdy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

# acr compare bin creation fills stack over native 1k limit
# use ipython limit of 3k instead
sys.setrecursionlimit(3000)

#### method output conversion tasks ####

def convert_ascat_output(ascat_df, # ascat segments_raw output file
                         outpath # path to save converted seg file
                         ):
    
    ascat_df = pd.read_csv(ascat_df, sep='\t')
    if 'nAraw' not in ascat_df.columns or 'nAraw' not in ascat_df.columns:
        raise ValueError("expected nAraw nBraw columns. Are you sure a ASCAT segments_raw file was passed?")
    ascat_df = ascat_df.drop(['sample', 'nMajor', 'nMinor'], axis=1)
    ascat_df = ascat_df.rename({'chr':'Chromosome', 'startpos':'Start.bp',
                                'endpos':'End.bp', 'nAraw':'mu.major',
                                'nBraw':'mu.minor'}, axis=1)
    ascat_df.Chromosome = mut.convert_chr(ascat_df.Chromosome)
    
    # for now we'll just set sigma to be 1/10 of the mu value
    ascat_df.loc[:, 'sigma.major'] = ascat_df['mu.major'] / 10
    ascat_df.loc[:, 'sigma.minor'] = ascat_df['mu.minor'] / 10
    
    ascat_df.to_csv(outpath, sep='\t', index=False)


def convert_facets_output(facets_df, # facets segments file
                           outpath): # path to save converted seg file
    
    facets_df = pd.read_csv(facets_df, sep=' ')
    if 'cnlr.median' not in facets_df.columns or 'mafR' not in facets_df.columns:
        raise ValueError("expected cnlr.median and mafR columns. Are you sure a Facets segments file was passed?")
    
    facets_df['mu.major'] = np.exp2(facets_df['cnlr.median']) * np.exp(facets_df['mafR'])
    facets_df['mu.minor'] = np.exp2(facets_df['cnlr.median']) * (1/np.exp(facets_df['mafR']))

    facets_df = facets_df.rename({'chrom':'Chromosome', 'start':'Start.bp', 'end':'End.bp'}, axis=1)
    facets_df = facets_df[['Chromosome', 'Start.bp', 'End.bp', 'mu.major', 'mu.minor']]
    facets_df.Chromosome = mut.convert_chr(facets_df.Chromosome)
    
    # for now we'll just set sigma to be 1/10 of the mu value
    facets_df['sigma.major'] = facets_df['mu.major'] / 10
    facets_df['sigma.minor'] = facets_df['mu.minor']/10
    
    facets_df.to_csv(outpath, sep='\t', index=False)


def convert_gatk_output(gatk_df, # GATK finalModel.seg file
                        outpath): # path to save converted seg file
    
    gatk_df = pd.read_csv(gatk_df, sep='\t', comment="@")
    if not 'LOG2_COPY_RATIO_POSTERIOR_50' in gatk_df.columns or not 'MINOR_ALLELE_FRACTION_POSTERIOR_50' in gatk_df.columns:
        raise ValueError("expected posterior log2 copy ratio and allele fraction columns. Are you sure a GATK finalModel seg file was passed?")
    gatk_df['mu.major'] = 2 * np.exp2(gatk_df['LOG2_COPY_RATIO_POSTERIOR_50']) * (1-gatk_df['MINOR_ALLELE_FRACTION_POSTERIOR_50'])
    gatk_df['mu.minor'] = 2 * np.exp2(gatk_df['LOG2_COPY_RATIO_POSTERIOR_50']) * gatk_df['MINOR_ALLELE_FRACTION_POSTERIOR_50']
    # it is possible to infer this from the CI width but for now just use dummy value
    gatk_df.loc[:, 'sigma.minor'] = gatk_df['mu.minor'] / 10
    gatk_df.loc[:, 'sigma.major'] = gatk_df['mu.major'] / 10

    gatk_df = gatk_df.rename({'CONTIG':'Chromosome', 'START':'Start.bp', 'END':'End.bp'}, axis=1)
    gatk_df = gatk_df[['Chromosome', 'Start.bp', 'End.bp', 'mu.major', 'mu.minor', 'sigma.major', 'sigma.minor']]
    gatk_df.Chromosome = mut.convert_chr(gatk_df.Chromosome)
    gatk_df.to_csv(outpath, sep='\t', index=False)
    

def convert_hatchet_output(hatchet_df, outpath):
    pass


#### plotting method output compared to ground truth ####

def plot_output_comp(overlap_seg_file, # seg file output from acr_compare
                     ref_fasta, # reference fasta from appropriate build
                     cytoband_file, # cytoband file from appropriate build
                     MAD_score=None, # optional MAD score from acr_compare
                     savepath=None, # file path to save to if desired
                     truth_index=2): # index of ground truth segs (i.e. order of gt passed to acr_compare)
    
    seg_df = pd.read_csv(overlap_seg_file, sep='\t')
    fig = plt.figure(1, (14,8))
    ax = plt.gca()
    
    gt_mu_major = f'mu.major_{truth_index}'
    gt_mu_minor = f'mu.minor_{truth_index}'
    gt_sigma_major = f'sigma.major_{truth_index}'
    gt_sigma_minor = f'sigma.minor_{truth_index}' 
    method_index = 1 if truth_index==2 else 2
    method_major = f'mu.major_{method_index}'
    method_minor = f'mu.minor_{method_index}'
    # get gpos for all segment starts
    seg_df['start_gpos'] = seq.chrpos2gpos(seg_df['chromosome'], seg_df['Start.bp'], ref = ref_fasta)
    
    # plot ground truth segments
    for i, seg in seg_df.iterrows():
        # gt major
        ax.add_patch(Rectangle((seg['start_gpos'], seg[gt_mu_major] - 1.95 * seg[gt_sigma_major]),
                                seg['length'], seg[gt_sigma_major] * 2 * 1.95,
                                fill = True, alpha = 0.1, facecolor='r',
                                #edgecolor=(1.0, 0.0, 0.0, 0.2),
                                #ls = (0, (0,5,5,0))
                                ))
        ax.plot((seg['start_gpos'], seg['start_gpos'] + seg['length']), (seg[gt_mu_major], seg[gt_mu_major]), color='k', alpha=0.2)
        # gt minor
        ax.add_patch(Rectangle((seg['start_gpos'], seg[gt_mu_minor] - 1.95 * seg[gt_sigma_minor]),
                                seg['length'], seg[gt_sigma_minor] * 2 * 1.95,
                                fill = True, alpha = 0.1, facecolor='b',
    #                            edgecolor=(0.0, 0.0, 1.0, 0.2),
     #                           ls = (0, (5,0,0,5))
                                ))
        ax.plot((seg['start_gpos'], seg['start_gpos'] + seg['length']), (seg[gt_mu_minor], seg[gt_mu_minor]), color='k', alpha=0.2)

    # plot method vals
    method_segs = seg_df.loc[~seg_df.unique]
    
    #plot linesegments
    major_line_segs = [((x['start_gpos'], x[method_major]), (x['start_gpos'] + x['length'], x[method_major])) for i, x in method_segs.iterrows()]
    lc = LineCollection(major_line_segs, colors='r', alpha=0.7, linewidths=3)
    ax.add_collection(lc)
    
    minor_line_segs = [((x['start_gpos'], x[method_minor]), (x['start_gpos'] + x['length'], x[method_minor])) for i, x in method_segs.iterrows()]
    lc = LineCollection(minor_line_segs, colors='b', alpha=0.7, linewidths=3)
    ax.add_collection(lc)
    
    # plot symbols as midpoint mus
    #ax.scatter(method_segs['start_gpos'] + method_segs['length'] /2, method_segs[method_major], marker="1", color='r', alpha=0.7)
    #ax.scatter(method_segs['start_gpos'] + method_segs['length'] /2, method_segs[method_minor], marker="2", color='b', alpha=0.7)
    
    # plot cytobands
    plot_chrbdy(cytoband_file)
    
    if MAD_score is not None:
        plt.title(f'CNV calling ground truth comparison MAD: {np.around(MAD_score, 3)}')
    
    if savepath is not None:
        plt.savefig(savepath)


#### method input plotting ####
# as a sanity check, plot the allelic coverage / alt snp counts across the genome
# derived from the method input files

def plot_facets_sim_input(sim_facets_input_file, # facets input counts file
                          ref_genome, # path to ref genome fasta
                          cytoband_file, # path to ref cytoband file
                          savepath): # path to save plot
    facets_input_counts = pd.read_csv(sim_facets_input_file)    
    facets_input_counts.loc[:, "gpos"] = seq.chrpos2gpos(facets_input_counts['Chromosome'], facets_input_counts['Position'], ref = ref_genome)
    fig = plt.figure(1, figsize=(14,8))
    
    facets_counts = facets_input_counts['File2R'] + facets_input_counts['File2A']
    facets_alt_imb = facets_input_counts['File2A']/ facets_counts
    # filter out homozygous vars
    facets_hets = facets_alt_imb < 0.99
    # subsample for visability
    sample_hets = np.random.choice(np.flatnonzero(facets_hets), 100000)
    plt.figure(figsize=(14,8))
    plots.pixplot(facets_input_counts['gpos'].iloc[sample_hets], facets_counts.iloc[sample_hets] * facets_alt_imb.iloc[sample_hets], alpha=0.3)
    plt.title('Facets het site alt coverage')
    plt.ylabel('alt count')
    # add chr lines
    plot_chrbdy(cytoband_file)

    plt.savefig(savepath)


def plot_gatk_sim_input(sim_gatk_cov_tsv, # gatk simulated tumor coverage data in hapaseg tsv format
                        sim_gatk_acounts, # gatk simulated tumor allele counts
                        ref_genome, # path to ref genome fasta
                        cytoband_file, # path to cytoband file
                        savepath): #path to save plot

    gatk_cov_df = pd.read_csv(sim_gatk_cov_tsv, sep='\t')
    gatk_aimb_df = pd.read_csv(sim_gatk_acounts, sep='\t', comment="@")
    gatk_aimb_df = gatk_aimb_df.rename({'CONTIG':'chr', 'POSITION':'pos'}, axis=1)
    gatk_aimb_df['chr'] = mut.convert_chr(gatk_aimb_df['chr'])
    gatk_aimb_df['aimb'] = gatk_aimb_df['ALT_COUNT'] / (gatk_aimb_df['REF_COUNT'] + gatk_aimb_df['ALT_COUNT'])

    mut.map_mutations_to_targets(gatk_aimb_df, gatk_cov_df)
    gatk_aimb_df = gatk_aimb_df.loc[(gatk_aimb_df.aimb >0.01) & (gatk_aimb_df.aimb < 0.99)]
    agg_counts = gatk_aimb_df.groupby('targ_idx').agg({'REF_COUNT':sum, 'ALT_COUNT':sum})
    
    gatk_acov_df = gatk_cov_df.join(agg_counts, how='inner')
    gatk_acov_df['aimb'] = gatk_acov_df['ALT_COUNT'] / (gatk_acov_df['REF_COUNT'] + gatk_acov_df['ALT_COUNT'])
    
    # add midpoint gpos for plotting
    gatk_acov_df['gpos'] = seq.chrpos2gpos(gatk_acov_df['chr'], (gatk_acov_df['start'] + gatk_acov_df['end']) / 2, ref=ref_genome)
    plt.figure(figsize=(14,8))
    plots.pixplot(gatk_acov_df['gpos'], gatk_acov_df['covcorr'] * gatk_acov_df['aimb'], alpha = 0.05)
    
    plt.title('GATK allelic coverage')
    plt.ylabel('allelic_coverage')

    plot_chrbdy(cytoband_file)
    # gatk does its own outlier removal so limit ylim to increase visability
    plt.ylim([0, np.quantile(gatk_acov_df['covcorr'] * gatk_acov_df['aimb'], 0.5) * 8])

    plt.savefig(savepath)


def plot_ascat_sim_input(sim_ascat_t_logr, # ascat simulated tumor logR tsv
                         sim_ascat_t_baf, # ascat simulated tumor BAF tsv
                         ref_genome, # path to ref genome fasta
                         cytoband_file, # path to cytoband file
                         savepath): # path to save plot

    logr_df = pd.read_csv(sim_ascat_t_logr, sep='\t', usecols=[1,2,3], low_memory=False) 
    logr_df = logr_df.rename({logr_df.columns[-1]: 'logr'}, axis=1)
    baf_df = pd.read_csv(sim_ascat_t_baf, sep='\t', usecols=[1,2,3], low_memory=False)    
    baf_df = baf_df.rename({baf_df.columns[-1]: 'aimb'}, axis=1)
    baf_df = baf_df.loc[(baf_df.aimb < 0.99) & (baf_df.aimb > 0.01)]
    
    ascat_df = logr_df.merge(baf_df, on=['chrs', 'pos'], how='inner')
    ascat_df['gpos'] = seq.chrpos2gpos(mut.convert_chr(ascat_df['chrs']), ascat_df['pos'], ref=ref_genome)
    plt.figure(figsize=(14,8))
    plots.pixplot(ascat_df['gpos'], np.exp2(ascat_df['logr']) * ascat_df['aimb'], alpha = 0.05)
    plot_chrbdy(cytoband_file)
    
    plt.title('ASCAT hetsite alt coverage')
    plt.ylabel('alt counts')
    plt.savefig(savepath)
    

#### downstream comparison analysis functions ####
# converts method output to common segfile format
## computes MAD scores
### plots method segs and ground truth segments
#### plots method inputs for sanity checking

def facets_downstream_analysis(facets_sim_input_file, # path to facets input counts file
                               facets_output_segs, # path to facets output segments file
                               gt_segfile, # path to groundtruth segfile
                               sample_name, # sample name. should be in sampleLabel_purity format
                               ref_fasta, # reference fasta
                               cytoband_file, # reference cytoband file
                               outdir='./' # directory to save outputs
                               ):
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # convert method output to seg format
    converted_seg_outpath = os.path.join(outdir, f'{sample_name}_facets_converted_seg_file.tsv')
    convert_facets_output(facets_output_segs, converted_seg_outpath)

    # compare seg file to ground truth + compute MAD
    mad_score, opt_scale_factor, opt_purity, non_ov_len, ov_len, seg_df = acr_compare(converted_seg_outpath, gt_segfile, fit_params=True)
    comparison_segfile_outpath = os.path.join(outdir, f'{sample_name}_facets_comparison_segfile.tsv')
    seg_df.to_csv(comparison_segfile_outpath, sep='\t', index=False)
    
    # save other data
    mad_results_outpath = os.path.join(outdir, f'{sample_name}_facets_comparison_results.txt')
    mad_res_df = pd.DataFrame({'mad_score':[mad_score],
                               'optimal_scale_factor':[opt_scale_factor],
                               'optimal_purity':[opt_purity],
                               'non_overlap_length':[non_ov_len],
                               'overlap_length':[ov_len]})
    mad_res_df.to_csv(mad_results_outpath, sep='\t', index=False)

    # plot comparison to gt
    plot_savepath = os.path.join(outdir, f'{sample_name}_facets_comparison_plot.png')
    plot_output_comp(comparison_segfile_outpath, ref_fasta, cytoband_file, mad_score, plot_savepath)

    # plot inputs for sanity check
    plot_inputs_savepath = os.path.join(outdir, f'{sample_name}_facets_input_plot.png')
    plot_facets_sim_input(facets_sim_input_file, ref_fasta, cytoband_file, plot_inputs_savepath)


def ascat_downstream_analysis(ascat_sim_t_logr, # path to ascat sim tumor logr file
                              ascat_sim_t_baf, # path to ascat sim tumor baf file
                              ascat_output_segs, # path to ascat segments_raw output file
                              gt_segfile, # path to groundtruth segfile
                              sample_name, # sample name. should be in sampleLabel_purity format
                              ref_fasta, # reference fasta
                              cytoband_file, # reference cytoband file
                              outdir='./' # directory to save outputs
                              ):
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
   
    # convert method output to seg format
    converted_seg_outpath = os.path.join(outdir, f'{sample_name}_ascat_converted_seg_file.tsv')
    convert_ascat_output(ascat_output_segs, converted_seg_outpath)

    # compare seg file to ground truth + compute MAD
    mad_score, opt_scale_factor, opt_purity, non_ov_len, ov_len, seg_df = acr_compare(converted_seg_outpath, gt_segfile, fit_params=True)
    comparison_segfile_outpath = os.path.join(outdir, f'{sample_name}_ascat_comparison_segfile.tsv')
    seg_df.to_csv(comparison_segfile_outpath, sep='\t', index=False)
    
    # save other data
    mad_results_outpath = os.path.join(outdir, f'{sample_name}_ascat_comparison_results.txt')
    mad_res_df = pd.DataFrame({'mad_score':[mad_score],
                               'optimal_scale_factor':[opt_scale_factor],
                               'optimal_purity':[opt_purity],
                               'non_overlap_length':[non_ov_len],
                               'overlap_length':[ov_len]})
    mad_res_df.to_csv(mad_results_outpath, sep='\t', index=False)

    # plot comparison to gt
    plot_savepath = os.path.join(outdir, f'{sample_name}_ascat_comparison_plot.png')
    plot_output_comp(comparison_segfile_outpath, ref_fasta, cytoband_file, mad_score, plot_savepath)

    # plot inputs for sanity check
    plot_inputs_savepath = os.path.join(outdir, f'{sample_name}_ascat_input_plot.png')
    plot_ascat_sim_input(ascat_sim_t_logr, ascat_sim_t_baf, ref_fasta, cytoband_file, plot_inputs_savepath)


def gatk_downstream_analysis(sim_gatk_cov_tsv, # gatk simulated tumor coverage data in hapaseg tsv format
                             sim_gatk_acounts, # gatk simulated tumor allele counts 
                             gatk_output_segs, # path to gatk modelFinal.seg file
                             gt_segfile, # path to groundtruth segfile
                             sample_name, # sample name. should be in sampleLabel_purity format
                             ref_fasta, # reference fasta
                             cytoband_file, # reference cytoband file
                             outdir='./' # directory to save outputs
                             ):
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # convert method output to seg format
    converted_seg_outpath = os.path.join(outdir, f'{sample_name}_gatk_converted_seg_file.tsv')
    convert_gatk_output(gatk_output_segs, converted_seg_outpath)

    # compare seg file to ground truth + compute MAD
    mad_score, opt_scale_factor, opt_purity, non_ov_len, ov_len, seg_df = acr_compare(converted_seg_outpath, gt_segfile, fit_params=True)
    comparison_segfile_outpath = os.path.join(outdir, f'{sample_name}_gatk_comparison_segfile.tsv')
    seg_df.to_csv(comparison_segfile_outpath, sep='\t', index=False)
    
    # save other acr_compare data
    mad_results_outpath = os.path.join(outdir, f'{sample_name}_gatk_comparison_results.txt')
    mad_res_df = pd.DataFrame({'mad_score':[mad_score],
                               'optimal_scale_factor':[opt_scale_factor],
                               'optimal_purity':[opt_purity],
                               'non_overlap_length':[non_ov_len],
                               'overlap_length':[ov_len]})
    mad_res_df.to_csv(mad_results_outpath, sep='\t', index=False)

    # plot comparison to gt
    plot_savepath = os.path.join(outdir, f'{sample_name}_gatk_comparison_plot.png')
    plot_output_comp(comparison_segfile_outpath, ref_fasta, cytoband_file, mad_score, plot_savepath)

    # plot inputs for sanity check
    plot_inputs_savepath = os.path.join(outdir, f'{sample_name}_gatk_input_plot.png')
    plot_gatk_sim_input(sim_gatk_cov_tsv, sim_gatk_acounts,  ref_fasta, cytoband_file, plot_inputs_savepath)

def hatchet_downstream_analysis():
    pass

def hapaseg_downstream_analysis(hapaseg_seg_file, # hapaseg output seg file
                                gt_segfile, # path to ground truth segfile
                                sample_name, # sample name. should be in sampleLabel_purity format
                                ref_fasta, # reference fasta
                                cytoband_file, # reference cytoband file
                                outdir='./' # directory to save outputs
                                ):
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # hapaseg output is already in proper format so doesnt need to be converted
    # output is on same scale as ground truth so no parameter fitting necessary
    mad_score, opt_scale_factor, opt_purity, non_ov_len, ov_len, seg_df = acr_compare(hapaseg_seg_file, gt_segfile, fit_params=False)
    comparison_segfile_outpath = os.path.join(outdir, f'{sample_name}_hapaseg_comparison_segfile.tsv')
    seg_df.to_csv(comparison_segfile_outpath, sep='\t', index=False)
    
    # save other acr compare data
    mad_results_outpath = os.path.join(outdir, f'{sample_name}_hapaseg_comparison_results.txt')
    mad_res_df = pd.DataFrame({'mad_score':[mad_score],
                               'optimal_scale_factor':[opt_scale_factor],
                               'optimal_purity':[opt_purity],
                               'non_overlap_length':[non_ov_len],
                               'overlap_length':[ov_len]})
    mad_res_df.to_csv(mad_results_outpath, sep='\t', index=False)
    
    # plot comparison to gt
    plot_savepath = os.path.join(outdir, f'{sample_name}_hapaseg_comparison_plot.png')
    plot_output_comp(comparison_segfile_outpath, ref_fasta, cytoband_file, mad_score, plot_savepath)
    
    # no need to plot sanity checks since hapaseg intermediate files should suffice

# CLI interface
def parse_args():
    
    parser = argparse.ArgumentParser(description = "post-process cnv method outputs and compare to ground truth")
    parser.add_argument("--ref_fasta", required=True, help="path to reference fasta")
    parser.add_argument("--cytoband_file", required=True, help="path to reference cytoband file")
    parser.add_argument("--sample_name", required=True, help="sample name in sampleLabel_purity format")
    parser.add_argument("--ground_truth_segfile", required=True, help="path to ground truth segfile")
    parser.add_argument("--outdir", default='./', help="directory to save outputs to")
    subparsers = parser.add_subparsers(dest="command")    

    hapaseg_post = subparsers.add_parser("hapaseg", help="run hapaseg post-processing")
    hapaseg_post.add_argument("--hapaseg_seg_file", help="hapaseg seg file output")

    facets_post = subparsers.add_parser("facets", help="run facets post-processing")
    facets_post.add_argument("--facets_input_counts", help="facets simulated input counts file")
    facets_post.add_argument("--facets_seg_file", help="facets output seg file")

    ascat_post = subparsers.add_parser("ascat", help="run ascat post-processing")
    ascat_post.add_argument("--ascat_t_logr", help="ascat simulated tumor logR file")
    ascat_post.add_argument("--ascat_t_baf", help="ascat simulated tumor BAF file")
    ascat_post.add_argument("--ascat_seg_file", help="ascat raw_segments output file")

    gatk_post = subparsers.add_parser("gatk", help="run gatk post-processing")
    gatk_post.add_argument("--gatk_sim_cov_input", help="gatk simulated coverage in tsv format")
    gatk_post.add_argument("--gatk_sim_acounts", help="gatk simulated allelic coverage")
    gatk_post.add_argument("--gatk_seg_file", help="gatk modelFinal seg file")
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()    
    
    if args.command == "hapaseg":
        print("running downstream analyses on hapaseg", flush=True)
        hapaseg_downstream_analysis(args.hapaseg_seg_file,
                                    args.ground_truth_segfile,
                                    args.sample_name,
                                    args.ref_fasta,
                                    args.cytoband_file,
                                    args.outdir)
    
    elif args.command == "facets":
        print("running downstream analyses on facets", flush=True)
        facets_downstream_analysis(args.facets_input_counts,
                                   args.facets_seg_file,
                                   args.ground_truth_segfile,
                                   args.sample_name,
                                   args.ref_fasta,
                                   args.cytoband_file,
                                   args.outdir)
        
    elif args.command == "ascat":
        print("running downstream analyses on ascat", flush=True)        
        ascat_downstream_analysis(args.ascat_t_logr,
                                  args.ascat_t_baf,
                                  args.ascat_seg_file,
                                  args.ground_truth_segfile,
                                  args.sample_name,
                                  args.ref_fasta,
                                  args.cytoband_file,
                                  args.outdir)

    elif args.command == "gatk":
        print("running downstream analyses on gatk", flush=True)
        gatk_downstream_analysis(args.gatk_sim_cov_input,
                                 args.gatk_sim_acounts,
                                 args.gatk_seg_file,
                                 args.ground_truth_segfile,
                                 args.sample_name,
                                 args.ref_fasta,
                                 args.cytoband_file,
                                 args.outdir)
        
    else:
        raise ValueError(f"did not recognize command {args.command}")


if __name__ == "__main__":
    main()
