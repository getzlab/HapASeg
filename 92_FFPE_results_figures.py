import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hapaseg.utils import plot_chrbdy
from capy import mut, seq, plots
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from cnv_suite.visualize.plot_cnv_profile import *

# simulated tumor of choice: benchmarking_profile_43819_35_0.7
## ffpe data:
ffpe_hapaseg_seg_comp_file = '/mnt/nfs/benchmarking_dir/FFPE_CH1022_benchmarking_profile_43819_35_0.7/Downstream_HapASeg_Analysis__2023-02-18--21-35-12_ftddxuy_qbvxcjy_crgrby0culosw/jobs/0/workspace/FFPE_CH1022_benchmarking_profile_43819_35_hapaseg_comparison_segfile.tsv'
ffpe_hapaseg_results_file = '/mnt/nfs/benchmarking_dir/FFPE_CH1022_benchmarking_profile_43819_35_0.7/Downstream_HapASeg_Analysis__2023-02-18--21-35-12_ftddxuy_qbvxcjy_crgrby0culosw/jobs/0/workspace/FFPE_CH1022_benchmarking_profile_43819_35_hapaseg_comparison_results.txt'
ffpe_gatk_seg_comp_file = '/mnt/nfs/benchmarking_dir/FFPE_CH1022_benchmarking_profile_43819_35_0.7/Downstream_GATK_Analysis__2023-02-18--21-40-07_dxgqioa_qbvxcjy_a2d3weicqlpg2/jobs/0/workspace/FFPE_CH1022_benchmarking_profile_43819_35_gatk_comparison_segfile.tsv'
ffpe_gatk_results_file = '/mnt/nfs/benchmarking_dir/FFPE_CH1022_benchmarking_profile_43819_35_0.7/Downstream_GATK_Analysis__2023-02-18--21-40-07_dxgqioa_qbvxcjy_a2d3weicqlpg2/jobs/0/workspace/FFPE_CH1022_benchmarking_profile_43819_35_gatk_comparison_results.txt'

ffpe_hapaseg_mad_score = pd.read_csv(ffpe_hapaseg_results_file, sep='\t')['mad_score'][0]
ffpe_gatk_mad_score = pd.read_csv(ffpe_gatk_results_file, sep='\t')['mad_score'][0]

## FF data:
ff_hapaseg_seg_comp_file = '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_43819_35_0.7/Downstream_HapASeg_Analysis__2023-02-18--21-32-12_ftddxuy_qbvxcjy_rv3ydnblldwvs/jobs/0/workspace/FF_benchmarking_profile_43819_35_hapaseg_comparison_segfile.tsv'
ff_hapaseg_results_file = '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_43819_35_0.7/Downstream_HapASeg_Analysis__2023-02-18--21-32-12_ftddxuy_qbvxcjy_rv3ydnblldwvs/jobs/0/workspace/FF_benchmarking_profile_43819_35_hapaseg_comparison_results.txt'
ff_gatk_seg_comp_file = '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_43819_35_0.7/Downstream_GATK_Analysis__2023-02-18--21-33-09_dxgqioa_qbvxcjy_dbstpo3qezihy/jobs/0/workspace/FF_benchmarking_profile_43819_35_gatk_comparison_segfile.tsv'
ff_gatk_results_file = '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_43819_35_0.7/Downstream_GATK_Analysis__2023-02-18--21-33-09_dxgqioa_qbvxcjy_dbstpo3qezihy/jobs/0/workspace/FF_benchmarking_profile_43819_35_gatk_comparison_results.txt'

ff_hapaseg_mad_score = pd.read_csv(ff_hapaseg_results_file, sep='\t')['mad_score'][0]
ff_gatk_mad_score = pd.read_csv(ff_gatk_results_file, sep='\t')['mad_score'][0]

ref_fasta = '/home/opriebe/data/ref/hg38/GRCh38.d1.vd1.fa'
cytoband_file = '/home/opriebe/data/ref/hg38/cytoBand.txt'


# modifying the comparison plot code to make segments more visible
def plot_output_comp(overlap_seg_file, # seg file output from acr_compare
                     ref_fasta, # reference fasta from appropriate build
                     cytoband_file, # cytoband file from appropriate build
                     MAD_score=None, # optional MAD score from acr_compare
                     savepath=None, # file path to save to if desired
                     truth_index=2): # index of ground truth segs (i.e. order of gt passed to acr_compare)
    
    seg_df = pd.read_csv(overlap_seg_file, sep='\t')
    fig = plt.figure(figsize=(14,8))
    ax = plt.gca()
    
    gt_mu_major = f'mu.major_{truth_index}'
    gt_mu_minor = f'mu.minor_{truth_index}'
    gt_sigma_major = f'sigma.major_{truth_index}'
    gt_sigma_minor = f'sigma.minor_{truth_index}' 
    method_index = 1 if truth_index==2 else 2
    method_major = f'mu.major_{method_index}'
    method_minor = f'mu.minor_{method_index}'
    # get gpos for all segment starts
    seg_df['start_gpos'] = seq.chrpos2gpos(seg_df['Chromosome'], seg_df['Start.bp'], ref = ref_fasta)
    
    # plot ground truth segments
    for i, seg in seg_df.iterrows():
        # gt major
        if not seg['unique']:
            ax.add_patch(Rectangle((seg['start_gpos'], seg[gt_mu_major] - 1.95 * seg[gt_sigma_major]),
                                    seg['length'], seg[gt_sigma_major] * 2 * 1.95,
                                    fill = True, alpha = 0.1, facecolor='r',
                                    ))
            ax.plot((seg['start_gpos'], seg['start_gpos'] + seg['length']), (seg[gt_mu_major], seg[gt_mu_major]), color='k', alpha=0.2)
            # gt minor
            ax.add_patch(Rectangle((seg['start_gpos'], seg[gt_mu_minor] - 1.95 * seg[gt_sigma_minor]),
                                    seg['length'], seg[gt_sigma_minor] * 2 * 1.95,
                                    fill = True, alpha = 0.1, facecolor='b',
                                    ))
            ax.plot((seg['start_gpos'], seg['start_gpos'] + seg['length']), (seg[gt_mu_minor], seg[gt_mu_minor]), color='k', alpha=0.2)

    # plot method vals
    method_segs = seg_df.loc[~seg_df.unique]
    
    #plot linesegments
    major_line_segs = [((x['start_gpos'] - 5e5, x[method_major]), (x['start_gpos'] + x['length'] + 1e6, x[method_major])) for i, x in method_segs.iterrows()]
    lc = LineCollection(major_line_segs, colors='r', alpha=1, linewidths=5)
    ax.add_collection(lc)
    
    minor_line_segs = [((x['start_gpos'] - 5e5, x[method_minor]), (x['start_gpos'] + x['length'] + 1e6, x[method_minor])) for i, x in method_segs.iterrows()]
    lc = LineCollection(minor_line_segs, colors='b', alpha=1, linewidths=5)
    ax.add_collection(lc)
    
    # plot cytobands
    plot_chrbdy(cytoband_file)
    
    plt.ylabel('Coverage')
    plt.xlabel('Chromosome')
    
    if MAD_score is not None:
        plt.title(f'CNV calling ground truth comparison MAD: {np.around(MAD_score, 3)}')
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

# plot ffpe
plot_output_comp(ffpe_hapaseg_seg_comp_file, ref_fasta, cytoband_file, ffpe_hapaseg_mad_score, savepath='./final_figures/FFPE_figure_plots/FFPE_hapaseg_results.pdf')
plot_output_comp(ffpe_gatk_seg_comp_file, ref_fasta, cytoband_file, ffpe_gatk_mad_score, savepath='./final_figures/FFPE_figure_plots/FFPE_gatk_results.pdf')
plot_output_comp(ffpe_hapaseg_seg_comp_file, ref_fasta, cytoband_file, ffpe_hapaseg_mad_score, savepath='./final_figures/FFPE_figure_plots/FFPE_hapaseg_results.png')
plot_output_comp(ffpe_gatk_seg_comp_file, ref_fasta, cytoband_file, ffpe_gatk_mad_score, savepath='./final_figures/FFPE_figure_plots/FFPE_gatk_results.png')

# plot FF
plot_output_comp(ff_hapaseg_seg_comp_file, ref_fasta, cytoband_file, ff_hapaseg_mad_score, savepath='./final_figures/FFPE_figure_plots/FF_hapaseg_results.pdf')
plot_output_comp(ff_hapaseg_seg_comp_file, ref_fasta, cytoband_file, ff_hapaseg_mad_score, savepath='./final_figures/FFPE_figure_plots/FF_hapaseg_results.png')
plot_output_comp(ff_gatk_seg_comp_file, ref_fasta, cytoband_file, ff_gatk_mad_score, savepath='./final_figures/FFPE_figure_plots/FF_gatk_results.pdf')
plot_output_comp(ff_gatk_seg_comp_file, ref_fasta, cytoband_file, ff_gatk_mad_score, savepath='./final_figures/FFPE_figure_plots/FF_gatk_results.png')


# plot allelic coverage

# can make these different colors if we want
#ccan make these different colors if we want
## FF first
plt.rcParams.update({'font.size': 14})
## load coverage
cov_df = pd.read_pickle('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_43819_35_0.7/Hapaseg_prepare_coverage_mcmc__2023-02-07--02-10-24_rly1scy_qbvxcjy_0lpr5qrcglhlm/jobs/0/workspace/cov_df.pickle')
cov_df['aimb'] = cov_df['min_count'] / (cov_df['maj_count'] + cov_df['min_count'])
## sample from coverage bins and plot allelic coverage
fig = plt.figure(figsize=(14,8))
# sampling approach 
#sm = cov_df.sample(100000)
#plt.scatter(sm.start_g.values, sm.aimb.values * sm.fragcorr.values, marker='.', alpha=0.1, s=1, color='tab:blue')
#plt.scatter(sm.start_g.values, (1-sm.aimb.values) * sm.fragcorr.values, marker='.', alpha=0.1, s=1, color='tab:blue')
# every pixel
plots.pixplot(cov_df.start_g.values, cov_df.aimb.values * cov_df.fragcorr.values, alpha=0.2, color='tab:blue')
plots.pixplot(cov_df.start_g.values, (1- cov_df.aimb.values) * cov_df.fragcorr.values, alpha=0.2, color='tab:blue')

plot_chrbdy(cytoband_file)
plt.ylabel('Allelic coverage')
plt.xlabel('Chromosome')
plt.savefig('./final_figures/FFPE_figure_plots/FF_allelic_coverage.pdf')
plt.savefig('./final_figures/FFPE_figure_plots/FF_allelic_coverage.png', bbox_inches='tight')


## FFPE
## load coverage
cov_df = pd.read_pickle('/mnt/nfs/benchmarking_dir/FFPE_CH1022_benchmarking_profile_43819_35_0.7/Hapaseg_prepare_coverage_mcmc__2023-02-07--00-52-50_qojemzq_qbvxcjy_gkkf4hxvralhy/jobs/0/workspace/cov_df.pickle')
cov_df['aimb'] = cov_df['min_count'] / (cov_df['maj_count'] + cov_df['min_count'])
## sample from coverage bins and plot allelic coverage
fig = plt.figure(figsize=(14,8))
## sampling
#sm = cov_df.sample(100000)
#plt.scatter(sm.start_g.values, sm.aimb.values * sm.fragcorr.values, marker='.', alpha=0.1, s=1, color='tab:blue')
#plt.scatter(sm.start_g.values, (1-sm.aimb.values) * sm.fragcorr.values, marker='.', alpha=0.1, s=1, color='tab:blue')
# every pixel
plots.pixplot(cov_df.start_g.values, cov_df.aimb.values * cov_df.fragcorr.values, alpha=0.2, color='tab:blue')
plots.pixplot(cov_df.start_g.values, (1- cov_df.aimb.values) * cov_df.fragcorr.values, alpha=0.2, color='tab:blue')
plot_chrbdy(cytoband_file)
plt.ylabel('Allelic coverage')
plt.xlabel('Chromosome')
plt.savefig('./final_figures/FFPE_figure_plots/FFPE_allelic_coverage.pdf')
plt.savefig('./final_figures/FFPE_figure_plots/FFPE_allelic_coverage.png', bbox_inches='tight')


# plot ground truth karyotype
fig = plt.figure(figsize=(14,8))
ax = plt.gca()
sim_profile = pd.read_pickle('/home/opriebe/data/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_43819_35.pickle')
plot_acr_static(sim_profile.cnv_profile_df, ax = ax, csize=sim_profile.csize, min_seg_lw=5, y_upper_lim = 8)
plt.savefig('./final_figures/FFPE_figure_plots/sim_tumor_karyotype.pdf')
plt.savefig('./final_figures/FFPE_figure_plots/sim_tumor_karyotype.png', bbox_inches='tight')
