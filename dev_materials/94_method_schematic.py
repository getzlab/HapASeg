import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hapaseg.utils import plot_chrbdy
from capy import mut, seq, plots
from cnv_suite.visualize.plot_cnv_profile import *
import distinctipy

cytoband_file = '/home/opriebe/data/ref/hg38/cytoBand.txt'
b_profile_root = '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7'

# allelic imbalance
fig = plt.figure(1, figsize=(14,8))
snps_df = pd.read_pickle(os.path.join(b_profile_root, 'Hapaseg_allelic_DP__2023-02-14--08-03-46_fg035ei_qbvxcjy_sj5dwaahwcbxi/jobs/0/workspace/all_SNPs.pickle'))
snps_df['aimb'] = snps_df['min'] / (snps_df['min'] + snps_df['maj'])
# filter to chromosomes of interest
tmp = snps_df.loc[snps_df.chr.isin([9,10,11,12])]
plots.pixplot(tmp.pos_gp.values, tmp.aimb.values, alpha=0.3, color='tab:blue')
plots.pixplot(tmp.pos_gp.values, 1 - tmp.aimb.values, alpha=0.3, color='tab:blue')
plot_chrbdy(cytoband_file)
plt.xlim([tmp.pos_gp.min(), tmp.pos_gp.max()])
plt.ylabel('VAF')
plt.xlabel('Chromosome')
plt.savefig('./final_figures/method_schematic/raw_VAF.png', bbox_inches='tight')

# allelic DP
from hapaseg.allelic_DP import *
fig = plt.figure(2, figsize=(14,8))
dp_obj = load_DP_object_from_outputs('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_allelic_DP__2023-02-14--08-03-46_fg035ei_qbvxcjy_sj5dwaahwcbxi/jobs/0/workspace/all_SNPs.pickle', '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_allelic_DP__2023-02-14--08-03-46_fg035ei_qbvxcjy_sj5dwaahwcbxi/jobs/0/workspace/allelic_DP_SNP_clusts_and_phase_assignments.npz', '/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_allelic_DP__2023-02-14--08-03-46_fg035ei_qbvxcjy_sj5dwaahwcbxi/jobs/0/workspace/segmentations.pickle')
# hardcode colors 
def new_get_colors():
    return np.array(plt.get_cmap('Dark2').colors)[[0,2,3,1,4,5,6,7]][[6, 7, 3, 2, 1, 0, 5, 4]]
dp_obj.get_colors = new_get_colors
dp_obj.visualize_segs(f=fig, show_snps=True, chroms=[9,10,11,12])
plot_chrbdy(cytoband_file)
plt.xlim([*dp_obj.S.loc[dp_obj.S["chr"].isin([9,10,11,12]), "pos_gp"].iloc[[0, -1]]])
plt.ylabel('Allelic imbalance')
plt.xlabel('Chromosome')
plt.savefig('./final_figures/method_schematic/ADP_results.png', bbox_inches='tight')


# raw coverage
fig = plt.figure(3, figsize=(14,8))
cov_df = pd.read_pickle('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_prepare_coverage_mcmc__2023-02-14--08-13-23_rly1scy_qbvxcjy_wmwfcxbmt3m5k/jobs/0/workspace/cov_df.pickle')
filt = cov_df.loc[cov_df.chr.isin([9,10,11,12])]
plots.pixplot(filt.start_g, filt.fragcorr, alpha=0.30, color='tab:blue')
plot_chrbdy(cytoband_file)
plt.xlim([filt.start_g.min(), filt.start_g.max()])
plt.xlabel('Chromosome')
plt.ylabel('Raw coverage')
plt.savefig('./final_figures/method_schematic/raw_coverage.png', bbox_inches='tight')

# corrected coverage
fig = plt.figure(4, figsize=(14,8))
covar_columns = sorted(cov_df.columns[cov_df.columns.str.contains("(?:^C_.*_l?z$|C_log_len)")])
C = np.c_[cov_df[covar_columns]]
mcmc_data = np.load('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_collect_coverage_mcmc__2023-02-14--08-54-57_pgl2efa_qbvxcjy_43ddcqtqtyxxu/jobs/0/workspace/cov_mcmc_collected_data.npz')
beta = mcmc_data['beta']
residuals = np.exp(np.log(cov_df.fragcorr.values) - (C @ beta).flatten())
filt_residuals = residuals[cov_df.chr.isin([9,10,11,12])]
plots.pixplot(filt.start_g, filt_residuals, alpha=0.30, color='tab:blue')
plot_chrbdy(cytoband_file)
plt.xlim([filt.start_g.min(), filt.start_g.max()])
plt.xlabel('Chromosome')
plt.ylabel('Corrected coverage')
plt.savefig('./final_figures/method_schematic/corrected_coverage.png', bbox_inches='tight')

def _get_color_palette(num_colors):
    base_colors = np.array([
          [0.368417, 0.506779, 0.709798],
          [0.880722, 0.611041, 0.142051],
          [0.560181, 0.691569, 0.194885],
          [0.922526, 0.385626, 0.209179],
          [0.528488, 0.470624, 0.701351],
          [0.772079, 0.431554, 0.102387],
          [0.363898, 0.618501, 0.782349],
          [1, 0.75, 0],
          [0.647624, 0.37816, 0.614037],
          [0.571589, 0.586483, 0.],
          [0.915, 0.3325, 0.2125],
          [0.400822, 0.522007, 0.85],
          [0.972829, 0.621644, 0.073362],
          [0.736783, 0.358, 0.503027],
          [0.280264, 0.715, 0.429209]
        ])
    extra_colors = np.array(
          distinctipy.distinctipy.get_colors(
            num_colors - base_colors.shape[0],
            exclude_colors = [list(x) for x in np.r_[np.c_[0, 0, 0], np.c_[1, 1, 1], np.c_[0.5, 0.5, 0.5], np.c_[1, 0, 1], base_colors]],
        rng = 1234
          )
        )
    return np.r_[base_colors, extra_colors if extra_colors.size > 0 else np.empty([0, 3])]


# coverage mcmc
acdp_df = pd.read_pickle('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_acdp_generate_df__2023-02-14--08-58-03_ntuaqpq_qbvxcjy_tzevrhctrqch2/jobs/0/workspace/acdp_df.pickle')
acdp_df  = acdp_df.loc[acdp_df.allele == -1]
colors = plt.get_cmap("tab10").colors
C = np.c_[acdp_df[covar_columns]]
residuals = np.exp(np.log(acdp_df.fragcorr.values) - (C @ beta).flatten())
filt = acdp_df.loc[acdp_df.chr.isin([9,10,11,12])]
#colors = _get_color_palette(len(filt.segment_ID.unique()) + 3)
seg_cmap = dict(zip(filt.segment_ID.value_counts().index, range(len(filt.segment_ID.unique()))))
fig = plt.figure(5, figsize=(14,8))
for i in filt.segment_ID.unique():
    tmp = filt.loc[filt.segment_ID == i]
    resi = residuals[acdp_df.chr.isin([9,10,11,12])][filt.segment_ID == i]
    plots.pixplot(tmp.start_g, resi, alpha=0.5, color = colors[seg_cmap[i] % len(colors)])
plot_chrbdy(cytoband_file)
plt.xlim([filt.start_g.min(), filt.start_g.max()])
plt.xlabel('Chromosome')
plt.ylabel('Corrected coverage')
plt.savefig('./final_figures/method_schematic/cov_mcmc.png', bbox_inches='tight')

# ACDP
acdp_obj = pd.read_pickle('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_39675_10_0.7/Hapaseg_run_acdp__2023-02-14--08-59-34_gzuq0bq_qbvxcjy_ovu5fufsbrikq/jobs/0/workspace/acdp_model.pickle')
acdp_obj.cytoband_file = cytoband_file
acdp_obj.visualize_ACDP(use_cluster_stats=True, plot_hist=False, plot_real_cov=False, show_cdp=False)
plt.xlim([filt.start_g.min(), filt.start_g.max()])
plt.ylim([0, 800])
plt.xlabel('Chromosome')
fig = plt.gcf()
fig.set_size_inches(14,8)
plt.savefig('./final_figures/method_schematic/acdp.png', bbox_inches='tight')
