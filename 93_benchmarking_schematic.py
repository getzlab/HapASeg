import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hapaseg.utils import plot_chrbdy
from capy import mut, seq, plots
from cnv_suite.visualize.plot_cnv_profile import *

coverage_path = '/mnt/nfs/workspace/NA12878_raw_data_gen_hapsaeg_realign/gather_coverage__2022-11-10--19-15-11_tdlrsdi_tbhx1ki_5akwueppznwx2/jobs/0/workspace/NA12878_platinum_realigned_covcollect.bed'
ref_fasta = '/home/opriebe/data/ref/hg38/GRCh38.d1.vd1.fa'
cytoband_file = '/home/opriebe/data/ref/hg38/cytoBand.txt'

def load_coverage(coverage_path, ref_fasta):
    Cov = pd.read_csv(coverage_path, sep="\t", names=["chr", "start", "end", "covcorr", "mean_frag_len", "std_frag_len", "num_frags", "tot_reads", "fail_reads"], low_memory=False)
    Cov.loc[Cov['chr'] == 'chrM', 'chr'] = 'chrMT'
    Cov["chr"] = mut.convert_chr(Cov["chr"])
    Cov = Cov.loc[Cov["chr"] != 0]
    Cov=Cov.reset_index(drop=True)
    Cov["start_g"] = seq.chrpos2gpos(Cov["chr"], Cov["start"], ref = ref_fasta)
    Cov["end_g"] = seq.chrpos2gpos(Cov["chr"], Cov["end"], ref = ref_fasta)
    Cov = Cov.loc[(Cov['tot_reads'] > 0) & (Cov['num_frags'] > 0)]
    with pd.option_context('mode.chained_assignment', None): # suppress erroneous SettingWithCopyWarning
        # note that unlike hapaseg processing here, we skip rounding
        Cov["fragcorr"] = Cov["covcorr"]/Cov["mean_frag_len"].mean()
    return Cov.reset_index(drop = True)

plt.rcParams.update({'font.size': 14})

Cov = load_coverage(coverage_path, ref_fasta)
fig = plt.figure(figsize=(14,8))
plt.ylim([300,700])
#plt.scatter(Cov.start_g.values, Cov.fragcorr.values, alpha=0.03, marker=',', s=1)
plots.pixplot(Cov.start_g.values, Cov.fragcorr.values, alpha=0.2, color='tab:blue')
plt.ylabel('Coverage')
plt.xlabel('Chromosome')
plot_chrbdy(cytoband_file)
plt.savefig('./final_figures/figure_3_benchmarking_schematic/FF_normal_raw_fragcorr.png', bbox_inches='tight')


# tumor coverage
tumor_df = load_coverage('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_42368_14_0.7/Generate_HapASeg_Sim_Data__2023-02-06--22-38-13_o2xztwa_qbvxcjy_zmctaom5zn2ei/jobs/0/workspace/FF_benchmarking_profile_42368_14_0.7_hapaseg_coverage.bed', ref_fasta)
fig = plt.figure(7, figsize=(14,8))
plots.pixplot(tumor_df.start_g.values, tumor_df.fragcorr.values, alpha=0.15, color='tab:blue')
plt.ylim([50, 1500])
plot_chrbdy(cytoband_file)
plt.ylabel('Coverage')
plt.xlabel('Chromosome')
plt.savefig('./final_figures/figure_3_benchmarking_schematic/FF_tumor_fragcorr.png', bbox_inches='tight')

# tumor VAF
acounts_df = pd.read_csv('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_42368_14_0.7/Generate_HapASeg_Sim_Data__2023-02-06--22-38-13_o2xztwa_qbvxcjy_zmctaom5zn2ei/jobs/0/workspace/FF_benchmarking_profile_42368_14_0.7_hapaseg_hets.bed', sep='\t')
acounts_df['aimb'] = acounts_df['REF_COUNT'] / (acounts_df['REF_COUNT'] + acounts_df['ALT_COUNT'])
acounts_df['start_g'] = seq.chrpos2gpos(acounts_df["CONTIG"], acounts_df["POSITION"], ref = ref_fasta)
fig = plt.figure(figsize=(14,8))
plots.pixplot(acounts_df.start_g.values, acounts_df.aimb.values, alpha=0.15, color='tab:blue')
plot_chrbdy(cytoband_file)
plt.ylabel('VAF')
plt.xlabel('Chromosome')
plt.savefig('./final_figures/figure_3_benchmarking_schematic/FF_tumor_vaf.png', bbox_inches='tight')

# tumor allelic coverage
## load coverage
cov_df = pd.read_pickle('/mnt/nfs/benchmarking_dir/FF_benchmarking_profile_42368_14_0.7/Hapaseg_prepare_coverage_mcmc__2023-02-06--23-31-00_rly1scy_qbvxcjy_s4fmwj1ocikn0/jobs/0/workspace/cov_df.pickle')
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
plt.savefig('./final_figures/figure_3_benchmarking_schematic/FF_tumor_allelic_coverage.png', bbox_inches='tight')

fig = plt.figure(figsize=(14,8))
ax = plt.gca()
sim_profile = pd.read_pickle('/home/opriebe/data/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_42368_14.pickle')
plot_acr_static(sim_profile.cnv_profile_df, ax = ax, csize=sim_profile.csize, min_seg_lw=5, y_upper_lim = 8)
plt.savefig('./final_figures/figure_3_benchmarking_schematic/sim_tumor_karyotype_42368.pdf')
plt.savefig('./final_figures/figure_3_benchmarking_schematic/sim_tumor_karyotype_42368.png', bbox_inches='tight')
