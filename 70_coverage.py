import itertools
import labellines
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc

from capy import mut, seq

# # Load MCMC trace over SNP DP cluster assignments
#
# For now, we are only looking at the high purity exome

clust = np.load("exome/6_C1D1_META.DP_clusts.auto_ref_correct.overdispersion92.no_phase_correct.npz")

# # Load coverage

Cov = pd.read_csv("exome/6_C1D1_META.cov", sep = "\t", names = ["chr", "start", "end", "covcorr", "covraw"])
Cov["chr"] = mut.convert_chr(Cov["chr"])
Cov = Cov.loc[Cov["chr"] != 0]
Cov["start_g"] = seq.chrpos2gpos(Cov["chr"], Cov["start"])
Cov["end_g"] = seq.chrpos2gpos(Cov["chr"], Cov["end"])

# # Load covariates {{{

# ### Target length

Cov["C_len"] = Cov["end"] - Cov["start"] + 1

# ## Replication timing

# +
# load track
F = pd.read_pickle("covars/GSE137764_H1.hg19_liftover.pickle")

# map targets to RT intervals
tidx = mut.map_mutations_to_targets(Cov.rename(columns = { "start" : "pos" }), F, inplace = False)
Cov.loc[tidx.index, "C_RT"] = F.iloc[tidx, 3:].mean(1).values

# z-transform
Cov["C_RT_z"] = (lambda x : (x - np.nanmean(x))/np.nanstd(x))(np.log(Cov["C_RT"] + 1e-20))
# -

# ### RT vs. coverage density

plt.figure(2, figsize = [19.2, 5.39]); plt.clf()
plt.scatter(Cov["start_g"], np.log(Cov["covcorr"]/Cov["C_len"]), s = 1, alpha = 0.1)
plt.scatter(Cov["start_g"], 5*Cov["C_RT"] + 4, alpha = 0.1, s = 1)
#plt.legend(["Coverage density", "Replication timing"], loc = "")

plt.figure()
plt.scatter(Cov["C_RT_z"], np.log(Cov["covcorr"]/Cov["C_len"]), s = 1, alpha = 0.05)
plt.xlim([-2.5, 2.5]);
plt.ylim([4, 7]);
plt.xlabel("Replication timing (z-score)")
plt.ylabel("Log coverage density");

# ## GC content

B = pd.read_pickle("covars/GC.pickle")
Cov = Cov.merge(B.rename(columns = { "gc" : "C_GC" }), left_on = ["chr", "start", "end"], right_on = ["chr", "start", "end"], how = "left")
Cov["C_GC_z"] = (lambda x : (x - np.nanmean(x))/np.nanstd(x))(np.log(Cov["C_GC"] + 1e-20))

# ### GC content vs. coverage density

plt.figure(11); plt.clf()
plt.scatter(Cov["C_GC_z"], np.log(Cov["covcorr"]/Cov["C_len"]), alpha = 0.01, s = 1)
plt.xlim([-0.5, 0.5]);
plt.ylim([4, 7]);
plt.xlabel("GC content (z-score)")
plt.ylabel("Log coverage density");

# ## GC content vs. RT

plt.figure(12); plt.clf()
plt.scatter(Cov["C_GC_z"], Cov["C_RT_z"], alpha = 0.01, s = 1)
plt.xlim([-0.75, 0.75])
plt.ylim([-5, 5])
plt.xlabel("GC content (z-score)")
plt.ylabel("Replication timing (z-score)");

# Covariates are correlated, but not to a problematic degree.

# }}}

# # Load SNPs

SNPs = pd.read_pickle("exome/6_C1D1_META.SNPs.pickle")
SNPs["chr"], SNPs["pos"] = seq.gpos2chrpos(SNPs["gpos"])

# #### Map to targets

SNPs["tidx"] = mut.map_mutations_to_targets(SNPs, Cov, inplace = False)
# TODO: handle NaN's better

# #### Generate unique clust assignments

clust_u, clust_uj = np.unique(clust["snps_to_clusters"], return_inverse = True)
clust_uj = clust_uj.reshape(clust["snps_to_clusters"].shape)

# #### Assign coverage intervals to clusters
# +
Cov_clust_probs = np.zeros([len(Cov), clust_u.max()])

for targ, snp_idx in SNPs.groupby("tidx").indices.items():
    targ_clust_hist = np.bincount(clust_uj[:, snp_idx].ravel(), minlength = clust_u.max())

    Cov_clust_probs[int(targ), :] = targ_clust_hist/targ_clust_hist.sum()
# -

# #### Subset to intervals containing SNPs

overlap_idx = Cov_clust_probs.sum(1) > 0
Cov_clust_probs_overlap = Cov_clust_probs[overlap_idx, :]

# #### Prune improbable assignments

Cov_clust_probs_overlap[Cov_clust_probs_overlap < 0.05] = 0
Cov_clust_probs_overlap /= Cov_clust_probs_overlap.sum(1)[:, None]
prune_idx = Cov_clust_probs_overlap.sum(0) > 0
Cov_clust_probs_overlap = Cov_clust_probs_overlap[:, prune_idx]

# #### Plot cluster assignment probabilities

plt.figure(1, figsize = [19.2, 5.39]); plt.clf()
plt.imshow(Cov_clust_probs_overlap[:, Cov_clust_probs_overlap.sum(0) > 0].T, aspect = "auto", interpolation = "none", cmap = "jet")
cb = plt.colorbar();
cb.set_label("Cluster assignment probability");
plt.xlabel("Target index");
plt.ylabel("DP cluster index");

# # Poisson regression

# ## Subset to only targets that overlap SNPs
# +
Cov_overlap = Cov.loc[overlap_idx, :]

r = np.c_[Cov_overlap["covcorr"]]
# -

# ## Make covariate matrix

# +
C = np.c_[np.log(Cov_overlap["C_len"]), Cov_overlap["C_RT_z"], Cov_overlap["C_GC_z"]]

# (experimenting with covariate subsets)
#C = np.c_[np.log(Cov_overlap["C_len"]), Cov_overlap["C_GC_z"]]
#C = np.c_[np.log(Cov_overlap["C_len"]), Cov_overlap["C_RT_z"]]
# -

# ## Cluster assignment vector ($\vec\pi_i$)

Pi = Cov_clust_probs_overlap.copy()

# #### Drop NaN's for now

naidx = np.isnan(C[:, 1])
r = r[~naidx]
C = C[~naidx]
Pi = Pi[~naidx]

# ## Define optimization functions

# +
# mu gradient
def gradmu(mu, beta, r, C, Pi):
    e = np.exp(C@beta + Pi@mu)
    return Pi.T@(r - e)

# mu Hessian
def hessmu(mu, beta, r, C, Pi): 
    e = np.exp(C@beta + Pi@mu)
    return -Pi.T@np.diag(e.ravel())@Pi

# beta gradient
def gradbeta(beta, mu, r, C, Pi):
    e = np.exp(C@beta + Pi@mu)
    return C.T@(r - e)

# beta Hessian
def hessbeta(beta, mu, r, C, Pi): 
    e = np.exp(C@beta + Pi@mu)
    return -C.T@np.diag(e.ravel())@C

# mu,beta Hessian
def hessmubeta(mu, beta, r, C, Pi):
    e = np.exp(C@beta + Pi@mu)
    return -C.T@np.diag(e.ravel())@Pi
# -

# ## Fit regression parameters

# ### Initialize parameters

mu = np.log(r.mean()*np.ones([Pi.shape[1], 1]))
beta = np.ones([C.shape[1], 1])

# ### Run Newton-Raphson iterations

for i in range(100):
    gmu = gradmu(mu, beta, r, C, Pi)
    gbeta = gradbeta(beta, mu, r, C, Pi)
    grad = np.r_[gmu, gbeta]

    hmu = hessmu(mu, beta, r, C, Pi)
    hbeta = hessbeta(beta, mu, r, C, Pi)
    hmubeta = hessmubeta(mu, beta, r, C, Pi)
    H = np.r_[np.c_[hmu, hmubeta.T], np.c_[hmubeta, hbeta]]

    delta = np.linalg.inv(H)@grad
    mu -= delta[0:len(mu)]
    beta -= delta[len(mu):]

    if np.linalg.norm(grad) < 1e-5:
        break

# # Plots

# #### Load color palette

colors = mpl.cm.get_cmap("tab10").colors

# #### Load chromosome boundary coordinates

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.auto_ref_correct.overdispersion92.no_phase_correct.pickle")
chrbdy = allelic_segs.dropna().loc[:, ["start", "end"]]
chr_ends = chrbdy.loc[chrbdy["start"] != 0, "end"].cumsum()

# ## Regressed coverage density
# +
plt.figure(3, figsize = [19.2,  5.39]); plt.clf()
_, axs = plt.subplots(3, 1, sharex = True, sharey = True, num = 3)
axs[0].scatter(Cov_overlap.loc[~naidx, "start_g"], np.exp(np.log(r) - C[:, [0]]@beta[[0]]), alpha = 1, s = 1)
axs[0].set_title("Raw coverage density")
for chrbdy in chr_ends[:-1]:
    axs[0].axvline(chrbdy, color = 'k')

axs[1].scatter(Cov_overlap.loc[~naidx, "start_g"], np.exp(np.log(r) - C@beta), alpha = 1, s = 1)
axs[1].set_title("Corrected coverage density")
axs[1].set_ylabel("Coverage density")
for chrbdy in chr_ends[:-1]:
    axs[1].axvline(chrbdy, color = 'k')

axs[2].scatter(Cov_overlap.loc[~naidx, "start_g"], np.exp(np.log(r) - C@beta), s = 0.1, color = 'k')

for i, p in enumerate(Pi.T):
    nzidx = p > 0
    x = Cov_overlap.loc[~naidx, "start_g"].loc[nzidx]
    axs[2].scatter(x, np.full(len(x), np.exp(mu[i])), alpha = p[nzidx]**4, s = 1, color = np.array(colors)[i % len(colors)])
axs[2].set_title("Inferred coverage density for each target")
axs[2].set_xlabel("Genomic position")

for chrbdy in chr_ends[:-1]:
    axs[2].axvline(chrbdy, color = 'k')

plt.xlim((0.0, 2879000000.0));
plt.ylim([0, 400]);
# -

# Segment colors correspond to their DP cluster. Looking at the allelic imbalance segementation, the red DP cluster is LoH; the brown cluster is balanced. There are some regions of genome doubling, which is why the brown cluster is sitting a little higher than the coverage densities of most targets it overlaps.
#
# This also means that the majority of the red LoH cluster is copy-neutral, owing to how close it is to the brown cluster.

# ## Allelic copy ratio

# #### Get average total min/maj counts of each DP cluster

# +
min_tots = np.zeros(clust_uj.max() + 1)
maj_tots = np.zeros(clust_uj.max() + 1)
for clusts, phases in zip(clust_uj, clust["snps_to_phases"]):
    # reset phases
    SNPs2 = SNPs.copy()
    SNPs2.iloc[phases, [0, 1]] = SNPs2.iloc[phases, [1, 0]]

    maj_tots += npg.aggregate(clusts, SNPs2["maj"], size = clust_uj.max() + 1)
    min_tots += npg.aggregate(clusts, SNPs2["min"], size = clust_uj.max() + 1)

min_tots /= clust_uj.shape[0]
maj_tots /= clust_uj.shape[0]
# -

# #### Remove DP clusters that had overall low assignment probabilities

f_prune = (min_tots/(min_tots + maj_tots))[np.flatnonzero(prune_idx)]

# ####  Plot

# To make targets clearer when plotting, show each target as \[start_i, start_{i+1}\]

Cov_overlap["next_g"] = np.r_[Cov_overlap.iloc[1:]["start_g"], 2880794554]

# +
plt.figure(6, figsize = [19.2, 5.39]); plt.clf()
for i, p in enumerate(Pi.T):
    nzidx = p > 0.3
    x = Cov_overlap.loc[~naidx, ["start_g", "next_g"]].loc[nzidx]
    for (_, st, en), clust_prob in zip(x.itertuples(), p[nzidx]):
        plt.plot(
            np.r_[st, en],
            np.exp(mu[i])*f_prune[i]*np.r_[1, 1],
            color = np.array(colors)[i % len(colors)],
            linewidth = 5,
            alpha = clust_prob**2,
            solid_capstyle = "butt"
        )
        plt.plot(
            np.r_[st, en],
            np.exp(mu[i])*(1 - f_prune[i])*np.r_[1, 1],
            color = np.array(colors)[i % len(colors)],
            linewidth = 5,
            alpha = clust_prob**2,
            solid_capstyle = "butt"
        )

for chrbdy in chr_ends[:-1]:
    plt.axvline(chrbdy, color = 'k')

plt.xlabel("Genomic position")
plt.ylabel("Coverage of major/minor alleles")

plt.xlim((0.0, 2879000000.0));
plt.ylim([0, 300]);
# -

# Looks like a comb is starting to come together nicely, despite the fact that the coverage segmentation does not yet account for balanced gains! We can clearly see that balanced segments (brown segments at n = 1) are a little too high, not properly accounting for a genome doubled region somewhere.

# ## Residuals

plt.figure(444); plt.clf()
plt.scatter(Cov_overlap.loc[~naidx, "start_g"], np.exp(np.log(r) - (C@beta + Pi@mu)), alpha = 1, s = 1)
plt.xlabel("Genomic position")
plt.ylabel("Residual coverage");

# Residuals are much lower than I would have expected.

# ## Predicted vs. residuals

plt.figure(4); plt.clf()
plt.scatter(Pi@mu + C@beta, np.log(r) - (Pi@mu + C@beta), alpha = 0.5, s = 1)
plt.xlabel("Predicted log coverage density")
plt.ylabel("Residual coverage");

# We observe no bias of predicted coverage versus residuals.

# ## Observed vs. predicted

plt.figure(5); plt.clf()
plt.scatter(np.log(r), Pi@mu + C@beta, alpha = 0.5, s = 1)
plt.xlabel("Observed coverage (log)")
plt.ylabel("Predicted coverage (log)");

# Regression model is surprisingly accurate.


# #### Scrap code below; interpolate coverage to targets that don't contain SNPs

bdy = np.unique(np.r_[0, np.flatnonzero(overlap_idx), len(Cov_clust_probs)])
bdy = np.c_[bdy[:-1], bdy[1:]]

Cov_clust_probs_interp = Cov_clust_probs.copy()

for st, en in bdy:
    if st == 0:
        Cov_clust_probs_interp[st:en, :] = Cov_clust_probs[en, :]
        continue
    if en == len(Cov_clust_probs):
        Cov_clust_probs_interp[st:en, :] = Cov_clust_probs[st, :]
        continue

    Cov_clust_probs_interp[(st + 1):en, :] = Cov_clust_probs[[st, en], :].mean(0)

# prune improbable assignments
Cov_clust_probs_interp = Cov_clust_probs_interp[:, prune_idx]

# define variables/covariates
r_int = np.c_[Cov["covcorr"]]

# covariates
C_int = np.c_[np.log(Cov["C_len"]), Cov["C_RT_z"], Cov["C_GC_z"]]

# cluster assignments
Pi_int = Cov_clust_probs_interp.copy()

# drop NaN's for now
naidx = np.isnan(C_int[:, 1])
r_int = r_int[~naidx]
C_int = C_int[~naidx]
Pi_int = Pi_int[~naidx]

plt.figure(33); plt.clf()
plt.scatter(Cov.loc[~naidx, "start_g"], Pi_int@mu + C_int[:, [1]]@beta[[1], :], alpha = 1, s = 1)

_, axs = plt.subplots(3, 1, sharex = True, sharey = True, num = 33)
axs[0].scatter(Cov.loc[~naidx, "start_g"], np.exp(np.log(r_int) - C_int[:, [0]]@beta[[0]]), alpha = 1, s = 1)
axs[0].set_title("Raw coverage density")

axs[1].scatter(Cov.loc[~naidx, "start_g"], np.exp(np.log(r_int) - C_int@beta), alpha = 1, s = 1)
axs[1].set_title("Corrected coverage density")

axs[2].scatter(Cov.loc[~naidx, "start_g"], np.exp(np.log(r_int) - C_int@beta), s = 0.1, color = 'k')

for i, p in enumerate(Pi_int.T):
    nzidx = p > 0
    x = Cov.loc[~naidx, "start_g"].loc[nzidx]
    axs[2].scatter(x, np.full(len(x), np.exp(mu[i])), alpha = p[nzidx]**4, s = 1, color = np.array(colors)[i % len(colors)])
axs[2].set_title("Inferred coverage for each target")

plt.xlim((0.0, 2879000000.0))
plt.ylim([0, 400])

#
# new coverage processing
#

### 
import hapaseg.run_coverage_MCMC
#from hapaseg.run_coverage_MCMC import CoverageMCMCRunner, aggregate_clusters, aggregate_burnin_files 

args = lambda:None

# ALCH
args.coverage_csv = "/mnt/nfs/workspace/ALCH_000b5e0e/gather_coverage__2022-04-26--11-56-17_g414gwy_tbhx1ki_2iix430p5rile/outputs/0/coverage/coverage_cat.bed" 
args.allelic_clusters_object = "/mnt/nfs/workspace/ALCH_000b5e0e/Hapaseg_allelic_DP__2022-04-20--08-28-23_rhbc0tq_eigugba_m5hwkvb1xdflo/outputs/0/cluster_and_phase_assignments/allelic_DP_SNP_clusts_and_phase_assignments.npz"
args.SNPs_pickle = "/mnt/nfs/workspace/ALCH_000b5e0e/Hapaseg_allelic_DP__2022-04-20--08-28-23_rhbc0tq_eigugba_m5hwkvb1xdflo/outputs/0/all_SNPs/all_SNPs.pickle"
# args.segmentations = # TODO: fill in!
args.repl_pickle = "gs://opriebe-tmp/GSE137764_H1.hg38.pickle"

# Richter's
args.coverage_csv = "/mnt/nfs/HapASeg_Richters/CH1011LN-CH1011GL/gather_coverage__2022-04-27--11-05-51_g414gwy_tbhx1ki_tknqgaiklbgdi/jobs/0/workspace/coverage_cat.bed"
args.allelic_clusters_object = "/mnt/j/proj/cnv/20201018_hapseg2/genome/CH1011LN-CH1011GL/allelic_DP_SNP_clusts_and_phase_assignments.npz"
args.SNPs_pickle = "/mnt/j/proj/cnv/20201018_hapseg2/genome/CH1011LN-CH1011GL/all_SNPs.pickle"
args.segmentations_pickle = "/mnt/j/proj/cnv/20201018_hapseg2/genome/CH1011LN-CH1011GL/segmentations.pickle"
args.repl_pickle = "gs://opriebe-tmp/GSE137764_H1.hg19_liftover.pickle"

# Richter's 2
args.coverage_csv = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/gather_coverage__2022-04-27--10-46-55_g414gwy_tbhx1ki_1um2ayzjbcu1a/jobs/0/workspace/coverage_cat.bed"
args.allelic_clusters_object = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_allelic_DP__2022-05-05--13-16-15_fg035ei_s2v0xea_tygpai5cxpelg/jobs/0/workspace/allelic_DP_SNP_clusts_and_phase_assignments.npz"
args.SNPs_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_allelic_DP__2022-05-05--13-16-15_fg035ei_s2v0xea_tygpai5cxpelg/jobs/0/workspace/all_SNPs.pickle"
args.segmentations_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_allelic_DP__2022-05-05--13-16-15_fg035ei_s2v0xea_tygpai5cxpelg/jobs/0/workspace/segmentations.pickle"
args.repl_pickle = "gs://opriebe-tmp/GSE137764_H1.hg19_liftover.pickle"
args.faire_pickle = "/mnt/j/proj/cnv/20201018_hapseg2/covars/FAIRE_GM12878.smooth5.hg19.pickle"

args.allelic_sample = None

# Richter's 2 (with symlinked inputs)
args.coverage_csv = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/coverage_cat.bed" 
args.allelic_clusters_object  = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/allelic_DP_SNP_clusts_and_phase_assignments.npz" 
args.SNPs_pickle  = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/all_SNPs.pickle"
args.segmentations_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/segmentations.pickle"
args.repl_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/GSE137764_H1.hg19_liftover.pickle"
args.faire_pickle  = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/FAIRE_GM12878.hg19.pickle"
args.ref_fasta = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/Homo_sapiens_assembly19.fasta"
args.bin_width = 2000

# Richter's 3 (with failing stats and fixed FAIRE + RT)
args.coverage_csv = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/gather_coverage__2022-07-27--23-28-37_g414gwy_tbhx1ki_iv55adsv4snga/jobs/0/workspace/coverage_cat.bed"
args.faire_pickle  = "/mnt/j/proj/cnv/20201018_hapseg2/covars/FAIRE/coverage.dedup.raw.10kb.pickle"
args.repl_pickle = "/mnt/j/proj/cnv/20201018_hapseg2/covars/RT.raw.hg19.pickle"

# platinum
args.coverage_csv = "/mnt/nfs/workspace/Hapaseg_prepare_coverage_mcmc__2022-08-01--14-47-27_k3t2mia_m4uwnti_erz1xgdjuiefy/jobs/0/inputs/wgs_sim_1_coverage_hapaseg_format.bed"
args.allelic_clusters_object = "/mnt/nfs/workspace/Hapaseg_prepare_coverage_mcmc__2022-08-01--14-47-27_k3t2mia_m4uwnti_erz1xgdjuiefy/jobs/0/inputs/allelic_DP_SNP_clusts_and_phase_assignments.npz"
args.SNPs_pickle = "/mnt/nfs/workspace/Hapaseg_prepare_coverage_mcmc__2022-08-01--14-47-27_k3t2mia_m4uwnti_erz1xgdjuiefy/jobs/0/inputs/all_SNPs.pickle"
args.segmentations_pickle = "/mnt/nfs/workspace/Hapaseg_prepare_coverage_mcmc__2022-08-01--14-47-27_k3t2mia_m4uwnti_erz1xgdjuiefy/jobs/0/inputs/segmentations.pickle"
args.repl_pickle = "/mnt/j/proj/cnv/20201018_hapseg2/covars/RT.raw.hg38.pickle" # "gs://opriebe-tmp/GSE137764_H1.hg38.pickle",
args.ref_fasta = "/mnt/j/db/hg38/ref/hg38.analysisSet.fa"
args.faire_pickle = None
args.bin_width = 2000

# VIP exome
args.SNPs_pickle="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/all_SNPs.pickle"
args.allelic_clusters_object="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/allelic_DP_SNP_clusts_and_phase_assignments.npz"
args.coverage_csv="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/coverage_cat.bed"
args.faire_pickle="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/coverage.dedup.raw.10kb.pickle"
args.ref_fasta="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/Homo_sapiens_assembly19.fasta"
args.repl_pickle="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/RT.raw.hg19.pickle"
args.segmentations_pickle="/mnt/nfs/mel_VIP/mel/Hapaseg_prepare_coverage_mcmc__2022-08-07--08-56-44_xwcw0ti_c3z12rq_xbrlktnvipu14/jobs/0/inputs/segmentations.pickle"
args.normal_coverage_csv = "/mnt/nfs/mel_VIP/mel/gather_coverage__2022-08-12--09-08-26_g414gwy_tbhx1ki_2zbvyxhtwuc52/outputs/0/coverage/coverage_cat.bed"
args.bin_width = 1
args.wgs = False

# run manually

cov_mcmc_runner = hapaseg.run_coverage_MCMC.CoverageMCMCRunner(
  args.coverage_csv,
  args.allelic_clusters_object,
  args.SNPs_pickle,
  args.segmentations_pickle,
  f_repl=args.repl_pickle,
  f_faire=args.faire_pickle,
  f_Ncov=args.normal_coverage_csv,
 # ref_fasta = "/mnt/j/db/hg38/ref/hg38.analysisSet.fa", # ALCH
  ref_fasta = args.ref_fasta, #"/mnt/j/db/hg19/ref/hs37d5.fa", # Richter's
  bin_width = args.bin_width,
  wgs = args.wgs if "wgs" in args.__dict__ else True
)
Pi, r, C, all_mu, global_beta, cov_df, adp_cluster = cov_mcmc_runner.prepare_single_cluster()

# run with wrapper
import itertools, sys
from hapaseg import __main__ as main
sys.argv = ["__", "--output_dir", "genome/CH1001LN-CH1001GL/cov_prep", "coverage_mcmc_preprocess"] + list(itertools.chain(*[[f"--{k}", f"{v}"] for k, v in args.__dict__.items()]))
main.main()

# run with wolF (if Docker is up-to-date)
from wolF import tasks
run = tasks.Hapaseg_prepare_coverage_mcmc(
  inputs = {
    "coverage_csv" : args.coverage_csv,
    "allelic_clusters_object" : args.allelic_clusters_object
    "SNPs_pickle" : args.SNPs_pickle,
    "segmentations_pickle" : args.segmentations_pickle,
    "repl_pickle" : args.repl_pickle,
    "faire_pickle" : args.faire_pickle,
    "ref_fasta" : "/mnt/j/db/hg38/ref/hg38.analysisSet.fa"
  }
)

# run platinum
pt = tasks.Hapaseg_prepare_coverage_mcmc(
  inputs = {
    "coverage_csv" : "gs://jh-xfer/HapASeg_platinum/coverage/wgs_sim_1_coverage_hapaseg_format.bed",
    "allelic_clusters_object" : "gs://jh-xfer/HapASeg_platinum/coverage/allelic_DP_SNP_clusts_and_phase_assignments.npz",
    "SNPs_pickle" : "gs://jh-xfer/HapASeg_platinum/coverage/all_SNPs.pickle",
    "segmentations_pickle" : "gs://jh-xfer/HapASeg_platinum/coverage/segmentations.pickle",
    "repl_pickle" : "/mnt/j/proj/cnv/20201018_hapseg2/covars/RT.raw.hg38.pickle", # "gs://opriebe-tmp/GSE137764_H1.hg38.pickle",
    "ref_fasta" : "/mnt/j/db/hg38/ref/hg38.analysisSet.fa"
  }
)
pt_results = pt.run()

# plots {{{
f, axs = plt.subplots(2, 1, num = 13, sharey=True,sharex=True, figsize = [16, 4])
axs[0].scatter(cov_df["start_g"], r, s= 0.1)
for i in range(cov_df["seg_idx"].max()):
    idx = cov_df["seg_idx"] == i
    axs[1].scatter(cov_df.loc[idx, "start_g"], np.exp(np.log(r[idx]) - C[idx]@global_beta), s=0.1)
    #axs[1].scatter(cov_df.loc[idx, "start_g"], np.exp(Pi[idx]@all_mu + C[idx]@global_beta), s=0.1)

plt.figure(14, figsize = [16, 4]); plt.clf()
f, axs = plt.subplots(2, 1, num = 14, sharex=True, figsize = [16, 4])
ph = self.allelic_clusters["snps_to_phases"]
for i in range(cov_df["allelic_cluster"].max() + 1):
    idx = cov_df["allelic_cluster"] == i
    x = axs[0].scatter(cov_df.loc[idx, "start_g"], np.exp(np.log(r[idx]) - C[idx, :]@global_beta)/2000, s=1, marker = ".", alpha = 0.2)
    idx = (self.SNPs["clust_choice"] == i) & self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    axs[1].scatter(self.SNPs.loc[idx, "pos_gp"], self.SNPs.loc[idx, "maj"]/self.SNPs.loc[idx, ["min", "maj"]].sum(1), color = x.get_facecolor(), s = 0.01)
    idx = (self.SNPs["clust_choice"] == i) & ~self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    axs[1].scatter(self.SNPs.loc[idx, "pos_gp"], self.SNPs.loc[idx, "min"]/self.SNPs.loc[idx, ["min", "maj"]].sum(1), color = x.get_facecolor(), s = 0.01)
plt.sca(axs[0])
axs[0].set_title("Coverage segmentation")
axs[0].set_ylabel("Coverage")
hapaseg.utils.plot_chrbdy("/mnt/j/db/hg19/ref/cytoBand.txt")
plt.sca(axs[1])
axs[1].set_title("Haplotypic imbalance segmentation")
axs[1].set_ylabel("Haplotypic imbalance")
hapaseg.utils.plot_chrbdy("/mnt/j/db/hg19/ref/cytoBand.txt")

axs[0].set_xlim([0, cov_df["start_g"].max() ])

plt.tight_layout()

plt.figure(15, figsize = [16, 4]); plt.clf()
for j, i in enumerate(cov_df["seg_idx"].unique()):
    idx = (self.SNPs["seg_idx"] == i) & self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    n = self.SNPs.loc[idx, "maj"].sum()
    d = self.SNPs.loc[idx, ["min", "maj"]].sum().sum()
    idx = (self.SNPs["seg_idx"] == i) & ~self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    n += self.SNPs.loc[idx, "min"].sum()
    d += self.SNPs.loc[idx, ["min", "maj"]].sum().sum()
    p = cov_df.loc[cov_df["allelic_cluster"] == j, "start_g"].iloc[[0, -1]]
    plt.plot(p, np.r_[1, 1]*np.exp(all_mu[j])*n/d, color = "r", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")
    plt.plot(p, np.r_[1, 1]*np.exp(all_mu[j])*(1 - n/d), color = "b", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")


f, axs = plt.subplots(2, 1, num = 14, sharex=True, figsize = [16, 4])
ph = self.allelic_clusters["snps_to_phases"]
for i in range(cov_df["allelic_cluster"].max() + 1):
    idx = cov_df["allelic_cluster"] == i
    x = axs[0].scatter(cov_df.loc[idx, "start_g"], np.exp(np.log(r[idx]) - C[idx, :]@global_beta)/2000, s=1, marker = ".", alpha = 0.2)
    idx = (self.SNPs["clust_choice"] == i) & self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    axs[1].scatter(self.SNPs.loc[idx, "pos_gp"], self.SNPs.loc[idx, "maj"]/self.SNPs.loc[idx, ["min", "maj"]].sum(1), color = x.get_facecolor(), s = 0.01)
    idx = (self.SNPs["clust_choice"] == i) & ~self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    axs[1].scatter(self.SNPs.loc[idx, "pos_gp"], self.SNPs.loc[idx, "min"]/self.SNPs.loc[idx, ["min", "maj"]].sum(1), color = x.get_facecolor(), s = 0.01)
plt.sca(axs[0])
axs[0].set_title("Coverage segmentation")
axs[0].set_ylabel("Coverage")
hapaseg.utils.plot_chrbdy("/mnt/j/db/hg19/ref/cytoBand.txt")
plt.sca(axs[1])
axs[1].set_title("Haplotypic imbalance segmentation")
axs[1].set_ylabel("Haplotypic imbalance")
hapaseg.utils.plot_chrbdy("/mnt/j/db/hg19/ref/cytoBand.txt")

axs[0].set_xlim([0, cov_df["start_g"].max() ])

plt.tight_layout()

##

plt.figure(15, figsize = [16, 4]); plt.clf()
f, axs = plt.subplots(2, 1, num = 15, sharex=True, figsize = [16, 4])
ph = self.allelic_clusters["snps_to_phases"]
for i in range(cov_df["seg_idx"].max() + 1):
    idx = cov_df["seg_idx"] == i
    axs[0].scatter(cov_df.loc[idx, "start_g"], np.exp(np.log(r[idx]) - C[idx, :]@global_beta)/2000, s=0.01, marker = ".", alpha = 1, color = ["orangered", "limegreen", "dodgerblue"][i % 3])
for i in range(cov_df["allelic_cluster"].max() + 1): 
    idx = (self.SNPs["clust_choice"] == i) & self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    x = axs[1].scatter(self.SNPs.loc[idx, "pos_gp"], self.SNPs.loc[idx, "maj"]/self.SNPs.loc[idx, ["min", "maj"]].sum(1), s = 0.01)
    idx = (self.SNPs["clust_choice"] == i) & ~self.allelic_clusters["snps_to_phases"][self.allelic_sample]
    axs[1].scatter(self.SNPs.loc[idx, "pos_gp"], self.SNPs.loc[idx, "min"]/self.SNPs.loc[idx, ["min", "maj"]].sum(1), color = x.get_facecolor(), s = 0.01)
plt.sca(axs[0])
axs[0].set_title("Coverage segmentation")
axs[0].set_ylabel("Coverage")
hapaseg.utils.plot_chrbdy("/mnt/j/db/hg19/ref/cytoBand.txt")
plt.sca(axs[1])
axs[1].set_title("Haplotypic imbalance segmentation")
axs[1].set_ylabel("Haplotypic imbalance")
hapaseg.utils.plot_chrbdy("/mnt/j/db/hg19/ref/cytoBand.txt")

axs[0].set_xlim([0, cov_df["start_g"].max() ])

plt.tight_layout()

# }}}

#
# coverage MCMC
#

## run with wrapper
import itertools, sys
from hapaseg import __main__ as main

args = lambda:None
args.allelic_seg_idx = 0
args.allelic_seg_indices = "/mnt/nfs/mel_VIP/mel/Hapaseg_coverage_mcmc_by_Aseg__2022-08-12--13-12-07_jqizvca_matpo2a_duoyxhs1m4jna/jobs/0/inputs/allelic_seg_groups.pickle"
args.preprocess_data = "/mnt/nfs/mel_VIP/mel/Hapaseg_coverage_mcmc_by_Aseg__2022-08-12--13-12-07_jqizvca_matpo2a_duoyxhs1m4jna/jobs/0/inputs/preprocess_data.npz"
args.num_draws = 5
args.bin_width = 1

sys.argv = ["__", "--output_dir", "genome/CH1001LN-CH1001GL/cov_mcmc", "coverage_mcmc_shard"] + list(itertools.chain(*[[f"--{k}", f"{v}"] for k, v in args.__dict__.items()]))
main.main()

## load in results




import hapaseg.NB_coverage_MCMC

args = lambda:None
args.allelic_seg_idx = 0 
args.allelic_seg_indices = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-05-05--14-27-07_wd1mmdy_xkoahjy_hftgpk1k4qnwk/jobs/0/inputs/allelic_seg_groups.pickle"
args.preprocess_data = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-05-05--14-27-07_wd1mmdy_xkoahjy_hftgpk1k4qnwk/jobs/0/inputs/preprocess_data.npz"
args.num_draws = 50
args.bin_width = 2000

args = lambda:None
args.allelic_seg_indices = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-05-16--15-39-20_kdo0bpa_x2kwcaa_hawbjnwiyqiby/jobs/0/inputs/allelic_seg_groups.pickle"
args.allelic_seg_idx = 0
args.bin_width = 2000
args.num_draws = 50
args.preprocess_data = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-05-16--15-39-20_kdo0bpa_x2kwcaa_hawbjnwiyqiby/jobs/0/inputs/preprocess_data.npz"

# version that outputs Hessian and has per-ADP segment mu
args = lambda:None
args.allelic_seg_indices = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-06-14--15-53-31_kdo0bpa_nixvisi_ahq5skg4a5mhu/jobs/0/inputs/allelic_seg_groups.pickle"
args.allelic_seg_idx = 0
args.bin_width = 2000
args.num_draws = 50
args.preprocess_data = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-06-14--15-53-31_kdo0bpa_nixvisi_ahq5skg4a5mhu/jobs/0/inputs/preprocess_data.npz"

# load preprocessed data
preprocess_data = np.load(args.preprocess_data)

# extract preprocessed data from this cluster
Pi = preprocess_data['Pi']
mu = preprocess_data["all_mu"]#[args.cluster_num]
beta = preprocess_data["global_beta"]
c_assignments = np.argmax(Pi, axis=1)
#cluster_mask = (c_assignments == args.cluster_num)
r = preprocess_data['r']#[cluster_mask]
C = preprocess_data['C']#[cluster_mask]

# load and (weakly) verify allelic segment indices
seg_g_idx = pd.read_pickle(args.allelic_seg_indices)
if len(np.hstack(seg_g_idx["indices"])) != C.shape[0]:
    raise ValueError("Size mismatch between allelic segment assignments and coverage bin data!")

# subset to a single allelic segment
if args.allelic_seg_idx > len(seg_g_idx) - 1:
    raise ValueError("Allelic segment index out of bounds!")

seg_indices = seg_g_idx.iloc[args.allelic_seg_idx]

mu = mu[seg_indices["allelic_cluster"]]
C = C[seg_indices["indices"], :]
r = r[seg_indices["indices"], :]

#pois_regr = PoissonRegression(r, C, np.ones_like(r))
#all_mu2, local_beta = pois_regr.fit()

# run cov MCMC
cov_mcmc = hapaseg.NB_coverage_MCMC.NB_MCMC_SingleCluster(args.num_draws, r, C, mu, beta, args.bin_width)
cov_mcmc.run()

for i in range(6):
    plt.figure(i + 1000); plt.clf()
    plt.title(covar_columns[i].replace("_", "\_"))
    for j in seg_g_idx["indices"].iloc[:10]:
        plt.scatter(C[j, i], np.log(r[j]) - Pi[j]@mu, s = 1)

i = 0
plt.figure(i + 10000); plt.clf()
_, axs = plt.subplots(6, 8, num = i + 10000, sharex = True, sharey = True)
axs = axs.ravel()
plt.title(covar_columns[i].replace("_", "\_"))
for k, j in enumerate(seg_g_idx["indices"]):
    axs[k].plot(C[j, i], np.log(r[j]) - Pi[j]@mu, marker = ',', antialiased=False, linewidth = 0)
plt.savefig("scatter.png", dpi = 400)

Pi = preprocess_data['Pi']
mu = preprocess_data["all_mu"]#[args.cluster_num]
beta = preprocess_data["global_beta"]
c_assignments = np.argmax(Pi, axis=1)
cluster_mask = (c_assignments == args.cluster_num)
r = preprocess_data['r']#[cluster_mask]
C = preprocess_data['C']#[cluster_mask]

i = 5
plt.figure(i + 20000); plt.clf()
_, axs = plt.subplots(6, 8, num = i + 20000, sharey = True)
axs = axs.ravel()
#plt.title(covar_columns[i].replace("_", "\_"))
for k, j in enumerate(seg_g_idx["indices"]):
    axs[k].plot(np.r_[0:len(j)], np.exp(np.log(r[j]) - Pi[j]@mu), marker = ',', antialiased=False, linewidth = 0)
    #axs[k].plot(np.r_[0:len(j)], 5000 + np.exp(np.log(r[j]) - (Pi[j]@mu + C[j]@beta)), marker = ',', antialiased=False, linewidth = 0)
    axs[k].plot(np.r_[0:len(j)], -600 + 400*C[j, i], marker = ',', antialiased=False, linewidth = 0)

plt.figure(i + 30000); plt.clf()
_, axs = plt.subplots(6, 8, num = i + 30000, sharey = True)
axs = axs.ravel()
#plt.title(covar_columns[i].replace("_", "\_"))
for k, j in enumerate(seg_g_idx["indices"]):
    axs[k].plot(np.r_[0:len(j)], r[j], marker = ',', antialiased=False, linewidth = 0)
    axs[k].axhline(np.exp(Pi[j[0]]@mu + np.log(2000)), color = "r")
    e_s = Pi[j[0]]@mu + C[j]@beta + np.log(2000)
    lik_dens = (-np.exp(e_s).sum() + r[j].T@e_s - ss.gammaln(r[j] + 1).sum())/len(j)
    axs[k].set_title(lik_dens[0, 0])
    #axs[k].plot(np.r_[0:len(j)], np.exp(np.log(r[j]) - Pi[j]@mu), marker = ',', antialiased=False, linewidth = 0)

plt.figure(i + 30200); plt.clf()
_, axs = plt.subplots(6, 8, num = i + 30200, sharey = True)
axs = axs.ravel()
#plt.title(covar_columns[i].replace("_", "\_"))
for k, j in enumerate(seg_g_idx["indices"]):
    e_s = Pi[j[0]]@mu + C[j]@beta + np.log(2000)
    axs[k].plot(np.r_[0:len(j)], np.exp(Pi[j[0]]@mu + np.log(2000)) + r[j] - np.exp(e_s), marker = ',', antialiased=False, linewidth = 0)
    axs[k].axhline(np.exp(Pi[j[0]]@mu + np.log(2000)), color = "r")




    lik_dens = (-np.exp(e_s).sum() + r[j].T@e_s - ss.gammaln(r[j] + 1).sum())/len(j)
    axs[k].set_title(lik_dens[0, 0])

# plots {{{

f, axs = plt.subplots(2, 1, num = 131, sharey=True,sharex=True, figsize = [16, 4])
axs[0].cla(); axs[1].cla()
axs[0].scatter(np.r_[0:len(r)], r/2000, s= 0.5)
axs[0].scatter(np.r_[0:len(r)], 10*C[:, 0], s= 0.5) # FAIRE
axs[0].scatter(np.r_[0:len(r)], 10*C[:, 1] - 10, s= 0.5) # RT
axs[0].scatter(np.r_[0:len(r)], 10*C[:, 2] - 10, s= 0.5) # RT
#axs[0].scatter(np.r_[0:len(r)], 10*cfz_smooth[cov_df.index[seg_indices["indices"]]], s= 0.5)
#axs[0].scatter(np.r_[0:len(r)], 10*C[:, 1], s= 0.5)
#axs[0].scatter(np.r_[0:len(r)], 10*C[:, 3], s= 0.5)
#axs[0].scatter(np.r_[0:len(r)], 10*C[:, 0], s= 0.5)
#axs[1].scatter(np.r_[0:len(r)], (r - np.exp(Pi@all_mu + C@global_beta + np.log(2000)) + np.exp(Pi@all_mu + np.log(2000)))/2000, s=0.5, c = cov_df["allelic_cluster"].map(lambda x : ["dodgerblue", "darkorange", "springgreen"][x % 3]))
axs[1].scatter(np.r_[0:len(r)], (r - np.exp(cov_mcmc.mu + C@cov_mcmc.beta + np.log(2000))), s=0.5)
#axs[1].scatter(np.r_[0:len(r)], (r - np.exp(C@local_beta)*np.exp(mu))/2000, s=0.5)
#axs[1].scatter(np.r_[0:len(r)], (r - np.exp(C@beta)*np.exp(mu))/2000, s=0.5)
bdy = np.c_[cov_mcmc.F_samples[-1][:-1:2], cov_mcmc.F_samples[-1][1::2]]
for st, en in bdy:
    axs[1].plot([st, en], np.r_[1, 1]*np.exp(cov_mcmc.mu_i_samples[-1][st])/2000)
#axs[1].scatter(np.r_[0:len(r)], np.exp(mu + cov_mcmc.mu_i_samples[-1][:, None] + C@beta)/2000, s=0.5)

# }}}

## prune uninformative segments
seg_mle_idx = np.r_[cov_mcmc.ll_samples].argmax()
bdy = np.c_[cov_mcmc.F_samples[seg_mle_idx][:-1:2], cov_mcmc.F_samples[seg_mle_idx][1::2]]

# local marginal likelihood based approach {{{
for st, en in bdy:
    # Laplace approximation of regression marginal likelihood
    #pois_regr = PoissonRegression(cov_mcmc.r[st:en], np.ones([en - st, 0]), np.ones([en - st, 1]), np.log(2000))
    pois_regr = PoissonRegression(cov_mcmc.r[st:en], cov_mcmc.C[st:en, [0, 2, 3, 4]], np.ones([en - st, 1]), np.log(2000))
    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    _, det = np.linalg.slogdet(pois_regr.hess())
    e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure
    regr_lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    regr_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + regr_lik

    # Poisson marginal likelihood from gamma distribution
    s = pois_regr.r.sum()
    beta = len(pois_regr.r)
    pois_marg_lik = (-1 - s)*np.log(beta) + ss.gammaln(s + 1) - ss.gammaln(pois_regr.r + 1).sum()

    print(regr_marg_lik - pois_marg_lik)

# doesn't work; model can always find a way to overfit to covariates, so Bayes factors always positive

# }}}

# global likelihood ratio based approach {{{

# instead, use MLE of global beta (really, we should integrate over the posterior distribution of global beta)

# experiment with using "glocal" beta -- beta re-fit specifically to the entire first allelic segment
pois_regr = PoissonRegression(cov_mcmc.r, np.zeros([len(cov_mcmc.r), 0]), np.ones([len(cov_mcmc.r), 1]), log_exposure = np.log(2000))
pois_regr = PoissonRegression(cov_mcmc.r, cov_mcmc.C, np.ones([len(cov_mcmc.r), 1]), log_exposure = np.log(2000))
pois_regr.fit()
mu_local = pois_regr.mu
beta_local = pois_regr.beta

hess_local = pois_regr.hess()
musig2 = -1/hess_local[0, 0]
betasiginv = -hess_local[1:, 1:]

plt.figure(1); plt.clf()
for st, en in bdy: 
    pois_regr = PoissonRegression(cov_mcmc.r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = mu_local + np.log(2000), log_offset = cov_mcmc.C[st:en]@beta_local)

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    _, det = np.linalg.slogdet(-pois_regr.hess())
    e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    regr_lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    regr_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + regr_lik

    exp_regr = e_s
    mu_regr = pois_regr.mu

    # Poisson marginal likelihood sans covariates
    pois_regr = PoissonRegression(cov_mcmc.r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = mu_local + np.log(2000))

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    _, det = np.linalg.slogdet(-pois_regr.hess())
    e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    regr_lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    pois_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + regr_lik

    exp_noregr = e_s

    #plt.scatter(np.r_[st:en], np.exp(mu_local + cov_mcmc.C[st:en]@cov_mcmc.beta + np.log(2000)), color = 'b', marker = '+')
    plt.scatter(np.r_[st:en], np.exp(mu_local + cov_mcmc.C[st:en]@beta_local + np.log(2000)), color = 'b', marker = '+')
    plt.scatter(np.r_[st:en], cov_mcmc.r[st:en], s = 1, color = 'k')
    plt.scatter(np.r_[st:en], cov_mcmc.r[st:en] - np.exp(exp_regr - mu_regr), s = 1)

    bf = (regr_marg_lik - pois_marg_lik)[0, 0]
    if bf < 0:
        plt.scatter(np.r_[st:en], cov_mcmc.r[st:en] - np.exp(exp_regr - mu_regr), s = 20, marker = "o", color = 'r', facecolor = "none")

#    # Poisson marginal likelihood from gamma distribution
#    s = pois_regr.r.sum()
#    beta = len(pois_regr.r)
#    pois_marg_lik = (-1 - s)*np.log(beta) + ss.gammaln(s + 1) - ss.gammaln(pois_regr.r + 1).sum()

    print("{st}-{en}: {lik}".format(st = st, en = en, lik = (regr_marg_lik - pois_marg_lik)[0,0]))

# }}}

# bilinear model {{{

# compute three marginal likelihoods

# 1. no covariates (nul)
# 2. global beta (lin)
# 3. global beta + seg-specific (bil)

plt.figure(2); plt.clf()
for st, en in bdy: 
    # no covariates
    pois_regr = PoissonRegression(cov_mcmc.r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000))

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    _, det = np.linalg.slogdet(-pois_regr.hess())
    e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    nul_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + lik

    mu_lin = pois_regr.mu

    # global beta (MLE)
    pois_regr = PoissonRegression(cov_mcmc.r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = mu_lin + np.log(2000), log_offset = cov_mcmc.C[st:en]@beta, intercept = False)

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    _, det = np.linalg.slogdet(-pois_regr.hess())
    e_s = pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    lin_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + lik

    exp_lin = e_s

    # global + local beta
    # fix intercept to be the same as global beta model (mu_lin); only use fine-grained covariates
    pois_regr = PoissonRegression(cov_mcmc.r[st:en], cov_mcmc.C[st:en, [0, 3, 4]], np.ones([en - st, 1]), log_exposure = mu_lin + np.log(2000), log_offset = cov_mcmc.C[st:en]@beta, intercept = False)
    #pois_regr = PoissonRegression(cov_mcmc.r[st:en], cov_mcmc.C[st:en], np.ones([en - st, 1]), log_exposure = mu_local + np.log(2000))

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta)
    _, det = np.linalg.slogdet(-pois_regr.hess())
    e_s = pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    bil_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + lik

    exp_bil = e_s


    #plt.scatter(np.r_[st:en], np.exp(mu_local + cov_mcmc.C[st:en]@beta + np.log(2000)), color = 'b', marker = '+')
    plt.scatter(np.r_[st:en], cov_mcmc.r[st:en], s = 1, color = 'k')
    plt.scatter(np.r_[st:en], cov_mcmc.r[st:en] - np.exp(exp_bil - mu_lin + mu_local) - 76000, marker = '+')
    plt.scatter(np.r_[st:en], cov_mcmc.r[st:en] - np.exp(exp_lin - mu_lin + mu_local), marker = 'x')

    print("{st}-{en}: {bil} {lin} {nul} ({mod})".format(st = st, en = en, bil = bil_marg_lik[0, 0], lin = lin_marg_lik[0, 0], nul = nul_marg_lik[0, 0], mod = np.r_[bil_marg_lik, lin_marg_lik, nul_marg_lik].argmax()))
    #print("{st}-{en}: {bil} {lin} {nul}".format(st = st, en = en, bil = (bil_marg_lik - lin_marg_lik)[0, 0], lin = (lin_marg_lik - nul_marg_lik)[0, 0], nul = (bil_marg_lik - nul_marg_lik)[0, 0]))

# }}}

# marginal likelihood with prior {{{

# using global regression as prior
mu = preprocess_data["all_mu"][seg_indices["allelic_cluster"]]
beta = preprocess_data["global_beta"]

hess = preprocess_data["pois_hess"]
musig2 = -1/hess[args.allelic_seg_idx, args.allelic_seg_idx]
betasiginv = -hess[-len(beta):, -len(beta):]
#betasiginv = np.c_[-hess[-6, -6]]

# using local regression on ADP segment as prior
pois_regr = PoissonRegression(r, C, np.ones([len(r), 1]), log_exposure = np.log(2000))
pois_regr.fit()
mu = pois_regr.mu
beta = pois_regr.beta

hess_local = pois_regr.hess()
musig2 = -1/hess_local[0, 0]
betasiginv = -hess_local[1:, 1:]

r[200:] = r[200:]*1.3

plt.figure(4); plt.clf()
_, axs = plt.subplots(2, 1, num = 4)
plt.figure(5); plt.clf()
for st, en in bdy: 
    # global beta (MLE)
    #pois_regr = PoissonRegression(r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000), log_offset = C[st:en]@beta)
    pois_regr = PoissonRegression(r[st:en], C[st:en], np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2, betamu = beta, betasiginv = betasiginv)
    #pois_regr = PoissonRegression(r[st:en], C[st:en, np.r_[:2, 3:6]], np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2)
    #pois_regr = PoissonRegression(r[st:en], C[st:en], np.ones([en - st, 1]), log_exposure = np.log(2000) + mu, intercept = False, betamu = beta, betasiginv = betasiginv)
    #pois_regr = PoissonRegression(r[st:en], C[st:en], np.ones([en - st, 1]), log_exposure = np.log(2000), betamu = beta, betasiginv = betasiginv)

    try:
        pois_regr.fit()
        #pois_regr.NR_f()
        #pois_regr.f = 1
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    #d = len(pois_regr.beta)
    _, det = np.linalg.slogdet(-pois_regr.hess())
    #e_s = pois_regr.mu + pois_regr.f*pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    e_s = pois_regr.mu + pois_regr.f*pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    #e_s = pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    lin_lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    lin_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + lin_lik + \
      s.norm.logpdf(pois_regr.mu, pois_regr.mumu, np.sqrt(pois_regr.musig2)) + \
      s.multivariate_normal.logpdf(pois_regr.beta.ravel(), pois_regr.betamu.ravel(), pois_regr.betasiginv)

    exp_lin = e_s
    mu_lin = pois_regr.mu

    # no covariates
    #pois_regr = PoissonRegression(r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2)
    #pois_regr = PoissonRegression(r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000) + mu, intercept = False)
    #pois_regr = PoissonRegression(r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2)
    pois_regr = PoissonRegression(r[st:en], np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000))
    #pois_regr = PoissonRegression(r[st:en], np.random.rand(en - st, 5), np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2)
    #pois_regr = PoissonRegression(r[st:en], np.ones([en - st, 5]), np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2)

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    d = len(pois_regr.beta) + 1
    _, det = np.linalg.slogdet(-pois_regr.hess())
    e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    #e_s = pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    nul_lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()
    nul_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + nul_lik + \
      s.norm.logpdf(pois_regr.mu, pois_regr.mumu, np.sqrt(pois_regr.musig2)) #+ \
      #s.multivariate_normal.logpdf(pois_regr.beta.ravel(), pois_regr.betamu.ravel(), pois_regr.betasiginv)
    #nul_marg_lik = d/2*np.log(2*np.pi) - 0.5*det + nul_lik

    plt.figure(5)
    plt.scatter(r[st:en], np.exp(exp_lin), marker = 'x')
    plt.scatter(r[st:en], np.exp(e_s), marker = '+')
    axs[0].scatter(np.r_[st:en], r[st:en], s = 1, color = 'k', marker = '.')
    axs[0].scatter(np.r_[st:en], np.exp(exp_lin), marker = 'x')
    axs[0].scatter(np.r_[st:en], np.exp(e_s), marker = '+')
    axs[1].scatter(np.r_[st:en], np.exp(exp_lin) - r[st:en], marker = 'x')
    #axs[1].scatter(np.r_[st:en], np.exp(e_s) - r[st:en], marker = '+')
    axs[1].axhline(color = 'k', linestyle = ":")
    if ((lin_marg_lik - nul_marg_lik)/(en - st) < -100)[0, 0]:
        axs[0].scatter(np.r_[st:en], r[st:en], s = 30, color = 'r', facecolor = "none")
        

    print("{st}-{en}: {dif}".format(st = st, en = en, dif = (lin_marg_lik[0, 0] - nul_marg_lik[0, 0])/(en - st)))
    #print("{st}-{en}: {lin} {nul}".format(st = st, en = en, lin = lin_marg_lik[0, 0], nul = nul_marg_lik[0, 0]))


# check Laplace approximation

sig = np.sqrt(-1/pois_regr.hess()[0, 0])
mu_rng = np.linspace(pois_regr.mu[0, 0] - 5*sig, pois_regr.mu[0, 0] + 5*sig, 100)
e_s = mu_rng + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
nul_lik = -np.exp(e_s).sum(0) + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()

plt.figure(10); plt.clf()
plt.plot(mu_rng, (nul_lik - nul_lik.max()).ravel())
lap = s.norm.logpdf(mu_rng, pois_regr.mu[0, 0], np.sqrt(-1/pois_regr.hess()[0, 0]))
plt.plot(mu_rng, lap - lap.max(), linestyle = "--")

# looks good in 1D!


# }}}

# autoregression {{{

pois_regr = PoissonRegression(r[10:], np.c_[C[10:, :], r[:-10]], np.ones([len(r) - 10, 1]), log_exposure = np.log(2000))
#pois_regr = PoissonRegression(r[1:], np.c_[C[1:, :], C[:-1]], np.ones([len(r) - 1, 1]), log_exposure = np.log(2000))
#pois_regr = PoissonRegression(r, C, np.ones([len(r), 1]), log_exposure = np.log(2000))
pois_regr.fit()

e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset

plt.figure(56); plt.clf()
_, axs = plt.subplots(2, 1, num = 56, sharex = True)
axs[0].scatter(np.r_[0:len(pois_regr.r)], np.exp(e_s), marker = '.', s = 1)
axs[0].scatter(np.r_[0:len(pois_regr.r)], pois_regr.r, marker = '.', s = 1)
axs[1].scatter(np.r_[0:len(pois_regr.r)], pois_regr.r - np.exp(e_s) , marker = '.', s = 1)

pois_regr = PoissonRegression(r[10:], np.c_[C[10:, :], r[:-10]], np.ones([len(r) - 10, 1]), log_exposure = np.log(2000))

# }}}

# likelihood approach (v2?) {{{

# using global regression as point estimate
mu = preprocess_data["all_mu"][seg_indices["allelic_cluster"]]
beta = preprocess_data["global_beta"]

hess_local = pois_regr.hess()
musig2 = -1/hess_local[0, 0]
betasiginv = -hess_local[1:, 1:]

plt.figure(40); plt.clf()
plt.figure(50); plt.clf()
for st, en in bdy: 
    # use global regression mu and beta
    r_seg = cov_mcmc.r[st:en]
    C_seg = cov_mcmc.C[st:en]

    e_s = mu + C_seg@beta + np.log(2000)
    lin_lik = -np.exp(e_s).sum() + r_seg.T@e_s - ss.gammaln(r_seg + 1).sum()

    exp_lin = e_s
    mu_lin = pois_regr.mu

    # no covariates; use local Poisson mu
    pois_regr = PoissonRegression(r_seg, np.zeros([en - st, 0]), np.ones([en - st, 1]), log_exposure = np.log(2000), mumu = mu, musig2 = musig2)

    try:
        pois_regr.fit()
    except np.linalg.LinAlgError:
        print(f"{st}-{en} is singular")
    e_s = pois_regr.mu + pois_regr.C@pois_regr.beta + pois_regr.log_exposure + pois_regr.log_offset
    nul_lik = -np.exp(e_s).sum() + pois_regr.r.T@e_s - ss.gammaln(pois_regr.r + 1).sum()

    plt.figure(50)
    #plt.scatter(np.r_[st:en], cov_mcmc.r[st:en], s = 1, color = 'k', marker = '.')
    plt.scatter(np.r_[st:en], -(cov_mcmc.r[st:en] - np.exp(exp_lin)), marker = '+')
    plt.scatter(np.r_[st:en], -(cov_mcmc.r[st:en] - np.exp(e_s)), marker = 'x')
    if (lin_lik - nul_lik < 0)[0, 0]:
        plt.scatter(np.r_[st:en], cov_mcmc.r[st:en], s = 30, color = 'r', facecolor = "none")
    plt.figure(40)
    plt.scatter(np.r_[st:en], cov_mcmc.r[st:en], s = 1, color = 'k', marker = '.')
    plt.scatter(np.r_[st:en], np.exp(exp_lin), marker = '+')
    plt.scatter(np.r_[st:en], np.exp(e_s), marker = 'x')
    if (lin_lik - nul_lik < 0)[0, 0]:
        plt.scatter(np.r_[st:en], cov_mcmc.r[st:en], s = 30, color = 'r', facecolor = "none")
        

    print("{st}-{en}: {dif}".format(st = st, en = en, dif = lin_lik[0, 0] - nul_lik[0, 0]))

# }}}

## aggregate cov MCMC
def nat_sort(lst): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(lst, key=alphanum_key)

allelic_seg_groups_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--15-35-16_040rmzi_pid3cty_0w3oyu5xxnfwe/jobs/0/workspace/allelic_seg_groups.pickle"

cov_df_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--15-35-16_040rmzi_pid3cty_0w3oyu5xxnfwe/jobs/0/workspace/cov_df.pickle"
cov_df = pd.read_pickle(cov_df_pickle)

R = pd.read_pickle("/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_coverage_mcmc__2022-05-17--16-43-11_kdo0bpa_1qbgw5i_hawbjnwiyqiby/WolfTaskResults.pickle")
all_lines = R["cov_segmentation_data"].values
read_files = []
for l in all_lines:
    to_add = l.rstrip('\n')
    if to_add != "nan":
        read_files.append(to_add)
adp_seg_files = nat_sort(read_files)

## plot
cov_df["mu_i"] = mu_is

SNPs = cov_mcmc_runner.SNPs.copy()
ph = cov_mcmc_runner.allelic_clusters["snps_to_phases"][cov_mcmc_runner.allelic_sample]

tidx = mut.map_mutations_to_targets(SNPs, cov_df, inplace=False)
tidx = pd.Series(cov_df.index[tidx], index = tidx.index)
SNPs["tidx2"] = -1
SNPs.loc[tidx.index, "tidx2"] = tidx.values
ph = ph[SNPs["tidx2"] != -1]
SNPs = SNPs.loc[SNPs["tidx2"] != -1]

SNPs = SNPs.merge(cov_df.loc[:, ["cov_seg_idx", "mu_i"]], left_on = "tidx2", right_index = True)

plt.figure(10); plt.clf()
for g, gidx in SNPs.groupby("cov_seg_idx").indices.items():
    i = SNPs.iloc[gidx[0]]["clust"]
    idx = (SNPs["clust"] == i) & ph
    n = SNPs.loc[idx, "maj"].sum()
    d = SNPs.loc[idx, ["min", "maj"]].sum().sum()
    idx = (SNPs["clust"] == i) & ~ph
    n += SNPs.loc[idx, "min"].sum()
    d += SNPs.loc[idx, ["min", "maj"]].sum().sum()
    p = SNPs.iloc[[gidx[0], gidx[-1]], :]["pos_gp"]
    mu = mu_refit[0] + SNPs.iloc[gidx[0]]["mu_i"]
    plt.plot(p, np.r_[1, 1]*np.exp(mu)*n/d, color = "r", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")
    plt.plot(p, np.r_[1, 1]*np.exp(mu)*(1 - n/d), color = "b", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")

## simple postprocessor {{{
from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB
from hapaseg.model_optimizers import PoissonRegression
from capy import seq

chrlens = seq.get_chrlens(ref = args.ref_fasta)

# 0. 1 allelic segment == 1 coverage segment, on the arm level

# 1. split each allelic segment into bins
allelic_clusters = cov_mcmc_runner.allelic_clusters
ph = allelic_clusters["snps_to_phases"][cov_mcmc_runner.allelic_sample]
SNPs = cov_mcmc_runner.SNPs

binsize = 4000
plt.figure(binsize, figsize = [16, 4]); plt.clf()
ax = plt.gca()

seg_file = []

for (seg, _), seg_idxs in cov_df.groupby(["seg_idx", "chr"]).indices.items():
    bdy = np.r_[seg_idxs[0]:seg_idxs[-1]:binsize, seg_idxs[-1]]
    bdy = np.c_[bdy[:-1], bdy[1:]]

    # get beta uncertainty for whole segment
    snp_idx = SNPs["seg_idx"] == seg

    den = SNPs.loc[snp_idx, ["min", "maj"]].sum().sum()
    num = SNPs.loc[snp_idx & ph, "maj"].sum() + SNPs.loc[snp_idx & ~ph, "min"].sum()

    # compute coverage segmentation on intervals
    for st, en in bdy:
        if en - st < 10:
            continue
        pois_regr = PoissonRegression(r[st:en], C[st:en], np.ones([en - st, 1]), np.log(2000) + C[st:en]@global_beta)
        mu, beta = pois_regr.fit()

        p = np.r_[cov_df["start_g"].iloc[st], cov_df["end_g"].iloc[en]]

        # get uncertainty around mu
        mu_post_sigma = np.linalg.inv(-pois_regr.hess())[0, 0]

#        plt.plot(p, np.r_[1, 1]*np.exp(mu[0])*num/den, color = "r", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")
#        plt.plot(p, np.r_[1, 1]*np.exp(mu[0])*(1 - num/den), color = "b", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")

        # empirically compute min/maj segment uncertainty
        maj_emp = s.beta.rvs(num + 1, den - num + 1, size = 1000)*np.exp(s.norm.rvs(mu, mu_post_sigma, size = 1000))
        min_emp = (1 - s.beta.rvs(num + 1, den - num + 1, size = 1000))*np.exp(s.norm.rvs(mu, mu_post_sigma, size = 1000))
        plt.plot(p, np.r_[1, 1]*maj_emp.mean(), color = "r", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")
        plt.plot(p, np.r_[1, 1]*min_emp.mean(), color = "b", alpha = 0.5, linewidth = 10, solid_capstyle = "butt")

        ## write entry to seg file
        seg_file.append([
          cov_df["chr"].iloc[st],    # Chromosome
          cov_df["start"].iloc[st],  # Start.bp
          cov_df["end"].iloc[en - 1],    # End.bp
          en - st,                   # n_probes
          cov_df["end_g"].iloc[en - 1] - cov_df["start_g"].iloc[st], # length
          en - st, # n_hets (standin
          num/den, # f
          (maj_emp + min_emp).mean(), # tau (will be rescaled)
          (maj_emp + min_emp).std(), # sigma.tau
          min_emp.mean(), # mu.minor
          min_emp.std(), # sigma.minor
          maj_emp.mean(), # mu.major
          maj_emp.std(), # sigma.major
          0, # SegLabelCNLOH (not used?)
        ])

S = pd.DataFrame(seg_file, columns = ["Chromosome", "Start.bp", "End.bp", "n_probes", "length", "n_hets", "f", "tau", "sigma.tau", "mu.minor", "sigma.minor", "mu.major", "sigma.major", "SegLabelCNLOH"]).astype({ "Chromosome" : int, "Start.bp" : int, "End.bp" : int })

# rescale such that mean = 2
sf = S["length"]@S["tau"]/S["length"].sum()/2

S.loc[:, ["tau", "sigma.tau", "mu.minor", "sigma.minor", "mu.major", "sigma.major"]] /= sf


#        ax.add_patch(mpl.patches.Rectangle(
#          (p[0], np.quantile(maj_emp, 0.05)),
#          p[1] - p[0],
#          np.maximum(0, np.diff(np.quantile(maj_emp, [0.05, 0.95]))[0]),
#          facecolor = 'r',
#          #edgecolor = 'k' if show_snps else None, linewidth = 0.5 if show_snps else None,
#          fill = True, alpha = 0.5
#        ))
#        ax.add_patch(mpl.patches.Rectangle(
#          (p[0], np.quantile(min_emp, 0.05)),
#          p[1] - p[0],
#          np.maximum(0, np.diff(np.quantile(min_emp, [0.05, 0.95]))[0]),
#          facecolor = 'b',
#          #edgecolor = 'k' if show_snps else None, linewidth = 0.5 if show_snps else None,
#          fill = True, alpha = 0.5
#        ))

# }}}
