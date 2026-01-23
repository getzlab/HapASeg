import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import copy
from capy import seq

R = pd.read_pickle("FF_benchmarking_profile_41611_1_0.7.concat_arms.pickle")

self = lambda : None

# load SNPs
self.SNPs = []
clust_offset = 0
for _, H in R.iterrows():
    S = copy.deepcopy(H["results"].P)
    S["A_alt"] = 0
    S.loc[S["aidx"], "A_alt"] = S.loc[S["aidx"], "ALT_COUNT"]
    S["A_ref"] = 0
    S.loc[S["aidx"], "A_ref"] = S.loc[S["aidx"], "REF_COUNT"]
    S["B_alt"] = 0
    S.loc[~S["aidx"], "B_alt"] = S.loc[~S["aidx"], "ALT_COUNT"]
    S["B_ref"] = 0
    S.loc[~S["aidx"], "B_ref"] = S.loc[~S["aidx"], "REF_COUNT"]

    S = S.rename(columns = { "MIN_COUNT" : "min", "MAJ_COUNT" : "maj" })
    S = S.loc[:, ["chr", "pos", "min", "maj", "A_alt", "A_ref", "B_alt", "B_ref", "aidx"]]

    # set initial cluster assignments based on segmentation
    S["clust"] = -1
    bpl = np.array(H["results"].breakpoints_MLE); bpl = np.c_[bpl[0:-1], bpl[1:]]
    for i, (st, en) in enumerate(bpl):
        S.iloc[st:en, S.columns.get_loc("clust")] = i + clust_offset
    clust_offset += i + 1

    # bug in segmentation omits final SNP?
    #S = S.iloc[:-1]
    #assert (S["clust"] != -1).all()

    self.SNPs.append(S)

self.SNPs = pd.concat(self.SNPs, ignore_index = True)

# convert chr-relative positions to absolute genomic coordinates
self.SNPs["pos_gp"] = seq.chrpos2gpos(self.SNPs["chr"], self.SNPs["pos"], ref = "/mnt/j/db/hg38/ref/hg38.analysisSet.fa")

from hapaseg import utils

# plot all SNPs to find some segments for good visualization

plt.figure(1); plt.clf()
minct = self.SNPs["min"]
cov = self.SNPs[["min", "maj"]].sum(1)
plt.scatter(self.SNPs["pos_gp"], minct/cov, marker = ".", s = 1, c = self.SNPs["aidx"], cmap = "bwr", alpha = np.minimum(1, cov/cov.median()))
segs = self.SNPs.groupby("clust").apply(lambda x : pd.Series({ "f" : x["min"].sum(0)/x[["min", "maj"]].sum(1).sum(0), "st" : x.iloc[0]["pos_gp"], "en" : x.iloc[-1]["pos_gp"]}))
for _, f, st, en in segs.iloc[:100].itertuples():
    plt.plot(np.r_[st, en], np.r_[1, 1]*f, color = "k")
utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")

# xlim:

SNPs_r = self.SNPs.loc[(668935395 < self.SNPs["pos_gp"]) & (self.SNPs["pos_gp"] < 711245229) | \
(1809009743 < self.SNPs["pos_gp"]) & (self.SNPs["pos_gp"] < 2027201276)]

# zoom in

rng_1 = (668935395 < self.SNPs["pos_gp"]) & (self.SNPs["pos_gp"] < 711245229)
SNPs_1 = self.SNPs.loc[rng_1]
SNPs_1["pos_gp"] -= SNPs_1["pos_gp"].min()
rng_2 = (1809009743 < self.SNPs["pos_gp"]) & (self.SNPs["pos_gp"] < 2027201276)
SNPs_2 = self.SNPs.loc[rng_2]
SNPs_2["pos_gp"] -= SNPs_2["pos_gp"].min() - SNPs_1["pos_gp"].max()

SNPs_rng = pd.concat([SNPs_1, SNPs_2])

plt.figure(2, figsize = [14, 12]); plt.clf()
_, axs = plt.subplots(2, 1, num = 2, sharex = True, sharey = True)
minct = SNPs_rng["min"]
cov = SNPs_rng[["min", "maj"]].sum(1)
axs[0].scatter(SNPs_rng["pos_gp"], minct/cov, marker = ".", s = 1, c = SNPs_rng["aidx"], cmap = "bwr", alpha = np.minimum(1, cov/cov.median()), zorder = 3)
segs = SNPs_rng.groupby("clust").apply(lambda x : pd.Series({ "f" : x["min"].sum(0)/x[["min", "maj"]].sum(1).sum(0), "st" : x.iloc[0]["pos_gp"], "en" : x.iloc[-1]["pos_gp"]}))
for _, f, st, en in segs.itertuples():
    axs[0].add_patch(plt.matplotlib.patches.Rectangle(
      (st, f - 0.01),
      np.maximum(en - st, 500000),
      0.02,
      facecolor = "lime",
      edgecolor = 'k', linewidth = 0.5,
      fill = True, alpha = 1, zorder = 1000
    ))
#plt.scatter((segs["en"] + segs["st"])/2, segs["f"], color = "cyan", marker = "s", s = 4)
axs[0].axhline(0.5, color = "k", linestyle = ":", zorder = 0)
axs[0].set_xlim([0, SNPs_rng["pos_gp"].max()])
axs[0].set_ylim([-0.05, 1.05])
axs[0].set_xticks([])
axs[0].set_yticks(np.r_[0:1.25:0.25])
axs[0].set_ylabel("Haplotypic imbalance")

# TODO: dress up the axis to emulate chrbdy
# utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")

# run DP clustering to show final result
# overlay ghost of original segmentation?

from hapaseg import allelic_DP

SNPs_rng["flipped"] = False

DPrunner = allelic_DP.DPinstance(
  S = SNPs_rng.reset_index(drop = True).copy(),
  dp_count_scale_factor = SNPs_rng["clust"].value_counts().mean(),
  betahyp = None
)

self.snps_to_clusters, self.snps_to_phases, self.likelihoods = DPrunner.run(n_samps = 100)

# excised from visualize function

colors = DPrunner.get_colors()
s2cu, s2cu_j = DPrunner.get_unique_clust_idxs()
cu = np.searchsorted(s2cu, DPrunner.S["clust"])

# flip phasing orientation to maximum likelihood
DPrunner_ph = copy.deepcopy(DPrunner)
DPrunner_ph.S["flipped"] = s2ph

mlidx = np.r_[DPrunner.likelihood_trace].argmax()
seg2c, s2ph = DPrunner.segment_trace[mlidx], DPrunner.phase_orientations[mlidx]

#uidx = ph_prob == 0
#ax.scatter(
#  DPrunner.S.loc[uidx, "pos_gp"],
#  DPrunner.S.loc[uidx, "min"]/DPrunner.S.loc[uidx, ["min", "maj"]].sum(1),
#  color = colors[cu[uidx] % len(colors)], marker = '.', alpha = 1, s = 1
#)
#uidx = ph_prob == 1
#ax.scatter(
#  DPrunner.S.loc[uidx, "pos_gp"],
#  DPrunner.S.loc[uidx, "maj"]/DPrunner.S.loc[uidx, ["min", "maj"]].sum(1),
#  color = colors[cu[uidx] % len(colors)], marker = '.', alpha = 1, s = 1
#)

plt_idx = DPrunner_ph.S["flipped"]
axs[1].scatter(
  DPrunner_ph.S.loc[plt_idx, "pos_gp"],
  DPrunner_ph.S.loc[plt_idx, "maj"]/DPrunner_ph.S.loc[plt_idx, ["min", "maj"]].sum(1),
  color = colors[cu[plt_idx] % len(colors)], marker = '.', s = 1, zorder = 3
)
plt_idx = ~DPrunner_ph.S["flipped"]
axs[1].scatter(
  DPrunner_ph.S.loc[plt_idx, "pos_gp"],
  DPrunner_ph.S.loc[plt_idx, "min"]/DPrunner_ph.S.loc[plt_idx, ["min", "maj"]].sum(1),
  color = colors[cu[plt_idx] % len(colors)], marker = '.', s = 1, zorder = 3
)

# overlay segments
seg_cu = np.searchsorted(s2cu, np.r_[list(seg2c.values())])
seg_bdy = np.r_[list(seg2c.keys()), len(DPrunner_ph.S)]
seg_bdy = np.c_[seg_bdy[:-1], seg_bdy[1:]]

# overlay original segments
for _, f, st, en in segs.itertuples():
    axs[1].add_patch(plt.matplotlib.patches.Rectangle(
      (st, f - 0.01),
      np.maximum(en - st, 500000),
      0.02,
      edgecolor = 'k', linewidth = 0.5, linestyle = ":",
      fill = False, alpha = 1, zorder = 1000
    ))

for i, (st, en) in enumerate(seg_bdy):
    mn = DPrunner_ph._Ssum_ph(np.r_[st:en], min = True)
    mx = DPrunner_ph._Ssum_ph(np.r_[st:en], min = False)
    f = mn/(mn + mx)
    axs[1].plot([DPrunner_ph.S.iloc[st]["pos_gp"], DPrunner_ph.S.iloc[en - 1]["pos_gp"]], [f, f], color = "k", linewidth = 1, zorder = 1001)

axs[1].axhline(0.5, color = "k", linestyle = ":", zorder = 0)
axs[1].set_ylabel("Haplotypic imbalance (refined)")
