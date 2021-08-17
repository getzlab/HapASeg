import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc

from capy import mut, seq

#
# load DP clusters
clust = np.load("exome/6_C1D1_META.DP_clusts.auto_ref_correct.overdispersion92.no_phase_correct.npz")

#
# load coverage
Cov = pd.read_csv("exome/6_C1D1_META.cov", sep = "\t", names = ["chr", "start", "end", "covcorr", "covraw"])
Cov["chr"] = mut.convert_chr(Cov["chr"])
Cov = Cov.loc[Cov["chr"] != 0]
Cov["start_g"] = seq.chrpos2gpos(Cov["chr"], Cov["start"])
Cov["end_g"] = seq.chrpos2gpos(Cov["chr"], Cov["end"])

#
# add covariates {{{
Cov["C_len"] = Cov["end"] - Cov["start"] + 1
#Cov["C_GC"] = 

#
# replication timing

# load track
F = pd.read_pickle("covars/GSE137764_H1.pickle")

# map targets to RT intervals
tidx = mut.map_mutations_to_targets(Cov.rename(columns = { "start" : "pos" }), F, inplace = False)
Cov.loc[tidx.index, "C_RT"] = F.iloc[tidx, 3:].mean(1).values

# z-transform
Cov["C_RT_z"] = (lambda x : (x - np.nanmean(x))/np.nanstd(x))(np.log(Cov["C_RT"] + 1e-20))

# show RT vs. scaled coverage
plt.figure(2); plt.clf()
plt.scatter(Cov["start_g"], np.log(Cov["covcorr"]/Cov["C_len"]), s = 1, alpha = 0.1)
plt.scatter(Cov["start_g"], 5*Cov["C_RT"] + 4, alpha = 0.1, s = 1)

#
# GC content

B = pd.read_pickle("covars/GC.pickle")
Cov = Cov.merge(B.rename(columns = { "gc" : "C_GC" }), left_on = ["chr", "start", "end"], right_on = ["chr", "start", "end"], how = "left")

plt.figure(11); plt.clf()
plt.scatter(Cov["C_GC"], Cov["covcorr"]/Cov["C_len"], alpha = 0.01, s = 1)

Cov["C_GC_z"] = (lambda x : (x - np.nanmean(x))/np.nanstd(x))(np.log(Cov["C_GC"] + 1e-20))

plt.figure(11); plt.clf()
plt.scatter(Cov["C_GC_z"], Cov["covcorr"]/Cov["C_len"], alpha = 0.01, s = 1)

# }}}

#
# load SNPs
SNPs = pd.read_pickle("exome/6_C1D1_META.SNPs.pickle")
SNPs["chr"], SNPs["pos"] = seq.gpos2chrpos(SNPs["gpos"])

# map to targets
SNPs["tidx"] = mut.map_mutations_to_targets(SNPs, Cov, inplace = False)

# TODO: handle NaN's better

# unique clust assignments
clust_u, clust_uj = np.unique(clust["snps_to_clusters"], return_inverse = True)
clust_uj = clust_uj.reshape(clust["snps_to_clusters"].shape)

# assign coverage intervals to clusters
Cov_clust_probs = np.zeros([len(Cov), clust_u.max()])

for targ, snp_idx in SNPs.groupby("tidx").indices.items():
    targ_clust_hist = np.bincount(clust_uj[:, snp_idx].ravel(), minlength = clust_u.max())

    Cov_clust_probs[int(targ), :] = targ_clust_hist/targ_clust_hist.sum()

# subset to intervals containing SNPs
overlap_idx = Cov_clust_probs.sum(1) > 0
Cov_clust_probs_overlap = Cov_clust_probs[overlap_idx, :]

# prune improbable assignments
Cov_clust_probs_overlap[Cov_clust_probs_overlap < 0.05] = 0
Cov_clust_probs_overlap /= Cov_clust_probs_overlap.sum(1)[:, None]
prune_idx = Cov_clust_probs_overlap.sum(0) > 0
Cov_clust_probs_overlap = Cov_clust_probs_overlap[:, prune_idx]

# plot cluster assignment probabilities
plt.figure(1); plt.clf()
plt.imshow(Cov_clust_probs_overlap[:, Cov_clust_probs_overlap.sum(0) > 0].T, aspect = "auto", interpolation = "none", cmap = "jet")

#
# Poisson regression
Cov_overlap = Cov.loc[overlap_idx, :]

r = np.c_[Cov_overlap["covcorr"]]

# covariates
C = np.c_[np.log(Cov_overlap["C_len"]), Cov_overlap["C_RT_z"], Cov_overlap["C_GC_z"]]

# cluster assignments
Pi = Cov_clust_probs_overlap.copy()

# drop NaN's for now
naidx = np.isnan(C[:, 1])
r = r[~naidx]
C = C[~naidx]
Pi = Pi[~naidx]

# define optimization functions

def gradmu(mu, beta, r, C, Pi):
    e = np.exp(C@beta + Pi@mu)
    return Pi.T@(r - e)

def hessmu(mu, beta, r, C, Pi): 
    e = np.exp(C@beta + Pi@mu)
    return -Pi.T@np.diag(e.ravel())@Pi

def gradbeta(beta, mu, r, C, Pi):
    e = np.exp(C@beta + Pi@mu)
    return C.T@(r - e)

def hessbeta(beta, mu, r, C, Pi): 
    e = np.exp(C@beta + Pi@mu)
    return -C.T@np.diag(e.ravel())@C

def hessmubeta(mu, beta, r, C, Pi):
    e = np.exp(C@beta + Pi@mu)
    return -C.T@np.diag(e.ravel())@Pi

# initialize parameters
mu = np.log(r.mean()*np.ones([Pi.shape[1], 1]))
beta = np.ones([C.shape[1], 1])

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

#
# plot
plt.figure(3); plt.clf()
# regressed coverage density
plt.scatter(Cov_overlap.loc[~naidx, "start_g"], Pi@mu + C@beta - C[:, [0]], alpha = 1, s = 1)
# original coverage density
plt.scatter(Cov_overlap.loc[~naidx, "start_g"], np.log(r) - C[:, [0]], alpha = 1, s = 1)

# regressed coverage
plt.scatter(Cov_overlap.loc[~naidx, "start_g"], Pi@mu + C@beta, alpha = 1, s = 1)

# residuals
plt.scatter(Cov_overlap.loc[~naidx, "start_g"], np.log(r) - (C@beta + Pi@mu), alpha = 1, s = 1)

# predicted vs. residuals
plt.figure(4); plt.clf()
plt.scatter(Pi@mu + C@beta, np.log(r - np.exp(Pi@mu + C@beta)), alpha = 0.5, s = 1)

# observed vs. predicted
plt.figure(5); plt.clf()
plt.scatter(np.log(r), Pi@mu + C@beta, alpha = 0.5, s = 1)

#
# interpolate

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
C_int = np.c_[np.log(Cov["C_len"]), Cov["C_RT_z"]]

# cluster assignments
Pi_int = Cov_clust_probs_interp.copy()

# drop NaN's for now
naidx = np.isnan(C_int[:, 1])
r_int = r_int[~naidx]
C_int = C_int[~naidx]
Pi_int = Pi_int[~naidx]

plt.figure(33); plt.clf()
# regressed coverage density
plt.scatter(Cov.loc[~naidx, "start_g"], Pi_int@mu + C_int[:, [1]]@beta[[1], :], alpha = 1, s = 1)
