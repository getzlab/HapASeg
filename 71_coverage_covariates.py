import liftover
import numpy as np
import pandas as pd
import pyfaidx
import pyBigWig
import tqdm
from capy import mut, seq

#
# replication timing {{{

F = pd.read_csv("/mnt/j/proj/cnv/20201018_hapseg2/covars/GSE137764_H1_GaussiansGSE137764_mooth_scaled_autosome.mat", sep = "\t", header = None).T.rename(columns = { 0 : "chr", 1 : "start", 2 : "end" })
F.iloc[:, 3:] = F.loc[:, 3:].astype(float)
F.loc[:, ["start", "end"]] = F.loc[:, ["start", "end"]].astype(int)
F["chr"] = mut.convert_chr(F["chr"])
F.to_pickle("covars/GSE137764_H1.hg38.pickle")

# liftover to hg19 {{{
F["chr_start_lift"] = 0
F["chr_end_lift"] = 0
F["start_lift"] = 0
F["end_lift"] = 0
F["start_strand_lift"] = 'x'
F["end_strand_lift"] = 'x'
converter = liftover.ChainFile("/mnt/j/db/hg38/liftover/hg38ToHg19.over.chain", "hg38", "hg19")
for x in tqdm.tqdm(F.itertuples()):
    conv = converter[x.chr][x.start]
    F.loc[x.Index, ["chr_start_lift", "start_lift", "start_strand_lift"]] = (-1, -1, '?') if len(conv) == 0 else conv[0]
    conv = converter[x.chr][x.end]
    F.loc[x.Index, ["chr_end_lift", "end_lift", "end_strand_lift"]] = (-1, -1, '?') if len(conv) == 0 else conv[0]

F.to_pickle("covars/GSE137764_H1.hg19_raw_liftover.pickle")

# fix weird intervals

# flip reverse strands
idx = ((F["start_strand_lift"] == "-") & (F["end_strand_lift"] == "-")).values
st_col = F.columns.get_loc("start_lift")
en_col = F.columns.get_loc("end_lift")
F.iloc[idx, [st_col, en_col]] = F.iloc[idx, [en_col, st_col]]
F.loc[idx, ["start_strand_lift", "end_strand_lift"]] = "+"

# get length distribution
plt.figure(1); plt.clf()
plt.hist(F["end_lift"] - F["start_lift"] + 1, np.linspace(0, 100000, 100))
plt.yscale("log")

# for now, let's not bother with any funky intervals
good_idx = (F["chr_start_lift"] == F["chr_end_lift"]) & (F["start_lift"] < F["end_lift"]) & (F["end_lift"] - F["start_lift"] < 60000)

F_lift = F.loc[good_idx]
F_lift = F_lift.drop(columns = ["chr", "start", "end"]).rename(columns = { "chr_start_lift" : "chr", "start_lift" : "start", "end_lift" : "end" }).iloc[:, np.r_[16, 18, 19, 0:16]]

# remove non-primary chromosomes
F_lift["chr"] = mut.convert_chr(F_lift["chr"])
F_lift = F_lift.loc[F_lift["chr"].apply(lambda x : type(x) == int)]

F_lift = F_lift.sort_values(["chr", "start", "end"])

F_lift.to_pickle("covars/GSE137764_H1.hg19_liftover.pickle")

# check for weird intervals
bad_idx = (F["chr_start_lift"] != F["chr"]) | \
(F["chr_end_lift"] != F["chr"]) | \
(F["start_strand_lift"].notin(["+", "?"])) | \
(F["start_lift"] > F["end_lift"])

# }}}

# }}}

#
# GC content {{{

## precompute GC content {{{
# note: this is obsolete; GC content is now computed on the fly

B = pd.read_csv("/mnt/j/proj/cnv/20210326_coverage_collector/targets.bed", sep = "\t", header = None, names = ["chr", "start", "end"])
B["chr"] = mut.convert_chr(B["chr"])
B["gc"] = np.nan
B = B.loc[B["chr"].apply(type) == int]
B = B.loc[B["chr"] > 0]
F = pyfaidx.Fasta("/mnt/j/db/hg19/ref/hs37d5.fa")

for (i, chrm, start, end, _) in B.itertuples():
    B.iat[i, -1] = F[chrm - 1][(start - 1):end].gc

B.to_pickle("covars/GC.pickle")

# }}}

# Terry Speed GC content estimator {{{

import hapaseg.run_coverage_MCMC

# load coverage

args = lambda : None
args.coverage_csv = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/coverage_cat.bed" 
args.allelic_clusters_object  = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/allelic_DP_SNP_clusts_and_phase_assignments.npz" 
args.SNPs_pickle  = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/all_SNPs.pickle"
args.segmentations_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/segmentations.pickle"
args.repl_pickle = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/GSE137764_H1.hg19_liftover.pickle"
args.faire_pickle  = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/FAIRE_GM12878.hg19.pickle"
args.ref_fasta = "/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--12-14-09_040rmzi_1kaanny_0w3oyu5xxnfwe/jobs/0/inputs/Homo_sapiens_assembly19.fasta"
args.bin_width = 2000

cov_mcmc_runner = hapaseg.run_coverage_MCMC.CoverageMCMCRunner(
  args.coverage_csv,
  args.allelic_clusters_object,
  args.SNPs_pickle,
  args.segmentations_pickle,
  f_repl=args.repl_pickle,
  f_faire=args.faire_pickle,
 # ref_fasta = "/mnt/j/db/hg38/ref/hg38.analysisSet.fa", # ALCH
  ref_fasta = args.ref_fasta, #"/mnt/j/db/hg19/ref/hs37d5.fa", # Richter's
  bin_width = args.bin_width
)
C = cov_mcmc_runner.full_cov_df

# bin intervals by GC content
C["GC_bin"] = np.round(C["C_GC"]*1000).astype(int)
C["num_frags_corr"] = C["covcorr"]/C["C_frag_len"].mean()

N_gc = C.groupby("GC_bin").size()
F_gc = C.groupby("GC_bin")["num_frags_corr"].sum()

plt.figure(1); plt.clf()
plt.scatter(N_gc.index, F_gc/N_gc, marker = '.', s = 1)

cov_df = pd.read_pickle("/mnt/nfs/HapASeg_Richters/CH1001LN-CH1001GL/Hapaseg_prepare_coverage_mcmc__2022-05-16--15-35-16_040rmzi_pid3cty_0w3oyu5xxnfwe/jobs/0/workspace/cov_df.pickle")
cov_df = cov_df.merge(C[["start_g", "C_GC"]], left_on = "start_g", right_on = "start_g")

cov_df["GC_bin"] = np.round(cov_df["C_GC"]*1000).astype(int)
cov_df["num_frags_corr"] = cov_df["covcorr"]/cov_df["C_frag_len"].mean()

N_gc = cov_df.groupby("GC_bin").size()
F_gc = cov_df.groupby("GC_bin")["num_frags_corr"].sum()

cov_df = cov_df.merge((F_gc/N_gc).rename("C_GC_f"), left_on = cov_df["GC_bin"], right_index = True)

import loess
_, y_l, _ = loess_1d.loess_1d(np.r_[N_gc.index], np.r_[F_gc/N_gc])

plt.figure(2); plt.clf()
plt.scatter(N_gc.index, F_gc/N_gc, marker = '.', s = 1)
#plt.plot(N_gc.index, y_l)
r = np.linspace(0, 1000, 1000)
v = np.polyfit(np.r_[N_gc.index]/1000, F_gc/N_gc, 2)
plt.plot(r, v[::-1]@(r**np.c_[0:3]))
plt.ylim([0, 500])

from capy import plots

plt.figure(3); plt.clf()
plots.pixplot(cov_df["C_GC_f"], cov_df["num_frags_corr"], alpha = 0.11)
plots.pixplot(v[::-1]@(cov_df["C_GC"].values**np.c_[0:3]), cov_df["num_frags_corr"], alpha = 0.11)

gc_g = []
N_gc_g = []
F_gc_g = []
plt.figure(4); plt.clf()
for _, cidx in cov_df.groupby("allelic_cluster").indices.items():
    N_gc = cov_df.iloc[cidx].groupby("GC_bin").size()
    F_gc = cov_df.iloc[cidx].groupby("GC_bin")["num_frags_corr"].sum()
    lplt = plt.scatter(N_gc.index, (F_gc/N_gc)/F_gc.sum(), marker = '.', s = 1)

    v = np.polyfit(N_gc.index, F_gc/N_gc, 2)
    rng = np.linspace(0, 1000, 200)
    plt.plot(rng, v[::-1]@(rng**np.c_[0:3]), color = lplt.get_edgecolor())

    gc_g.extend(N_gc.index)
    N_gc_g.extend(N_gc)
    F_gc_g.extend(F_gc)

N_gc_g = np.r_[N_gc_g]
F_gc_g = np.r_[F_gc_g]
gc_g = np.r_[gc_g]

v = np.polyfit(gc_g, F_gc_g/N_gc_g, 2)
plt.plot(r, v[::-1]@(r**np.c_[0:3]))
_, y_l, _ = loess_1d.loess_1d(gc_g, F_gc_g/N_gc_g, xnew = r, degree = 2)
plt.plot(r, y_l)

plt.figure(3); plt.clf()
_, y_l, _ = loess_1d.loess_1d(gc_g/1000, F_gc_g/N_gc_g, xnew = cov_df["C_GC"], degree = 2)
plots.pixplot(cov_df["C_GC_f"], cov_df["num_frags_corr"], alpha = 0.11)

## simulate quadratic relationship
seg_sim = np.r_[np.ones([500, 1]), 1.5*np.ones([500, 1])].T
gc_sim = np.random.rand(1000)*0.6 + 0.2
rng = np.linspace(0, 1, 100)
x = stats.poisson.rvs(np.exp(-30*(gc_sim - 0.5)**2 + 5*seg_sim))[:, None]
C = np.c_[gc_sim**2, gc_sim]

import hapaseg.model_optimizers
PR = hapaseg.model_optimizers.PoissonRegression

Pi = np.r_[np.tile([1, 0], [500, 1]), np.tile([0, 1], [500, 1])]
pois_regr = PR(x, C, Pi)
pois_regr.fit()
pois_regr2 = PR(x, C[:, [1]], Pi)
pois_regr2.fit()
plt.figure(2); plt.clf()
plt.scatter(x, np.exp(Pi@pois_regr.mu + C@pois_regr.beta), marker = '.', s = 1)
plt.scatter(x, np.exp(Pi@pois_regr2.mu + C[:, [1]]@pois_regr2.beta), marker = '.', s = 1)


# }}}

# }}}

#
# DNAse HS/FAIRE {{{

## DNAse {{{

bw = pyBigWig.open("covars/wgEncodeUwDnaseGm12878RawRep1.bigWig")

# WGS (2kb chunks)
clen = seq.get_chrlens()
C = []
for i, chrname in enumerate(["chr" + str(x) for x in list(range(1, 23)) + ["X", "Y"]]):
    bins = np.r_[0:clen[i]:2000, clen[i]]; bins = np.c_[bins[:-1], bins[1:]]
    tmp = pd.DataFrame({ "chr" : chrname, "start" : bins[:, 0], "end" : bins[:, 1], "DNAse" : 0 })
    for j, (st, en) in enumerate(tqdm.tqdm(bins)):
        tmp.loc[j, "DNAse"] = np.nanmean(np.r_[bw.values(chrname, st, en)])
    C.append(tmp)

# preliminary results not so great; stick with FAIRE for now

# TODO: liftover to hg38

# WES

# }}}

## FAIRE {{{

## convert bigWig to FWB

# for some reason pyBigWig can't process this file
# bw = pyBigWig.open("covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.bigwig")

# use bigWig2FWB instead
# git clone git@github.com:getzlab/bigWig2FWB.git

# figure out range of file
# bigWig2FWB/bigWig2FWB covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.bigWig covars/bwtest
# ./getmax
# -> max = 5478
# set scale factor to 11
# bigWig2FWB/bigWig2FWB covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.bigWig covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal

## WGS
from capy import fwb, mut

F = fwb.FWB("covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.fwb");

clen = seq.get_chrlens()
C = []
for i, chrname in enumerate(["chr" + str(x) for x in list(range(1, 23)) + ["X"]]):
    bins = np.r_[0:clen[i]:2000, clen[i]]; bins = np.c_[bins[:-1], bins[1:]]
    tmp = pd.DataFrame({ "chr" : chrname, "start" : bins[:, 0], "end" : bins[:, 1], "FAIRE" : 0 })
    for j, (st, en) in enumerate(tqdm.tqdm(bins)):
        tmp.loc[j, "FAIRE"] = F.get(chrname, np.r_[st:en] + 1).mean()
    C.append(tmp)

FAIRE = pd.concat(C, ignore_index = True)
FAIRE["chr"] = mut.convert_chr(FAIRE["chr"])
FAIRE.to_pickle("covars/FAIRE_GM12878.hg19.pickle")

# smoothed version
FAIRE_smooth = FAIRE.copy()
FAIRE_smooth["FAIRE"] = np.convolve(FAIRE["FAIRE"], np.ones(5), mode = "same")/5
FAIRE_smooth.to_pickle("covars/FAIRE_GM12878.smooth5.hg19.pickle")

# }}}

# }}}
