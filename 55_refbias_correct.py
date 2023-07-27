

#
# infer reference bias
chunk = runner.chunks["results"].iloc[1]
bdy = np.array(chunk.breakpoints); bdy = np.c_[bdy[:-1], bdy[1:]]

a = chunk.P.iloc[333:513]["MIN_COUNT"].sum()
b = chunk.P.iloc[333:513]["MAJ_COUNT"].sum()
(s.beta.rvs(a, b, 1000) - s.beta.rvs(b, a, 1000)).mean()

plt.figure(10); plt.clf()
plt.plot(np.linspace(0.48, 0.52, 200), s.beta.pdf(np.linspace(0.48, 0.52, 200), a, b))


refs = hapaseg.load.HapasegReference(phased_VCF = "exome/phased.vcf", allele_counts = "exome/6_C1D1_CFDNA.normal.tsv", ref_bias = 0.0)

st = 0; en = 118

H.P.iloc[st:en, H.P.columns.get_loc("ALT_COUNT")]
H.P.iloc[st:en, H.P.columns.get_loc("REF_COUNT")]

P = H.P.copy()

bias = P["ALT_COUNT"].sum()/P["REF_COUNT"].sum()
P["REF_COUNT"] *= bias

aidx = P["allele_A"] > 0
bidx = P["allele_B"] > 0

P["MAJ_COUNT"] = pd.concat([P.loc[aidx, "ALT_COUNT"], P.loc[bidx, "REF_COUNT"]])
P["MIN_COUNT"] = pd.concat([P.loc[aidx, "REF_COUNT"], P.loc[bidx, "ALT_COUNT"]])

bpl = np.array(H.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]
for st, en in bpl:
    print(s.beta.ppf([0.05, 0.5, 0.95], P.iloc[st:en, H.maj_idx].sum() + 1, P.iloc[st:en, H.min_idx].sum() + 1))

refs = hapaseg.load.HapasegReference(phased_VCF = "exome/phased.vcf", allele_counts = "exome/6_C1D1_CFDNA.normal.tsv", ref_bias = 0.936365296327212)
runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(refs.allele_counts, refs.chromosome_intervals, c, misphase_prior = 0.00001)
allelic_segs2 = runner.run_all()


alt_idx = H.P.columns.get_loc("ALT_COUNT")
ref_idx = H.P.columns.get_loc("REF_COUNT")
bpl = np.array(H.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]
probs = np.full(len(bpl), np.nan)
for i, (st, en) in enumerate(bpl):
    p = s.beta.cdf(0.5,
      H.P.iloc[st:en, alt_idx].sum() + 1,
      H.P.iloc[st:en, ref_idx].sum() + 1
    )
    probs[i] = np.min(2*np.r_[p, 1 - p])


allelic_segs = pd.read_pickle("exome/allelic_segs.pickle")
self = allelic_segs["results"].iloc[0]

st = 332; en = 508

idx_a = (self.P.index >= st) & (self.P.index < en) & self.P["aidx"]
idx_b = (self.P.index >= st) & (self.P.index < en) & ~self.P["aidx"]

plt.figure(10); plt.clf()
r = np.linspace(0.45, 0.55, 100)
plt.plot(r, s.beta.pdf(r, self.P.loc[idx_a, "MIN_COUNT"].sum() + 1, self.P.loc[idx_a, "MAJ_COUNT"].sum() + 1))
plt.plot(r, s.beta.pdf(r, self.P.loc[idx_b, "MAJ_COUNT"].sum() + 1, self.P.loc[idx_b, "MIN_COUNT"].sum() + 1))

self = allelic_segs["results"].iloc[1]

st = 485; en = 491

idx_a = (self.P.index >= st) & (self.P.index < en) & self.P["aidx"]
idx_b = (self.P.index >= st) & (self.P.index < en) & ~self.P["aidx"]

plt.figure(11); plt.clf()
r = np.linspace(0.3, 0.65, 100)
plt.plot(r, s.beta.pdf(r, self.P.loc[idx_a, "MIN_COUNT"].sum() + 1, self.P.loc[idx_a, "MAJ_COUNT"].sum() + 1))
plt.plot(r, s.beta.pdf(r, self.P.loc[idx_b, "MAJ_COUNT"].sum() + 1, self.P.loc[idx_b, "MIN_COUNT"].sum() + 1))

# separate
lik_S = ss.betaln(
  self.P.loc[idx_a, "MIN_COUNT"].sum() + 1, self.P.loc[idx_a, "MAJ_COUNT"].sum() + 1
) + \
ss.betaln(
  self.P.loc[idx_b, "MAJ_COUNT"].sum() + 1, self.P.loc[idx_b, "MIN_COUNT"].sum() + 1
)

# together
lik_T = ss.betaln(
  self.P.loc[idx_b, "MAJ_COUNT"].sum() + self.P.loc[idx_a, "MIN_COUNT"].sum() + 1,
  self.P.loc[idx_b, "MIN_COUNT"].sum() + self.P.loc[idx_a, "MAJ_COUNT"].sum() + 1
)


# take 20 random breakpoint samples
balanced_intervals = {}
for b_idx in np.random.choice(len(self.breakpoint_list), 20, replace = False):
    bpl = np.r_[self.breakpoint_list[b_idx]]; bpl = np.c_[bpl[:-1], bpl[1:]]
    # only look at segments with more than 2 SNPs
    bpl = bpl[np.diff(bpl, 1).ravel() > 2]
    for st, en in bpl:
        idx_a = (self.P.index >= st) & (self.P.index < en) & self.P["aidx"]
        idx_b = (self.P.index >= st) & (self.P.index < en) & ~self.P["aidx"]

        # separate
        lik_S = ss.betaln(
          self.P.loc[idx_a, "MIN_COUNT"].sum() + 1, self.P.loc[idx_a, "MAJ_COUNT"].sum() + 1
        ) + \
        ss.betaln(
          self.P.loc[idx_b, "MAJ_COUNT"].sum() + 1, self.P.loc[idx_b, "MIN_COUNT"].sum() + 1
        )

        # together
        lik_T = ss.betaln(
          self.P.loc[idx_b, "MAJ_COUNT"].sum() + self.P.loc[idx_a, "MIN_COUNT"].sum() + 1,
          self.P.loc[idx_b, "MIN_COUNT"].sum() + self.P.loc[idx_a, "MAJ_COUNT"].sum() + 1
        )

        if lik_T - lik_S > 3.5:
            balanced_intervals[(st, en)] = (lik_T-lik_S, lik_T, lik_S, self.P.loc[st:(en - 1), "ALT_COUNT"].sum(), self.P.loc[st:(en - 1), "REF_COUNT"].sum())

        print(lik_T - lik_S, st, en)


######

bp_samp = A.breakpoint_list[0]
bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]

X = []
for i, (st, en) in enumerate(bpl):
    x = A.P.iloc[st:en].groupby("allele_A")[["MIN_COUNT", "MAJ_COUNT"]].sum()
    x["idx"] = i
    X.append(x)

X = pd.concat(X)
g = X.groupby("idx").size() == 2
Y = X.loc[X["idx"].isin(g[g].index)]

f = np.zeros([len(Y)//2, 100])
for i, (_, g) in enumerate(Y.groupby("idx")):
    f[i] = g.loc[0, "MIN_COUNT"]/(g.loc[0, "MIN_COUNT"] + g.loc[0, "MAJ_COUNT"])/(g.loc[1, "MIN_COUNT"]/(g.loc[1, "MIN_COUNT"] + g.loc[1, "MAJ_COUNT"]))

    f[i, :] = s.beta.rvs(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1, size = 100)/s.beta.rvs(g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1, size = 100)
    #f[i] = g.loc[0, "MIN_COUNT"]/(g.loc[0, "MIN_COUNT"] + g.loc[0, "MAJ_COUNT"])/(g.loc[1, "MIN_COUNT"]/(g.loc[1, "MIN_COUNT"] + g.loc[1, "MAJ_COUNT"]))

f.mean() # gives a pretty good overall estimate in line with empirical values

# can we add covariates to improve this?

# 0. distance to target/bait boundary
T = pd.read_csv("exome/broad_custom_exome_v1.Homo_sapiens_assembly19.targets.interval_list", comment = "@", sep = "\t", header = None, names = ["chr", "start", "end", "x", "y"]).loc[:, ["chr", "start", "end"]]
B = pd.read_csv("exome/broad_custom_exome_v1.Homo_sapiens_assembly19.baits.interval_list", comment = "@", sep = "\t", header = None, names = ["chr", "start", "end", "x", "y"]).loc[:, ["chr", "start", "end"]]
T = T.append(pd.Series({ "chr" : -1, "start" : -1, "end" : -1 }, name = -1))

from capy import mut
A.P["targ"] = -1
tmap = mut.map_mutations_to_targets(A.P, T, inplace = False).astype(np.int64)
A.P.loc[tmap.index, "targ"] = tmap

# for targets that mapped, distance from closest boundary
# or relative position [0, 1] within target?
A.P["dists"] = np.abs(np.c_[A.P["pos"].values] - T.loc[A.P["targ"], ["start", "end"]].values).min(1)

# for targets that didn't map, get closest target
Pg = A.P.groupby("chr")
Tg = T.groupby("chr")

for ch, g in Pg:
    if ch not in Tg.groups:
        continue

    nomap = g.loc[g["targ"] == -1, "pos"]

    Tch = Tg.get_group(ch)
    Tch_e = Tch.sort_values("end", ignore_index = True) # sort by end as well
    # nearest targets
    nidx_l = Tch["start"].searchsorted(nomap, side = "left")
    nidx_r = Tch_e["end"].searchsorted(nomap, side = "right")

    A.P.loc[nomap.index, "dists"] = -np.c_[
      Tch.loc[nidx_l, "start"].values - nomap.values,
      nomap.values - Tch_e.loc[nidx_r - 1, "end"].values
    ].min(1)

A.P["seg_res"] = np.nan
for st, en in bpl:
    if en - st < 10:
        continue

    snp_f = s.beta.rvs(A.P.loc[st:en, "MIN_COUNT"] + 1, A.P.loc[st:en, "MAJ_COUNT"] + 1, size = (100, en - st + 1))
    seg_f = s.beta.rvs(A.P.loc[st:en, "MIN_COUNT"].sum() + 1, A.P.loc[st:en, "MAJ_COUNT"].sum() + 1, size = (100, 1))

    A.P.loc[st:en, "seg_res"] = np.abs(np.log((snp_f/seg_f).mean(0)))

plt.figure(1337); plt.clf()
plt.scatter(A.P["dists"], A.P["seg_res"], alpha = 0.5)

# 1. use matched normal as empirical measure of ref bias

import hapaseg
refs = hapaseg.load.HapasegReference(phased_VCF = "exome/6_C1D1_META.eagle.vcf", allele_counts = "exome/6_C1D1_META.tumor.tsv", allele_counts_N = "exome/6_C1D1_META.normal.tsv")

A = refs.allele_counts
f_T = A["ALT_COUNT"]/(A["REF_COUNT"] + A["ALT_COUNT"])
f_N = A["ALT_COUNT_N"]/(A["REF_COUNT_N"] + A["ALT_COUNT_N"])

T_het_idx = s.beta.cdf(0.6, A["ALT_COUNT"] + 1, A["REF_COUNT"] + 1) - s.beta.cdf(0.4, A["ALT_COUNT"] + 1, A["REF_COUNT"] + 1) > 0.7

plt.figure(21); plt.clf()
plt.scatter(f_T.loc[T_het_idx], f_N[T_het_idx], alpha = 0.2)
plt.scatter(f_T, f_N, alpha = 0.05)
plt.xlabel("Tumor")
plt.ylabel("Normal")
plt.xlim([0.5, 1.5])
plt.ylim([0.5, 1.5])

#######

# new method of ref bias correction; 1. compute scale factor empirically
# 2. also exclude segments where we're not powered to compute it

# for 1., use a heavily biased exome
# for 2., use a cell line

import pickle
import pandas as pd
import numpy as np
import scipy.stats as ss
import wolf

hapaseg_workflow = wolf.ImportTask(".", main_task = "hapaseg_workflow")

! gsutil cat gs://getzlab-workflows-reference_files-oa/hg19/twist/broad_custom_exome_v1.Homo_sapiens_assembly19.targets.interval_list | sed '/^@/d' | cut -f1-3 > hg19_twist.bed

## 1. 

with wolf.Workflow(workflow = hapaseg_workflow, namespace = "refbias_fix") as w:
    w.run(
      RUN_NAME = "asgari_exome",
      tumor_bam="gs://fc-8cc4e03b-d91c-4abb-a63b-c79dc0251f60/Getz_Asgari_Cutaneous_Squamous_Cell_Carcinoma_2015P000925_Exomes_91samples_PDO-27396_27397_May2022/RP-2457/Exome/AG17RT/v1/AG17RT.bam",
      tumor_bai="gs://fc-8cc4e03b-d91c-4abb-a63b-c79dc0251f60/Getz_Asgari_Cutaneous_Squamous_Cell_Carcinoma_2015P000925_Exomes_91samples_PDO-27396_27397_May2022/RP-2457/Exome/AG17RT/v1/AG17RT.bai",
      normal_bam="gs://fc-8cc4e03b-d91c-4abb-a63b-c79dc0251f60/Getz_Asgari_PQ3_55samples_WES_May2022/RP-2457/Exome/AG17RN/v1/AG17RN.bam",
      normal_bai="gs://fc-8cc4e03b-d91c-4abb-a63b-c79dc0251f60/Getz_Asgari_PQ3_55samples_WES_May2022/RP-2457/Exome/AG17RN/v1/AG17RN.bai",
      normal_coverage_bed=None,
      ref_genome_build="hg19",
      target_list="./hg19_twist.bed",
      is_ffpe=True,
    )

# load in results
w = wolf.Workflow(workflow = lambda:None, namespace = "refbias_fix")
w.load_results("asgari_exome")

args = lambda:None
args.chunks = list(pd.read_csv("/mnt/nfs/refbias_fix/asgari_exome/Hapaseg_concat__2023-07-25--15-44-10_epc5bvq_npn15qq_zep0omnayyems/jobs/0/chunks_array.txt", header = None).iloc[:, 0])
args.scatter_intervals = "/mnt/nfs/refbias_fix/asgari_exome/Hapaseg_concat__2023-07-25--15-44-10_epc5bvq_npn15qq_zep0omnayyems/jobs/0/inputs/scatter_chunks.tsv"

## excised from __main__.py

#
# load scatter intervals
intervals = pd.read_csv(args.scatter_intervals, sep="\t")

if len(intervals) != len(args.chunks):
    raise ValueError("Length mismatch in supplied chunks and interval file!")

# load results
R = []
for chunk_path in args.chunks:
    with open(chunk_path, "rb") as f:
        chunk = pickle.load(f)
    R.append(chunk)
R = pd.DataFrame({"results": R})

# ensure results are in the correct order
R["first"] = R["results"].apply(lambda x: x.P.loc[0, "index"])
R = R.sort_values("first", ignore_index=True)

# concat with intervals
R = pd.concat([R, intervals], axis=1).drop(columns=["first"])

X = []
j = 0
for chunk in R["results"]:
    bpl = np.array(chunk.breakpoints);
    bpl = np.c_[bpl[0:-1], bpl[1:]]

    for i, (st, en) in enumerate(bpl):
        g = chunk.P.iloc[st:en].groupby("allele_A")
        x = g[["REF_COUNT", "ALT_COUNT"]].sum()
        x["idx"] = i + j
        x["n_SNP"] = g.size()
        X.append(x)

    j += i + 1

X = pd.concat(X)
g = X.groupby("idx").size() == 2

## new code to compute reference bias
X = X.loc[X["idx"].isin(g[g].index)]
X = X.set_index([X["idx"], X.index]).drop(columns = "idx")

# TODO: remove segments with too few reference SNPs (due to purity ~100%)
#       for now, only use segments with at least 20 SNPs assigned to each haplotype

tot_SNPs = X.groupby(level = 0)["n_SNP"].sum()
use_idx = X.groupby(level = 0)["n_SNP"].apply(lambda x : (x > 20).all())

refbias_dom = np.linspace(0.8, 1, 10)
plt.figure(3); plt.clf()
for opt_iter in range(3):
    n_beta_samp = np.r_[10, 100, 10000][opt_iter]
    if opt_iter > 0:
        refbias_dom = np.linspace(
          *refbias_dom[np.argmin(refbias_dif) + np.r_[-2, 2]],
          np.r_[10, 20, 30][opt_iter]
        )
    refbias_dif = np.full(len(refbias_dom), np.nan)
    for j, rb in enumerate(refbias_dom):
        absdif = np.full(use_idx.sum(), np.nan)
        for i, seg in enumerate(tot_SNPs.index[use_idx]):
            f_A = ss.beta.rvs(
              X.loc[(seg, 0), "ALT_COUNT"] + 1,
              X.loc[(seg, 0), "REF_COUNT"]*rb + 1,
              size = n_beta_samp
            )
            f_B = ss.beta.rvs(
              X.loc[(seg, 1), "REF_COUNT"]*rb + 1,
              X.loc[(seg, 1), "ALT_COUNT"] + 1,
              size = n_beta_samp
            )

            absdif[i] = np.abs(f_A - f_B).mean()

        refbias_dif[j] = absdif@tot_SNPs[use_idx]/tot_SNPs[use_idx].sum()

    ref_bias = refbias_dom[np.argmin(refbias_dif)]
    plt.scatter(refbias_dom, refbias_dif, marker = "x")
