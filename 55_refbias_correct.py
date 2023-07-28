

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
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import sys
import tqdm
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

## 2.

args = lambda:None
args.chunks = list(pd.read_csv("./Hapaseg_concat__2023-06-09--05-41-56_epc5bvq_npn15qq_34ceqoi0eehzw/jobs/0/chunks_array.txt", header = None).iloc[:, 0].str.replace("/mnt/nfs/workspace/hg38_wgs_HCM-BROD-0214-C71_TNGCM-NB/", "./"))
args.scatter_intervals = "./Hapaseg_concat__2023-06-09--05-41-56_epc5bvq_npn15qq_34ceqoi0eehzw/jobs/0/inputs/scatter_chunks.tsv"

# re-run
with wolf.Workflow(workflow = hapaseg_workflow, namespace = "refbias_fix") as w:
    w.run(
      RUN_NAME = "HCMI_genome",
      tumor_bam="https://api.awg.gdc.cancer.gov/data/298e0d6a-8876-4038-8625-b0d518aabbba",
      tumor_bai="https://api.awg.gdc.cancer.gov/data/069ac47e-8bd2-4123-815e-23938dcc83d3",
      normal_bam="https://api.awg.gdc.cancer.gov/data/e3461169-fc11-41a3-b858-6f939421d644",
      normal_bai="https://api.awg.gdc.cancer.gov/data/e8c838a8-655c-42cf-82c1-6961858b2d57",
      localization_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJzdWIiOiI4N2YwNTQ4ZGEyYzE0YzI0YWNhMTMzODk3YjQwMGY0YSIsImlhdCI6MTY5MDI5NjQ5NiwiZXhwIjoxNjkyODg4NDk2LCJvcGVuc3RhY2tfbWV0aG9kcyI6WyJzYW1sMiJdLCJvcGVuc3RhY2tfYXVkaXRfaWRzIjpbIlQxdlBlWU15UlppdE1KSHBDdXc3UEEiXSwib3BlbnN0YWNrX2dyb3VwX2lkcyI6W3siaWQiOiIwNjNlYWEyN2NjMTY0MzhlODJkYjVlYzA2ZDkzYTUzYyJ9XSwib3BlbnN0YWNrX2lkcF9pZCI6ImVyYV9jb21tb24iLCJvcGVuc3RhY2tfcHJvdG9jb2xfaWQiOiJzYW1sMiJ9.LS_om2S9y_OON11RRaUMxVm8KAcK8Po19kPpu1mk0KUCvP4CJAadDPfLl-ik4Rwe50eAnl6ZlYtSB4hpIQfQAQapi_key:eyJ0eXAiOiJKV1QiLCJraWQiOiJrZXktMDEiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJodHRwczovL2xvZ2luLmF3Zy5nZGMuY2FuY2VyLmdvdiIsImF1ZCI6WyJmZW5jZSIsIm9wZW5pZCJdLCJwdXIiOiJhcGlfa2V5IiwianRpIjoiOTg4MDQwYjctZmJiNy00ZGJjLWEyYWQtODUxOTI3M2RiYTZlIiwic3ViIjoiNzAiLCJleHAiOjE2OTI4ODg0OTYsImlhdCI6MTY5MDI5NjQ5NiwiYXpwIjoiIn0.OAWektpb884Ox1vZNFmJnwqHhautmWZTsIdh49NJutAOiLMgU9uwPrGOqlMjiRLCYkDJ0x1OsTAnzo9gzkW1ZA-_43-k471iI26HVn87h9J6EBI9Mh2a3G5CWu4dg1q8xhanjU3DtMJNnGJLq_2hnEt9b8SW8NL6BFHn_nAIrPFW_RwhqPPdIBtxPcf4kTDo4DJOHMSfbh4rVpOvF9UemeolkfwqHJoVwqEmE8gHJw2xIT-Nfokq0GsDrPU9RtXxvu_4peo_b3HW53_bzAorSG0T9Etr1Q_wmpbbITRM2eIDW2xsjBsRlQ2ruXTcsFF-i2hUS8AwwSwsZi80A369dA",
      ref_genome_build="hg38",
      target_list=2000,
    )

w = wolf.Workflow(workflow = lambda:None, namespace = "refbias_fix")
w.load_results("HCMI_genome")

args = lambda:None
args.chunks = list(pd.read_csv("/mnt/nfs/refbias_fix/HCMI_genome/Hapaseg_concat__2023-07-28--16-48-45_epc5bvq_npn15qq_0azrc5r3ufiry/jobs/0/chunks_array.txt", header = None).iloc[:, 0])
args.scatter_intervals = "/mnt/nfs/refbias_fix/HCMI_genome/Hapaseg_concat__2023-07-28--16-48-45_epc5bvq_npn15qq_0azrc5r3ufiry/jobs/0/inputs/scatter_chunks.tsv"

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

# {{{
Y = X.set_index([X["idx"], X.index]).drop(columns = "idx")

plt.figure(1); plt.clf()
aidx = Y.index.get_level_values(1) == 0
bidx = Y.index.get_level_values(1) == 1
plt.scatter(Y.index[aidx].get_level_values(0), Y.loc[aidx, "ALT_COUNT"]/Y.loc[aidx, ["ALT_COUNT", "REF_COUNT"]].sum(1), color = 'r', marker = ".")
plt.scatter(Y.index[bidx].get_level_values(0), Y.loc[bidx, "REF_COUNT"]/Y.loc[bidx, ["ALT_COUNT", "REF_COUNT"]].sum(1), color = 'b', marker = ".")

f_a = np.full(Y.index.get_level_values(0).max() + 1, np.nan)
f_b = np.full(Y.index.get_level_values(0).max() + 1, np.nan)

f_a[Y.index[aidx].get_level_values(0)] = Y.loc[aidx, "ALT_COUNT"]/Y.loc[aidx, ["ALT_COUNT", "REF_COUNT"]].sum(1)
f_b[Y.index[bidx].get_level_values(0)] = Y.loc[bidx, "REF_COUNT"]/Y.loc[bidx, ["ALT_COUNT", "REF_COUNT"]].sum(1)

plt.figure(2); plt.clf()
plt.scatter(f_a, f_b, marker = ".", alpha = 0.1)
plt.plot([0, 1], [0, 1], color = 'r')
plt.xlabel("f\_alt")
plt.ylabel("f\_ref")

Z = X.loc[X["idx"].isin(g[g].index)]
Z = Z.set_index([Z["idx"], Z.index]).drop(columns = "idx")

Z.index.get_level_values(1)
# }}}



## new code to compute reference bias
X = X.loc[X["idx"].isin(g[g].index)]
X = X.set_index([X["idx"], X.index]).drop(columns = "idx")

# don't use LoH segments at ~100% purity in reference bias calculations, since
# these consistently have f_alt ~ 1, f_ref ~ 0, yielding optimal reference bias of 1
# segments with very little allelic imbalance density between 0.1 and 0.9 are considered LoH
a = X.loc[(slice(None), 0), "ALT_COUNT"].droplevel(1) + X.loc[(slice(None), 1), "REF_COUNT"].droplevel(1) + 1
b = X.loc[(slice(None), 0), "REF_COUNT"].droplevel(1) + X.loc[(slice(None), 1), "ALT_COUNT"].droplevel(1) + 1
rbdens = ss.beta.cdf(0.9, a, b) - ss.beta.cdf(0.1, a, b)

# plot beta densitites of all segments
plt.figure(6); plt.clf()
r = np.r_[np.linspace(0, 0.05, 200), np.linspace(0.05, 0.95, 200), np.linspace(0.95, 1, 200)]
for _, A, B in pd.concat([a, b], axis = 1).itertuples():
    plt.plot(r, ss.beta.pdf(r, A, B), alpha = 0.1, color = 'k')

# X = X.loc[X.index.get_level_values(0) < 5400]
# X = X.loc[(X.index.get_level_values(0) > 5500) & (X.index.get_level_values(0) < 6000)]

tot_SNPs = X.groupby(level = 0)["n_SNP"].sum()
# we also don't want to use segments with too few supporting SNPs (20) in reference
# bias calculations
use_idx = X.groupby(level = 0)["n_SNP"].apply(lambda x : (x > 20).all()) & (rbdens > 0.01)

# visualizations {{{

# show all SNPs
plt.figure(1); plt.clf()
aidx = X.index.get_level_values(1) == 0
bidx = X.index.get_level_values(1) == 1
plt.scatter(X.index[aidx].get_level_values(0), X.loc[aidx, "ALT_COUNT"]/X.loc[aidx, ["ALT_COUNT", "REF_COUNT"]].sum(1), color = 'r', marker = ".", alpha = np.minimum(1, 10*tot_SNPs/tot_SNPs.max()))
plt.scatter(X.index[bidx].get_level_values(0), X.loc[bidx, "REF_COUNT"]/X.loc[bidx, ["ALT_COUNT", "REF_COUNT"]].sum(1), color = 'b', marker = ".", alpha = np.minimum(1, 10*tot_SNPs/tot_SNPs.max()))
plt.ylim([-0.05, 1.05])

# show just SNPs being used in refbias calculation
plt.figure(10); plt.clf()
cidx = use_idx & (rbdens > 0.01)
X2 = X.loc[cidx[cidx].index]
aidx = X2.index.get_level_values(1) == 0
bidx = X2.index.get_level_values(1) == 1
plt.scatter(X2.index[aidx].get_level_values(0), X2.loc[aidx, "ALT_COUNT"]/X2.loc[aidx, ["ALT_COUNT", "REF_COUNT"]].sum(1), color = 'r', marker = ".", alpha = np.minimum(1, 10*tot_SNPs[cidx]/tot_SNPs.max()))
plt.scatter(X2.index[bidx].get_level_values(0), X2.loc[bidx, "REF_COUNT"]/X2.loc[bidx, ["ALT_COUNT", "REF_COUNT"]].sum(1), color = 'b', marker = ".", alpha = np.minimum(1, 10*tot_SNPs[cidx]/tot_SNPs.max()))
plt.ylim([-0.05, 1.05])

# }}}

refbias_dom = np.linspace(0.8, 1, 10)
plt.figure(3); plt.clf()
print("Computing reference bias ...", file = sys.stderr)
for opt_iter in range(3):
    # number of Monte Carlo samples to draw from beta distribution
    # we need fewer samples for more total SNPs
    n_beta_samp = np.r_[10, 100, np.maximum(100, int(1e8/tot_SNPs.sum()))][opt_iter]
    # perform increasingly fine grid searches around neighborhood of previous optimum
    if opt_iter > 0:
        refbias_dom = np.linspace(
          *refbias_dom[np.minimum(np.argmin(refbias_dif) + np.r_[-2, 2], len(refbias_dom) - 1)], # search +- 2 grid points of previous optimum
          np.r_[10, 20, 30][opt_iter] # fineness of grid search
        )
    refbias_dif = np.full(len(refbias_dom), np.inf)
    pbar = tqdm.tqdm(enumerate(refbias_dom), total = len(refbias_dom))
    for j, rb in pbar:
        pbar.set_description(f"[{refbias_dom.min():0.2f}:{refbias_dom.max():0.2f}:{len(refbias_dom)}] {refbias_dom[refbias_dif.argmin()]:0.4f} ({n_beta_samp} MC samples)")
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

# more visualizations {{{

## show all SNPs after correction
plt.figure(2); plt.clf()
aidx = X.index.get_level_values(1) == 0
bidx = X.index.get_level_values(1) == 1
plt.scatter(X.index[aidx].get_level_values(0), X.loc[aidx, "ALT_COUNT"]/(X.loc[aidx, ["ALT_COUNT", "REF_COUNT"]]*np.r_[1, ref_bias]).sum(1), color = 'r', marker = ".", alpha = np.minimum(1, 10*tot_SNPs/tot_SNPs.max()))
plt.scatter(X.index[bidx].get_level_values(0), X.loc[bidx, "REF_COUNT"]*ref_bias/(X.loc[bidx, ["ALT_COUNT", "REF_COUNT"]]*np.r_[1, ref_bias]).sum(1), color = 'b', marker = ".", alpha = np.minimum(1, 10*tot_SNPs/tot_SNPs.max()))

## scatterplot of A allele fraction vs. B allele fraction before/after correction
f_a = np.full(X.index.get_level_values(0).max() + 1, np.nan)
f_b = np.full(X.index.get_level_values(0).max() + 1, np.nan)

f_a[X.index[aidx].get_level_values(0)] = X.loc[aidx, "ALT_COUNT"]/X.loc[aidx, ["ALT_COUNT", "REF_COUNT"]].sum(1)
f_b[X.index[bidx].get_level_values(0)] = X.loc[bidx, "REF_COUNT"]/X.loc[bidx, ["ALT_COUNT", "REF_COUNT"]].sum(1)

f_acorr = np.full(X.index.get_level_values(0).max() + 1, np.nan)
f_bcorr = np.full(X.index.get_level_values(0).max() + 1, np.nan)

f_acorr[X.index[aidx].get_level_values(0)] = X.loc[aidx, "ALT_COUNT"]/(X.loc[aidx, ["ALT_COUNT", "REF_COUNT"]]*np.r_[1, ref_bias]).sum(1)
f_bcorr[X.index[bidx].get_level_values(0)] = X.loc[bidx, "REF_COUNT"]*ref_bias/(X.loc[bidx, ["ALT_COUNT", "REF_COUNT"]]*np.r_[1, ref_bias]).sum(1)

plt.figure(4); plt.clf()
plt.scatter(f_a, f_b, marker = ".", alpha = 0.5)
plt.plot([0, 1], [0, 1], color = 'r')
plt.xlabel("f\_alt")
plt.ylabel("f\_ref")

plt.scatter(f_acorr, f_bcorr, marker = "+", alpha = 0.5)

# }}}
