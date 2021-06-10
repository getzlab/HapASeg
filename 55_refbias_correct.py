

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

# 1. panel of tumors?
