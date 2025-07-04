import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc
import sys

from capy import txt

#
# pull down het sites from gnomAD
/mnt/j/proj/cnv/20200909_hetpull/hetpull.py -c 3328_WGS.MuTect1.call_stats.txt \
 -s /mnt/j/db/hg19/gnomad/ACNV_sites/gnomAD_MAF10.txt -r /mnt/j/db/hg19/ref/hs37d5.fa -o 3328 -g

#
# convert to VCF

bcftools convert --tsv2vcf 3328.genotype.tsv -c CHROM,POS,AA -s 3328 -f /mnt/j/db/hg19/ref/hs37d5.fa -Ov -o test.vcf

#
# phase

eagle ...

# compute which sites in the SNP list are confidently heterozygous in the normal
# bdens = \int_{af_lb}^{af_ub} df beta(f | n_alt + 1, n_ref + 1)

# load in phased VCF
P = pd.read_csv("test.vcf", sep = "\t", comment = "#", names = ["chr", "pos", "x", "ref", "alt", "y", "z", "a", "b", "hap"], header = None)
P = P.loc[:, ~P.columns.str.match('^.$')]

P = txt.parsein(P, 'hap', r'(.)\|(.)', ["allele_A", "allele_B"]).astype({"allele_A" : int, "allele_B" : int })

# pull in altcounts
C = pd.read_csv("3328.tumor.tsv", sep = "\t")
P = P.merge(C, how = "inner", left_on = ["chr", "pos"], right_on = ["CONTIG", "POSITION"]).drop(columns = ["CONTIG", "POSITION"])

# also get normal altcounts
C = pd.read_csv("3328.normal.tsv", sep = "\t")
P = P.merge(C, how = "inner", left_on = ["chr", "pos"], right_on = ["CONTIG", "POSITION"], suffixes = (None, "_N")).drop(columns = ["CONTIG", "POSITION"])

# note that implicitly drops homozygous sites, since they were never had their
# alt/refcounts computed. perhaps we should use these as coverage probes?

#
# alt/ref -> major/minor
aidx = P["allele_A"] > 0
bidx = P["allele_B"] > 0
P["aidx"] = P["allele_A"] > 0

P["MAJ_COUNT"] = pd.concat([P.loc[aidx, "ALT_COUNT"], P.loc[bidx, "REF_COUNT"]])
P["MIN_COUNT"] = pd.concat([P.loc[aidx, "REF_COUNT"], P.loc[bidx, "ALT_COUNT"]])

#
# compute beta CI's
CI = s.beta.ppf([0.05, 0.5, 0.95], P["ALT_COUNT"][:, None] + 1, P["REF_COUNT"][:, None] + 1)
P[["CI_lo", "median", "CI_hi"]] = CI

CI = s.beta.ppf([0.05, 0.5, 0.95], P["MAJ_COUNT"][:, None] + 1, P["MIN_COUNT"][:, None] + 1)
P[["CI_lo_hap", "median_hap", "CI_hi_hap"]] = CI

# don't bother phasing for normal
CI = s.beta.ppf([0.05, 0.5, 0.95], P["ALT_COUNT_N"][:, None] + 1, P["REF_COUNT_N"][:, None] + 1)
P[["CI_lo_N", "median_N", "CI_hi_N"]] = CI

#
# visualize

# major/minor

cmap = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]]

plt.figure(4); plt.clf()
Ph = P.iloc[:40000]
#plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.75)
plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.75, color = cmap[aidx.astype(np.int)])
plt.errorbar(Ph["pos"], y = Ph["median_N"], yerr = np.c_[Ph["median_N"] - Ph["CI_lo_N"], Ph["CI_hi_N"] - Ph["median_N"]].T, fmt = 'none', alpha = 0.25, color = "g")
#plt.xlim([0, 36e6])
plt.xticks(np.linspace(*plt.xlim(), 20), P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)))
plt.xlabel("SNP index")

plt.figure(5); plt.clf()
plt.scatter(Ph["pos"], Ph["REF_COUNT"] + Ph["ALT_COUNT"], alpha = 0.5)
#plt.scatter(Ph["pos"], Ph["REF_COUNT_N"] + Ph["ALT_COUNT_N"], alpha = 0.5)

plt.figure(6); plt.clf()
cov = Ph["REF_COUNT"] + Ph["ALT_COUNT"]
plt.errorbar(Ph["pos"], y = cov*Ph["median_hap"], yerr = cov[None, :]*np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.75, color = cmap[aidx.astype(np.int)])

# alt/ref

PA = Ph.loc[aidx]
PB = Ph.loc[bidx]
plt.figure(2); plt.clf()
plt.errorbar(PA["pos"], y = PA["median"], yerr = np.c_[PA["median"] - PA["CI_lo"], PA["CI_hi"] - PA["median"]].T, fmt = 'none', alpha = 0.75)
plt.errorbar(PB["pos"], y = PB["median"], yerr = np.c_[PB["median"] - PB["CI_lo"], PB["CI_hi"] - PB["median"]].T, fmt = 'none', alpha = 0.75)
plt.xticks(np.linspace(*plt.xlim(), 20), P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)))
plt.xlabel("SNP index (A)")

#
#

min_idx = P.columns.get_loc("MIN_COUNT")
maj_idx = P.columns.get_loc("MAJ_COUNT")
alt_idx = P.columns.get_loc("ALT_COUNT")
ref_idx = P.columns.get_loc("REF_COUNT")

#
# load functions from Hapaseg

sys.path.append(".")
import hapaseg

#
# save
# PP = H.P
# BP = H.breakpoints
# CSMA = H.cs_MAJ
# CSMI = H.cs_MIN
# SPLP = H.split_prob
# SML = H.seg_marg_liks
# ML = H.marg_lik

H = hapaseg.A_MCMC(P.iloc[0:10000], quit_after_burnin = True)

#
# scatter over tiles up to burnin, then run on concatenation

chunks = np.r_[0:len(P):5000, len(P)]
chunk_bdy = np.c_[chunks[0:-1], chunks[1:]]

import dask.distributed as dd

c = dd.Client()

class Poo:
    def __init__(self, P, c):
        self.client = c
        self.P = c.scatter(P)

    @staticmethod
    def run(rng, P):
        H = hapaseg.A_MCMC(P.iloc[rng], quit_after_burnin = True)
        H.run()
        return H

    def run_all(self, ranges):
        return self.client.map(self.run, ranges, P = self.P)

p = Poo(P, c)

# run scatter
results = p.run_all([slice(*x) for x in chunk_bdy])

# gather
results_g = c.gather(results)

# concat P dataframes
H = hapaseg.A_MCMC(P)
H.P = pd.concat([x.P for x in results_g], ignore_index = True)
H.P["index"] = range(0, len(H.P))

# concat breakpoint lists
breakpoints = [None]*len(chunk_bdy)
H.seg_marg_liks = sc.SortedDict()
for i, (r, b) in enumerate(zip(results_g, chunk_bdy[:, 0])):
    breakpoints[i] = np.array(r.breakpoints) + b
    for k, v in r.seg_marg_liks.items():
        H.seg_marg_liks[k + b] = v
H.breakpoints = sc.SortedSet(np.hstack(breakpoints))

H.marg_lik = np.full(H.n_iter, np.nan)
H.marg_lik[0] = np.array(H.seg_marg_liks.values()).sum()

H.run()

#
# test new framework
sys.path.append(".")
import hapaseg

import dask.distributed as dd

c = dd.Client()

refs = hapaseg.load.HapasegReference()

runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(refs.allele_counts, refs.chromosome_intervals, c)
allelic_segs = runner.run_all()

#allelic_segs.to_pickle("allelic_segs.pickle")

allelic_segs = pd.read_pickle("allelic_segs.pickle")

H = allelic_segs["results"].iloc[0]
H.P = H.P.drop(columns = ["level_0", "index"])
H2 = hapaseg.allelic_MCMC.A_MCMC(H.P)
H2.breakpoint_counter = H.breakpoint_counter
H2.breakpoint_list = H.breakpoint_list
H2.breakpoints = H.breakpoints

H = allelic_segs["results"].iloc[1]
H.P = H.P.drop(columns = ["level_0", "index"])
H2 = hapaseg.allelic_MCMC.A_MCMC(H.P)
H2.breakpoint_counter = H.breakpoint_counter
H2.breakpoint_list = H.breakpoint_list
H2.breakpoints = H.breakpoints


# to run on an individual chunk
H = hapaseg.allelic_MCMC.A_MCMC(refs.allele_counts.iloc[0:500])

#
# phase correction HMM

# compute misphase probs for each segment

allelic_segs = pd.read_pickle("allelic_segs.pickle")
H = allelic_segs["results"].iloc[0]
H.no_phase_correct = False

bpl = np.array(H.breakpoint_list[0]); bpl = np.c_[bpl[:-1], bpl[1:]]

p_mis = np.full(len(bpl) - 1, np.nan)
p_A = np.full(len(bpl) - 1, np.nan)
p_B = np.full(len(bpl) - 1, np.nan)

V = np.full([len(bpl) - 1, 2], np.nan)
B = np.zeros([len(bpl) - 1, 2], dtype = np.uint8)

for i, (st, mid, _, en) in enumerate(np.c_[bpl[:-1], bpl[1:]]):
    p_mis, p_nomis = H.prob_misphase([st, mid], [mid, en])

    # prob. that left segment is on hap. A
    p_A1 = s.beta.logsf(0.5, H.P.iloc[st:mid, H.min_idx].sum() + 1, H.P.iloc[st:mid, H.maj_idx].sum() + 1)
    # prob. that right segment is on hap. A
    p_A2 = s.beta.logsf(0.5, H.P.iloc[mid:en, H.min_idx].sum() + 1, H.P.iloc[mid:en, H.maj_idx].sum() + 1)

    # prob. that left segment is on hap. B
    p_B1 = s.beta.logcdf(0.5, H.P.iloc[st:mid, H.min_idx].sum() + 1, H.P.iloc[st:mid, H.maj_idx].sum() + 1)
    # prob. that right segment is on hap. B
    p_B2 = s.beta.logcdf(0.5, H.P.iloc[mid:en, H.min_idx].sum() + 1, H.P.iloc[mid:en, H.maj_idx].sum() + 1)

    if i == 0:
        V[i, :] = [p_A1, p_B1]
        continue

    p_AB = p_mis + p_A1 + p_B2
    p_BA = p_mis + p_B1 + p_A2
    p_AA = p_nomis + p_A1 + p_A2
    p_BB = p_nomis + p_B1 + p_B2

    V[i, 0] = np.max(np.r_[p_AA + V[i - 1, 0], p_BA + V[i - 1, 1]])
    V[i, 1] = np.max(np.r_[p_AB + V[i - 1, 0], p_BB + V[i - 1, 1]])

    B[i, 0] = np.argmax(np.r_[p_AA + V[i - 1, 0], p_BA + V[i - 1, 1]])
    B[i, 1] = np.argmax(np.r_[p_AB + V[i - 1, 0], p_BB + V[i - 1, 1]])

# backtrace
BT = np.full(len(B), -1, dtype = np.uint8)
ix = np.argmax(V[-1])
BT[-1] = ix
for i, b in reversed(list(enumerate(B[:-1]))):
    ix = b[ix]
    BT[i] = ix

plt.figure(100); plt.clf()
for i, (st, en) in enumerate(bpl):
    a = H.P.iloc[st:en, H.min_idx]
    b = H.P.iloc[st:en, H.maj_idx]
    p = H.P.iloc[st:en, H.P.columns.get_loc("pos")]

    #plt.scatter(p, a/(a + b), color = np.r_[np.c_[0, 1, 1], np.c_[1, 1, 0]][BT[i]])
    if BT[i] == 0:
        plt.scatter(p, a/(a + b), color = 'k', s = 5, alpha = 0.2)
    else:
        plt.scatter(p, b/(a + b), color = 'k', s = 5, alpha = 0.2)

#
# load
# H.P = PP
# H.breakpoints = BP
# H.cs_MAJ = CSMA
# H.cs_MIN = CSMI
# H.split_prob = SPLP
# H.seg_marg_liks = SML
# H.marg_lik = ML

## first pass: merge sequentially from the left, up to N_INITIAL_PASSES times
## initial version will lack any memoization and be slow. we can add this later.
#
#for i in range(0, 30):
#    st = 0
#    while st != -1:
#        st = H.combine(st)
#
#bps = []
#while True:
#    last_len = len(H.breakpoints)
#    H.combine(np.random.choice(H.breakpoints[:-1]), force = False)
#    H.split(b_idx = np.random.choice(len(H.breakpoints)))
#    #H.combine(b_idx = np.random.choice(len(H.breakpoints)), force = False)
#    if len(H.breakpoints) != last_len:
#        print(len(H.breakpoints))
#    if H.burned_in and not H.iter % 100:
#        bps.append(H.breakpoints.copy())

#
# visualize

CI = s.beta.ppf([0.05, 0.5, 0.95], H.P["MAJ_COUNT"][:, None] + 1, H.P["MIN_COUNT"][:, None] + 1)
H.P[["CI_lo_hap", "median_hap", "CI_hi_hap"]] = CI

plt.figure(40); plt.clf()
ax = plt.gca()
Ph = H.P
#plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.75)
plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.5, color = cmap[aidx.astype(np.int)])

# phase switches
o = 0
for i in Ph["flip"].unique():
    if i == 0:
        continue
    plt.scatter(Ph.loc[Ph["flip"] == i, "pos"], o + np.zeros((Ph["flip"] == i).sum()))
    o -= 0.01

# breakpoints

bp_prob = H.breakpoint_counter[:, 0]/H.breakpoint_counter[:, 1]
bp_idx = np.flatnonzero(bp_prob > 0)
for i in bp_idx:
    col = 'k' if bp_prob[i] < 0.8 else 'm'
    alph = bp_prob[i]/2 if bp_prob[i] < 0.8 else bp_prob[i]
    plt.axvline(Ph.iloc[i, Ph.columns.get_loc("pos")], color = col, alpha = alph)
ax2 = ax.twiny()
ax2.set_xticks(Ph.iloc[H.breakpoints, Ph.columns.get_loc("pos")]);
ax2.set_xticklabels(bp_idx);
ax2.set_xlim(ax.get_xlim());
ax2.set_xlabel("Breakpoint number in current MCMC iteration")

# beta CI's weighted by breakpoints
for bp_samp in H.breakpoint_list:
    bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]
    for st, en in bpl:
        ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], Ph.iloc[st:en, maj_idx].sum() + 1, Ph.iloc[st:en, min_idx].sum() + 1)
        ax.add_patch(mpl.patches.Rectangle((Ph.iloc[st, 1], ci_lo), Ph.iloc[en, 1] - Ph.iloc[st, 1], ci_hi - ci_lo, fill = True, facecolor = 'k', alpha = 0.01, zorder = 1000))

ax.set_xticks(np.linspace(*plt.xlim(), 20));
ax.set_xticklabels(Ph["pos"].searchsorted(np.linspace(*plt.xlim(), 20)));
ax.set_xlabel("SNP index")

for i in range(0, 100):
    seg_A = P.iloc[bdy[0, 0]:bdy[0, 1], min_idx].sum()
    seg_B = P.iloc[bdy[0, 0]:bdy[0, 1], maj_idx].sum()

    breakpoints = []

    for b in bdy[1:]:
        Brv = s.beta.rvs(seg_A + 1, seg_B + 1, size = 1000)

        A = P.iloc[b[0]:b[1], min_idx].sum()
        B = P.iloc[b[0]:b[1], maj_idx].sum()

        Brv_cur = s.beta.rvs(A + 1, B + 1, size = 1000)
        p_gt = (Brv_cur > Brv).mean()
        prob_same = np.maximum(np.minimum(np.min(2*np.c_[p_gt, 1 - p_gt], 1), 1.0 - np.finfo(float).eps), np.finfo(float).eps)[0]

        if np.random.rand() < prob_same:
            seg_A += A
            seg_B += B
        else:
            seg_A = A
            seg_B = B
            breakpoints.append(b[0])

    bdy = np.r_[np.c_[0, breakpoints[0]], np.c_[np.r_[breakpoints[:-1]], np.r_[breakpoints[1:]]], np.c_[breakpoints[-1], MAX_SNP_IDX]]

    # compute marginal likelihood/posterior numerator
    post_num = 0
    marglik = 0
    for b in bdy:
        A = P.iloc[b[0]:b[1], min_idx].sum() + 1
        B = P.iloc[b[0]:b[1], maj_idx].sum() + 1

        f_hat = A/(A + B)
        post_num += (A - 1)*np.log(f_hat) + (B - 1)*np.log(1 - f_hat)
        marglik += ss.betaln(A, B)
    #print(marglik, post_num - marglik, post_num)

plt.figure(4); plt.clf()
Ph = P.iloc[:MAX_SNP_IDX]

for i in breakpoints:
    plt.axvline(Ph.iloc[i, P.columns.get_loc("pos")], color = 'k', alpha = 0.2)

plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.75, color = cmap[aidx.astype(np.int)])
#plt.xlim([0, 36e6])
plt.xticks(np.linspace(*plt.xlim(), 20), P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)))
plt.xlabel("SNP index")

#
# compute beta distribution overlaps {{{

B = s.beta.rvs(P["MIN_COUNT"][:, None] + 1, P["MAJ_COUNT"][:, None] + 1, size=(len(P), 1000))

p_gt = (B[:-1] - B[1:] > 0).mean(1)
P_next = np.maximum(np.minimum(np.min(2*np.c_[p_gt, 1 - p_gt], 1), 1.0 - np.finfo(float).eps), np.finfo(float).eps)

P["lP_next"] = np.nan
P["lP_next_s"] = np.nan
P["lP_prev"] = np.nan
P["lP_prev_s"] = np.nan

pc_idx = P.columns.get_loc("lP_prev")
pcs_idx = P.columns.get_loc("lP_prev_s")
nc_idx = P.columns.get_loc("lP_next")
ncs_idx = P.columns.get_loc("lP_next_s")

P.iloc[:-1, nc_idx] = np.nan
P.iloc[:-1, nc_idx] = np.log(P_next)
P.iloc[:-1, ncs_idx] = np.log(1 - P_next)
P.iloc[1:, pc_idx] = np.log(1 - P_next)
P.iloc[1:, pcs_idx] = np.log(P_next)

#
# visualize switch prob

plt.figure(5); plt.clf()
plt.errorbar(P.index, y = P["median_hap"], yerr = np.c_[P["median_hap"] - P["CI_lo_hap"], P["CI_hi_hap"] - P["median_hap"]].T, fmt = 'none', alpha = 0.75)
plt.scatter(P.index[:-1], 0.1*P.iloc[:-1, nc_idx])
plt.xlim([0, 300])
plt.ylim([-1, 1])
#plt.xticks(np.linspace(*plt.xlim(), 20), P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)))
plt.xlabel("SNP index")

#
# compute fill matrix

# XXX: to be fast, could do with integers bindigo style

FA = -np.inf*np.ones((1000, 1000))
#FA = np.zeros((1000, 1000))

#FA[0, 1] = PA.iat[1, pc_idx]
#FA[1, 0] = 1 - PA.iat[1, pc_idx]

FA[0, 0] = 0
FA[0, 1:1000] = np.cumsum(P.iloc[0:999, nc_idx])
#FA[1, 1] = P.iat[0, ncs_idx]
#FA[0:1000, 0] = PA.iloc[0:1000, pcs_idx]

O = -np.inf*np.ones((1000, 1000))
O[0, :] = 0

for i in range(1, 300):
    n_last = 0
    for j in range(i, 300):
        p_sw = FA[i - 1, j - 1] + P.iat[j - 1, ncs_idx]
        p_st = FA[i, j - 1] + P.iat[j - 1, nc_idx]

        FA[i, j] = np.maximum(p_sw, p_st)
        if p_sw > p_st:
            n_last = 0
        else:
            n_last += 1

        O[i, j] = n_last

plt.figure(1); plt.clf()
plt.imshow(FA[:300, :300])
plt.figure(11); plt.clf()
plt.imshow(O[:300, :300], cmap = "binary")

plt.figure(2); plt.clf()
plt.step(np.r_[0:299], np.exp(FA[:299, 299]))

# traceback

i_t = FA[:299, 299].argmax()
j_t = 299

stay = np.zeros(299)
for i in range(0, 299):
    f = np.argmax(np.r_[FA[i_t - 1, j_t - 1], FA[i_t, j_t - 1]])
    stay[298 - i] = f

    j_t -= 1
    if f == 0:
        i_t -= 1

plt.figure(5); plt.clf()
Ps = P.iloc[0:300]
plt.errorbar(Ps.index, y = Ps["median_hap"], yerr = np.c_[Ps["CI_hi_hap"] - Ps["median_hap"], Ps["median_hap"] - Ps["CI_lo_hap"]].T, fmt = 'none', alpha = 0.75)
plt.scatter(Ps.index, Ps["median_hap"], c = np.r_[0, Ps["lP_next"].iloc[:-1]] > np.log(0.5), s = 100)
plt.scatter(Ps.index, Ps["median_hap"], c = np.r_[1, stay], s = 32)


plt.hlines(y = stay, xmin = np.flatnonzero(stay)

# }}}

#
# beta-binomial likelihood of segment extension {{{

# sum matrices

# MAJOR
#M = np.zeros((100, 100))
mcidx = P.columns.get_loc("MAJ_COUNT")
ncidx = P.columns.get_loc("MIN_COUNT")

M = np.diag(P["MAJ_COUNT"].iloc[:1000])

for i in range(0, 1000):
    for j in range(i + 1, 1000):
        M[i, j] = M[i, j - 1] + P.iat[j, mcidx]

plt.figure(1); plt.clf()
plt.imshow(M)

# MINOR
N = np.diag(P["MIN_COUNT"].iloc[:1000])

for i in range(0, 1000):
    for j in range(i + 1, 1000):
        N[i, j] = N[i, j - 1] + P.iat[j, ncidx]

# beta binomial probabilities

# backward
BB_b = np.zeros((1000, 1000))
for i in range(1, 300):
    for j in range(i + 1, 300):
        BB_b[i, j] = s.betabinom.pmf(
          k = P.iat[j, ncidx],
          n = P.iloc[j, [ncidx, mcidx]].sum(),
          a = N[i, j - 1] + 30,
          b = M[i, j - 1] + 30,
        )

# forward
BB_f = np.zeros((1000, 1000))
for i in range(1, 300):
    for j in range(i + 1, 300):
        BB_f[i, j] = s.betabinom.pmf(
          k = P.iat[i, ncidx],
          n = P.iloc[i, [ncidx, mcidx]].sum(),
          a = N[i + 1, j] + 30,
          b = M[i + 1, j] + 30,
        )

# plot BB matrices

# backwards
plt.figure(10); plt.clf()
plt.imshow(BB_b[0:300, 0:300])#, origin = "lower")
plt.ylabel("Segment start position")
plt.xlabel("Segment end position")

# forwards
plt.figure(105); plt.clf()
plt.imshow(BB_f[0:300, 0:300])#, origin = "lower")
plt.ylabel("Segment start position")
plt.xlabel("Segment end position")

# joint
plt.figure(106); plt.clf()
plt.imshow(np.log(BB_f[0:300, 0:300]) + np.log(BB_b[0:300, 0:300]))#, origin = "lower")
plt.ylabel("Segment start position")
plt.xlabel("Segment end position")

# plot backward BB probs for segment ending at 300
plt.figure(12); plt.clf()
plt.step(np.r_[0:300], BB_b[:300, 299])
plt.step(np.r_[0:300], BB_f[:300, 299])
plt.step(np.r_[0:300], BB_f[1, 0:300])

# traceback
midx_b = [299]
while True:
    next_max = BB_b[:300, midx_b[-1]].argmax()
    midx_b.append(next_max)
    if next_max == 0:
        break

# traceforward
midx_f = [1]
while True:
    next_max = BB_f[midx_f[-1], :300].argmax()
    midx_f.append(next_max)
    if next_max == 299:
        break

# alternate joint

# plot changepoints overlayed on SNPs
plt.figure(11); plt.clf()
plt.errorbar(Ph.index, y = Ph["median_hap"], yerr = np.c_[Ph["CI_hi_hap"] - Ph["median_hap"], Ph["median_hap"] - Ph["CI_lo_hap"]].T, fmt = 'none', alpha = 0.75)

for i in midx_bj:
    plt.axvline(i, color = 'r', linestyle = (0, [1, 1]))
for i in midx_b:
    plt.axvline(i, color = 'b', linestyle = (1, [1, 1]))

plt.xlim([0, 300])

# }}}

plt.figure(1); plt.clf()
plt.imshow(FA[0:300, 0:300])

plt.figure(3);
plt.clf();
plt.step(np.r_[1:299], FA[1:299, 299])
plt.step(np.r_[1:299], PA["lP_prev"].iloc[1:299])

x, y = np.unravel_index(FA[0:200,0:200].argmax(), FA[0:200,0:200].shape)
FA[x, y]
