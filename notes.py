import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.special as ss

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

def adj(bdy1, bdy2, A1 = None, B1 = None, A2 = None, B2 = None):
    A1 = P.iloc[bdy1[0]:bdy1[1], min_idx].sum() if A1 is None else A1
    B1 = P.iloc[bdy1[0]:bdy1[1], maj_idx].sum() if B1 is None else B1
    brv1 = s.beta.rvs(A1 + 1, B1 + 1, size = 1000)

    A2 = P.iloc[bdy2[0]:bdy2[1], min_idx].sum() if A2 is None else A2
    B2 = P.iloc[bdy2[0]:bdy2[1], maj_idx].sum() if B2 is None else B2
    brv2 = s.beta.rvs(A2 + 1, B2 + 1, size = 1000)

    #
    # probability of segment similarity

    p_gt = (brv1 > brv2).mean()
    prob_same = np.maximum(np.minimum(np.min(2*np.c_[p_gt, 1 - p_gt], 1), 1.0 - np.finfo(float).eps), np.finfo(float).eps)[0]

    #
    # probability of phase switch

    # haps = x/y, segs = 1/2, beta params. = A/B

    # seg 1
    x1_A = P.loc[(P.index >= bdy1[0]) & (P.index < bdy1[1]) & P["aidx"], "ALT_COUNT"].sum() + 1
    x1_B = P.loc[(P.index >= bdy1[0]) & (P.index < bdy1[1]) & P["aidx"], "REF_COUNT"].sum() + 1 

    y1_A = P.loc[(P.index >= bdy1[0]) & (P.index < bdy1[1]) & ~P["aidx"], "ALT_COUNT"].sum() + 1 
    y1_B = P.loc[(P.index >= bdy1[0]) & (P.index < bdy1[1]) & ~P["aidx"], "REF_COUNT"].sum() + 1 

    # seg 2
    x2_A = P.loc[(P.index >= bdy2[0]) & (P.index < bdy2[1]) & P["aidx"], "ALT_COUNT"].sum() + 1 
    x2_B = P.loc[(P.index >= bdy2[0]) & (P.index < bdy2[1]) & P["aidx"], "REF_COUNT"].sum() + 1 

    y2_A = P.loc[(P.index >= bdy2[0]) & (P.index < bdy2[1]) & ~P["aidx"], "ALT_COUNT"].sum() + 1 
    y2_B = P.loc[(P.index >= bdy2[0]) & (P.index < bdy2[1]) & ~P["aidx"], "REF_COUNT"].sum() + 1 

    lik_mis   = ss.betaln(x1_A + y1_B + y2_A + x2_B, y1_A + x1_B + x2_A + y2_B)
    lik_nomis = ss.betaln(x1_A + y1_B + x2_A + y2_B, y1_A + x1_B + y2_A + x2_B)

    # TODO: this could be a function of the actual SNP phasing 
    p_mis = 0.001

    # logsumexp
    m = np.maximum(lik_mis, lik_nomis)
    denom = m + np.log(np.exp(lik_mis - m)*p_mis + np.exp(lik_nomis - m)*(1 - p_mis))

    prob_misphase = np.exp(lik_mis + np.log(p_mis) - denom)

    return prob_same, prob_misphase

MAX_SNP_IDX = 1001
bdy = np.c_[np.r_[0:(MAX_SNP_IDX - 1)], np.r_[1:MAX_SNP_IDX]]

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
