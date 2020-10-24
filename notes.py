import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s

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

# note that implicitly drops homozygous sites, since they were never had their
# alt/refcounts computed. perhaps we should use these as coverage probes?

#
# alt/ref -> major/minor
aidx = P["allele_A"] > 0
bidx = P["allele_B"] > 0

P["MAJ_COUNT"] = pd.concat([P.loc[aidx, "ALT_COUNT"], P.loc[bidx, "REF_COUNT"]])
P["MIN_COUNT"] = pd.concat([P.loc[aidx, "REF_COUNT"], P.loc[bidx, "ALT_COUNT"]])

#
# compute beta CI's
CI = s.beta.ppf([0.05, 0.5, 0.95], P["ALT_COUNT"][:, None] + 1, P["REF_COUNT"][:, None] + 1)
P[["CI_lo", "median", "CI_hi"]] = CI

CI = s.beta.ppf([0.05, 0.5, 0.95], P["MAJ_COUNT"][:, None] + 1, P["MIN_COUNT"][:, None] + 1)
P[["CI_lo_hap", "median_hap", "CI_hi_hap"]] = CI

#
# visualize

# major/minor

plt.figure(4); plt.clf()
Ph = P.iloc[:2000]
plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["CI_hi_hap"] - Ph["median_hap"], Ph["median_hap"] - Ph["CI_lo_hap"]].T, fmt = 'none', alpha = 0.75)
plt.xlim([0, 5e6])
plt.xticks(np.linspace(*plt.xlim(), 20), P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)))
plt.xlabel("SNP index")

# alt/ref

PA = Ph.loc[aidx]
PB = Ph.loc[bidx]
plt.figure(2); plt.clf()
plt.errorbar(PA["pos"], y = PA["median"], yerr = np.c_[PA["CI_hi"] - PA["median"], PA["median"] - PA["CI_lo"]].T, fmt = 'none', alpha = 0.75)
plt.errorbar(PB["pos"], y = PB["median"], yerr = np.c_[PB["CI_hi"] - PB["median"], PB["median"] - PB["CI_lo"]].T, fmt = 'none', alpha = 0.75)
plt.xticks(np.linspace(*plt.xlim(), 20), P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)))
plt.xlabel("SNP index (A)")

#
# compute beta distribution overlaps {{{

# easiest to do diff of two betas, even though this maxes out at ~50%
# TODO: better overlap test

B = s.beta.rvs(P["ALT_COUNT"][:, None] + 1, P["REF_COUNT"][:, None] + 1, size=(len(P), 1000))

for idx in [aidx, bidx]: 
    Ba = B[idx, :]

    P_next = np.maximum(np.minimum(1 - (Ba[:-1] - Ba[1:] > 0).mean(1), 1.0 - np.finfo(float).eps), np.finfo(float).eps)

    Pa = P.loc[idx]

    Pa["lP_next"] = np.nan
    Pa["lP_next_s"] = np.nan
    Pa["lP_prev"] = np.nan
    Pa["lP_prev_s"] = np.nan

    pc_idx = Pa.columns.get_loc("lP_prev")
    pcs_idx = Pa.columns.get_loc("lP_prev_s")
    nc_idx = Pa.columns.get_loc("lP_next")
    ncs_idx = Pa.columns.get_loc("lP_next_s")

    Pa.iloc[:-1, nc_idx] = np.nan
    Pa.iloc[:-1, nc_idx] = np.log(P_next)
    Pa.iloc[:-1, ncs_idx] = np.log(1 - P_next)
    Pa.iloc[1:, pc_idx] = np.log(1 - P_next)
    Pa.iloc[1:, pcs_idx] = np.log(P_next)

    #
    # compute fill matrix

    # XXX: to be fast, could do with integers bindigo style

    #FA = -np.inf*np.ones((1000, 1000))
    FA = np.zeros((1000, 1000))

    #FA[0, 1] = PA.iat[1, pc_idx]
    #FA[1, 0] = 1 - PA.iat[1, pc_idx]

    #FA[0, 0:1000] = PA.iloc[0:1000, pc_idx]
    #FA[0:1000, 0] = PA.iloc[0:1000, pcs_idx]

    for i in range(1, 300):
        for j in range(i, 300):
            FA[i, j] = np.maximum(
              FA[i - 1, j - 1] + Pa.iat[j, pcs_idx],
              FA[i, j - 1] + Pa.iat[j, pc_idx]
            )

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
BB = np.zeros((1000, 1000))
for i in range(1, 300):
    for j in range(i + 1, 300):
        BB[i, j] = s.betabinom.pmf(
          k = P.iat[j, ncidx],
          n = P.iloc[j, [ncidx, mcidx]].sum(),
          a = N[i, j - 1] + 30,
          b = M[i, j - 1] + 30,
        )

# plot BB matrix

plt.figure(10); plt.clf()
plt.imshow(BB[0:300, 0:300])#, origin = "lower")
plt.ylabel("Segment start position")
plt.xlabel("Segment end position")
bdys = np.c_[midx[:-1], midx[1:]]

# plot backward BB probs for segment ending at 300
plt.figure(12); plt.clf()
plt.step(np.r_[0:300], BB[:300, 299])

# traceback
midx_b = [299]
while True:
    next_max = BB[:300, midx_b[-1]].argmax()
    midx_b.append(next_max)
    if next_max == 0:
        break

# traceforward
midx_f = [0]
while True:
    next_max = BB[:300, midx_f[-1]].argmax()
    midx_f.append(next_max)
    if next_max == :
        break

# joint?

# plot changepoints overlayed on SNPs
plt.figure(11); plt.clf()
plt.errorbar(Ph.index, y = Ph["median_hap"], yerr = np.c_[Ph["CI_hi_hap"] - Ph["median_hap"], Ph["median_hap"] - Ph["CI_lo_hap"]].T, fmt = 'none', alpha = 0.75)

for i in midx_b:
    plt.axvline(i, color = 'k', linestyle = ':')

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
