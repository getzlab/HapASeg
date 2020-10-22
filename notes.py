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
aidx = P["allele_A"] > 0
bidx = P["allele_B"] > 0

# pull in altcounts
C = pd.read_csv("3328.tumor.tsv", sep = "\t")
P = P.merge(C, how = "inner", left_on = ["chr", "pos"], right_on = ["CONTIG", "POSITION"]).drop(columns = ["CONTIG", "POSITION"])

# note that implicitly drops homozygous sites, since they were never had their
# alt/refcounts computed. perhaps we should use these as coverage probes?

#
# compute beta CI's
CI = s.beta.ppf([0.05, 0.5, 0.95], P["ALT_COUNT"][:, None] + 1, P["REF_COUNT"][:, None] + 1)
P[["CI_lo", "median", "CI_hi"]] = CI

#
# compute beta distribution overlaps

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

plt.figure(1); plt.clf()
plt.imshow(FA[0:300, 0:300])

plt.figure(3);
plt.clf();
plt.step(np.r_[1:299], FA[1:299, 299])
plt.step(np.r_[1:299], PA["lP_prev"].iloc[1:299])

x, y = np.unravel_index(FA[0:200,0:200].argmax(), FA[0:200,0:200].shape)
FA[x, y]

plt.figure(2); plt.clf()
plt.errorbar(PA["pos"], y = PA["median"], yerr = np.c_[PA["CI_hi"] - PA["median"], PA["median"] - PA["CI_lo"]].T, fmt = 'none', alpha = 0.75)
plt.errorbar(PB["pos"], y = PB["median"], yerr = np.c_[PB["CI_hi"] - PB["median"], PB["median"] - PB["CI_lo"]].T, fmt = 'none', alpha = 0.75)
plt.xticks(np.linspace(0, plt.xlim()[1], 20), P["pos"].searchsorted(np.linspace(5e5, 5e6, 20)))
plt.xlabel("SNP index")
