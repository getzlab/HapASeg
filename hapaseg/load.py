import numpy as np
import pandas as pd
import scipy.stats as s

from capy import txt, df, mut
from itertools import zip_longest

_chrmap = dict(zip(["chr" + str(x) for x in list(range(1, 23)) + ["X", "Y"]], range(1, 25)))

class HapasegReference:
    def __init__(self,
      phased_VCF = "test.vcf",
      readbacked_phased_VCF = None,
      allele_counts = "3328.tumor.tsv",
      allele_counts_N = None,
      coverage = None,
      cytoband_file = "cytoBand.txt"
    ):
        #
        # load in VCF
        self.allele_counts = self.load_VCF(phased_VCF, allele_counts, allele_counts_N, readbacked_phased_VCF)

        #
        # load in coverage; merge with allele counts

        #
        # parse cytoband file for chromosome arm boundaries
        self.chromosome_intervals = self.parse_cytoband(cytoband_file)

    @staticmethod
    def load_VCF(VCF, allele_counts, allele_counts_N = None, RBP_VCF = None):
        P = pd.read_csv(VCF, sep = "\t", comment = "#", names = ["chr", "pos", "x", "ref", "alt", "y", "z", "a", "b", "hap"], header = None)
        P = P.loc[:, ~P.columns.str.match('^.$')]
        P["chr"] = mut.convert_chr(P["chr"])

        P = txt.parsein(P, 'hap', r'(.)[/|](.)', ["allele_A", "allele_B"]).astype({"allele_A" : int, "allele_B" : int })
        P["phased"] = P["hap"].str[1] == "|"

        # XXX: one day, we may support loading VCFs with misphasing probabilities
        #      already given. but today is not that day. this is just a placeholder
        #      for values that may be filled in from read-backed phasing.
        P["misphase_prob"] = np.nan

        #
        # merge with read-backed phasing VCF, if given
        if RBP_VCF is not None:
            P = HapasegReference.load_RBP_VCF(RBP_VCF, P)

        #
        # pull in altcounts
        C = pd.read_csv(allele_counts, sep = "\t")
        P = P.merge(C, how = "inner", left_on = ["chr", "pos"], right_on = ["CONTIG", "POSITION"]).drop(columns = ["CONTIG", "POSITION"])

        if allele_counts_N is not None:
            C = pd.read_csv(allele_counts_N, sep = "\t").rename(columns = lambda x : x + "_N")
            P = P.merge(C, how = "inner", left_on = ["chr", "pos"], right_on = ["CONTIG_N", "POSITION_N"]).drop(columns = ["CONTIG_N", "POSITION_N"])

        # for now we discard all unphased sites. later we might want to use them
        # if they provide addition power
        P = P.loc[P["phased"]]

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

        return P

    @staticmethod
    def load_RBP_VCF(RBP_VCF, P):
        P_RBP = pd.read_csv(RBP_VCF, sep = "\t", comment = "#", names = ["chr", "pos", "x", "ref", "alt", "y", "z", "a", "b", "hap"], header = None)
        P_RBP = P_RBP.loc[:, ~P_RBP.columns.str.match('^.$')]

        P_RBP = txt.parsein(P_RBP, 'hap', r'(.)[/|](.)(?::(\d+))?', ["allele_A", "allele_B", "PS"]).astype({"allele_A" : int, "allele_B" : int })
        P_RBP = P_RBP.astype({"allele_A" : pd.Int8Dtype(), "allele_B" : pd.Int8Dtype() })
        P_RBP["phased"] = P_RBP["hap"].str[1] == "|"
        P_RBP = P_RBP.loc[P_RBP["phased"]].astype({ "PS" : int }).astype({ "PS" : pd.Int32Dtype() })

        # compare RBP phasing orientation to imputed orientation within each RBP phase sets
        # set imputed orientation to RBP orientation with max Jaccard similarity
        P_RBP = P_RBP.merge(P.loc[:, ["chr", "pos", "allele_A", "allele_B"]].astype({"allele_A" : pd.Int8Dtype(), "allele_B" : pd.Int8Dtype() }), left_on = ["chr", "pos"], right_on = ["chr", "pos"], suffixes = ("_RBP", "_IMP"), how = "left")
        P_RBP["misphase_prob"] = np.nan

        PS_idx = P_RBP.groupby("PS").indices

        for rng in PS_idx.values():
            PS = P_RBP.iloc[rng]
            RBP_f = PS.loc[:, ["allele_A_RBP", "allele_B_RBP"]].values
            RBP_r = PS.loc[:, ["allele_B_RBP", "allele_A_RBP"]].values
            IMP = PS.loc[:, ["allele_A_IMP", "allele_B_IMP"]].values

            # sometimes we can RBP phase alleles that we couldn't impute; in that case,
            # the imputed values will be pd.na. ignore these in the Jaccard comparison
            nidx = ~pd.isnull(IMP)

            if (RBP_f[nidx] & IMP[nidx]).sum()/(RBP_f[nidx] | IMP[nidx]).sum() < \
               (RBP_r[nidx] & IMP[nidx]).sum()/(RBP_r[nidx] | IMP[nidx]).sum():
                P_RBP.loc[rng, ["allele_A_IMP", "allele_B_IMP"]] = RBP_r
            else:
                P_RBP.loc[rng, ["allele_A_IMP", "allele_B_IMP"]] = RBP_f

            # this SNP is correctly phased relative to the downstream SNP
            P_RBP.loc[rng[:-1], "misphase_prob"] = 0

        P_RBP = P_RBP.rename(columns = { "allele_A_IMP" : "allele_A", "allele_B_IMP" : "allele_B" })

        # update imputed phasing to reflect RBF
        mm = df.multimap(P.loc[:, ["chr", "pos"]], P_RBP.loc[:, ["chr", "pos"]]).dropna()
        P.loc[mm.index, ["allele_A", "allele_B", "misphase_prob"]] = P_RBP.loc[mm, ["allele_A", "allele_B", "misphase_prob"]].values

        # add sites recovered by RBF
        mm = df.multimap(P_RBP.loc[:, ["chr", "pos"]], P.loc[:, ["chr", "pos"]])
        return pd.concat([P, P_RBP.loc[mm[mm.isna()].index, P.columns]], ignore_index = True).sort_values(["chr", "pos"])

    @staticmethod
    def parse_cytoband(cytoband):
        cband = pd.read_csv(cytoband, sep = "\t", names = ["chr", "start", "end", "band", "stain"])
        cband["chr"] = cband["chr"].apply(lambda x : _chrmap[x])

        chrs = cband["chr"].unique()
        ints = dict(zip(chrs, [{0} for _ in range(0, len(chrs))]))
        last_end = None
        last_stain = None
        last_chrom = None
        for _, chrom, start, end, _, stain in cband.itertuples():
            if start == 0:
                if last_end is not None:
                    ints[last_chrom].add(last_end)
            if stain == "acen" and last_stain != "acen":
                ints[chrom].add(start)
            if stain != "acen" and last_stain == "acen":
                ints[chrom].add(start)
            
            last_end = end
            last_stain = stain
            last_chrom = chrom
        ints[chrom].add(end)

        CI = np.full([len(ints), 4], 0)
        for c in chrs:
            CI[c - 1, :] = sorted(ints[c])

        return pd.DataFrame(
          np.c_[np.tile(np.c_[np.r_[1:25]], [1, 2]).reshape(-1, 1), CI.reshape(-1, 2)],
          columns = ["chr", "start", "end"]
        )
