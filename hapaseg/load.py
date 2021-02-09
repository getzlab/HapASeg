import numpy as np
import pandas as pd

from capy import txt
from itertools import zip_longest

_chrmap = dict(zip(["chr" + str(x) for x in list(range(1, 23)) + ["X", "Y"]], range(1, 25)))

class HapasegReference:
    def __init__(self,
      phased_VCF = "test.vcf",
      allele_counts = "3328.tumor.tsv",
      cytoband_file = "cytoBand.txt"
    ):
        #
        # load in VCF
        self.allele_counts = self.load_VCF(phased_VCF, allele_counts)

        #
        # parse cytoband file for chromosome arm boundaries
        self.chromosome_intervals = self.parse_cytoband(cytoband_file)

    @staticmethod
    def load_VCF(VCF, allele_counts):
        P = pd.read_csv(phased_VCF, sep = "\t", comment = "#", names = ["chr", "pos", "x", "ref", "alt", "y", "z", "a", "b", "hap"], header = None)
        P = P.loc[:, ~P.columns.str.match('^.$')]

        P = txt.parsein(P, 'hap', r'(.)\|(.)', ["allele_A", "allele_B"]).astype({"allele_A" : int, "allele_B" : int })

        # pull in altcounts
        C = pd.read_csv(allele_counts, sep = "\t")
        P = P.merge(C, how = "inner", left_on = ["chr", "pos"], right_on = ["CONTIG", "POSITION"]).drop(columns = ["CONTIG", "POSITION"])

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
    def parse_cytoband(cytoband):
        cband = pd.read_csv(cytoband_file, sep = "\t", names = ["chr", "start", "end", "band", "stain"])
        cband["chr"] = cband["chr"].apply(lambda x : _chrmap[x])

        chrs = cband["chr"].unique()
        ints = dict(zip(chrs, [{0} for _ in range(0, len(chrs))]))
        last_end = None
        last_stain = None
        for _, chrom, start, end, _, stain in cband.itertuples():
            if start == 0:
                if last_end is not None:
                    ints[chrom - 1].add(last_end)
            if stain == "acen" and last_stain is not None:
                ints[chrom].add(start)
            
            last_end = end
            last_stain = stain
        ints[chrom].add(end)

        CI = np.full([len(ints), 4], 0)
        for c in chrs:
            CI[c - 1, :] = sorted(ints[c])
