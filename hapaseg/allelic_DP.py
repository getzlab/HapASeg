import colorama
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

from capy import seq

class A_DP:
    def __init__(self, allelic_segs_pickle):
        self.allelic_segs = pd.read_pickle(allelic_segs_pickle)
        self.n_samp = len(self.allelic_segs.iloc[0]["results"].breakpoint_list)

    def load_samp(self, samp_idx):
        if samp_idx > self.n_samp:
            raise ValueError(f"Only {self.n_samp} MCMC samples were taken!")

        all_segs = []
        all_SNPs = []

        maj_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("MAJ_COUNT")
        min_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("MIN_COUNT")

        alt_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("ALT_COUNT")
        ref_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("REF_COUNT")

        chunk_offset = 0
        for _, H in self.allelic_segs.dropna(subset = ["results"]).iterrows():
            r = copy.deepcopy(H["results"])

            # set phasing orientation back to original
            for st, en in r.F.intervals():
                # code excised from flip_hap
                x = r.P.iloc[st:en, maj_idx].copy()
                r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
                r.P.iloc[st:en, min_idx] = x

            # save SNPs for this chunk
            all_SNPs.append(pd.DataFrame({ "maj" : r.P["MAJ_COUNT"], "min" : r.P["MIN_COUNT"], "gpos" : seq.chrpos2gpos(r.P.loc[0, "chr"], r.P["pos"]) }))

            # draw breakpoint, phasing, and SNP inclusion sample from segmentation MCMC trace
            bp_samp, pi_samp, inc_samp = (r.breakpoint_list[samp_idx], r.phase_interval_list[samp_idx] if r.phase_correct else None, r.include[samp_idx])
            # flip everything according to sample
            if r.phase_correct:
                for st, en in pi_samp.intervals():
                    x = r.P.iloc[st:en, maj_idx].copy()
                    r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
                    r.P.iloc[st:en, min_idx] = x

            bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]

            # get major/minor sums for each segment
            # also get {alt, ref} x {aidx, bidx}
            for st, en in bpl:
                all_segs.append([
                  st + chunk_offset, en + chunk_offset,                        # SNP index for seg
                  r.P.loc[st, "chr"], r.P.loc[st, "pos"], r.P.loc[en, "pos"],  # chromosomal position of seg
                  r._Piloc(st, en, min_idx, inc_samp).sum(),                   # min/maj counts
                  r._Piloc(st, en, maj_idx, inc_samp).sum(),

                  r._Piloc(st, en, alt_idx, inc_samp & r.P["aidx"]).sum(),     # allele A alt/ref
                  r._Piloc(st, en, ref_idx, inc_samp & r.P["aidx"]).sum(),
                  r._Piloc(st, en, alt_idx, inc_samp & ~r.P["aidx"]).sum(),    # allele B alt/ref
                  r._Piloc(st, en, ref_idx, inc_samp & ~r.P["aidx"]).sum()
                ])

            chunk_offset += len(r.P)

        # convert samples into dataframe
        S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj", "A_alt", "A_ref", "B_alt", "B_ref"])

        # convert chr-relative positions to absolute genomic coordinates
        S["start_gp"] = seq.chrpos2gpos(S["chr"], S["start"])
        S["end_gp"] = seq.chrpos2gpos(S["chr"], S["end"])

        # initial cluster assignments
        S["clust"] = -1 # initially, all segments are unassigned
        S.iloc[0, S.columns.get_loc("clust")] = 0 # first segment is assigned to cluster 0

        # initial phasing orientation
        S["flipped"] = False

        return S, pd.concat(all_SNPs, ignore_index = True)
