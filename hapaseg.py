import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc

class Hapaseg:
    def __init__(self, P):
        #
        # dataframe stuff
        self.P = P.copy() 

        self.min_idx = P.columns.get_loc("MIN_COUNT")
        self.maj_idx = P.columns.get_loc("MAJ_COUNT")
        self.alt_idx = P.columns.get_loc("ALT_COUNT")
        self.ref_idx = P.columns.get_loc("REF_COUNT")

        # number of times we've correct the phasing of a segment
        self.P["flip"] = 0

        #
        # config stuff

        # highest SNP to analyze
        self.MAX_SNP_IDX = 5001
        self.N_INITIAL_PASSES = 10

        #
        # breakpoint storage

        # breakpoints of last iteration
        self.breakpoints = sc.SortedSet(range(0, self.MAX_SNP_IDX))

        # count of all breakpoints ever created
        self.breakpoint_counter = sc.SortedDict(itertools.zip_longest(range(0, self.MAX_SNP_IDX), [0], fillvalue = 0))

        #
        # cumsum arrays for each segment

        # will be populated by compute_cumsums()

        self.cs_MAJ = sc.SortedDict()
        self.cs_MIN = sc.SortedDict()

        # probability of picking a breakpoint
        self.split_prob = sc.SortedDict()

        #
        # marginal likelihoods

        # log marginal likelihoods for each segment
        self.seg_marg_liks = sc.SortedDict(zip(
          range(0, self.MAX_SNP_IDX),
          ss.betaln(
            P.iloc[0:self.MAX_SNP_IDX, self.min_idx] + 1,
            P.iloc[0:self.MAX_SNP_IDX, self.maj_idx] + 1
          )
        ))

        # total log marginal likelihood of all segments
        self.marg_lik = np.array(self.seg_marg_liks.values()).sum()

    def combine(self, st = None, b_idx = None, force = True):
        """
        Probabilistically combine segment starting at `st` with the subsequent segment.
        Returns `st` if segments are combined, breakpoint if segments aren't combined,
        and -1 if `st` is invalid (past the end).

        If force is True, then we will unconditionally accept the transition.
        If force is False, we will probabilistically accept based on the marginal
        likelihood ratio (Metropolis-like sampling)
        """

        if st is not None:
            st = self.breakpoints[self.breakpoints.bisect_left(st)]
        elif b_idx is not None:
            st = self.breakpoints[b_idx]
        else:
            raise ValueError("You must either specify nearest start coordinate or breakpoint index!")
        br = self.breakpoints.bisect_right(st)

        # we're trying to combine past the last segment 
        if br + 1 >= len(self.breakpoints):
            return -1

        mid = self.breakpoints[br]
        en = self.breakpoints[br + 1]

        prob_same, prob_same_mis, prob_misphase = self.t_probs(np.r_[st, mid], np.r_[mid, en])

        trans = np.random.choice(np.r_[0:4],
          p = np.r_[
            prob_same*(1 - prob_misphase),         # 0: extend seg, phase is correct
            prob_same_mis*prob_misphase,           # 1: extend seg, phase is wrong
            (1 - prob_same)*(1 - prob_misphase),   # 2: new segment, phase is correct
            (1 - prob_same_mis)*prob_misphase,     # 3: new segment, phase is wrong
          ]
        )

        # flip phase
        flipped = False
        if trans == 1 or trans == 3:
            self.flip_hap(mid, en)
            flipped = True

        # extend segment
        if trans <= 1:
            prev_marg_lik = self.marg_lik
            self.marg_lik -= self.seg_marg_liks[st]
            self.marg_lik -= self.seg_marg_liks[mid]
            seg_lik = ss.betaln(
              self.P.loc[st:(en - 1), "MIN_COUNT"].sum() + 1,
              self.P.loc[st:(en - 1), "MAJ_COUNT"].sum() + 1
            )
            self.marg_lik += seg_lik

            # accept transition
            if force or np.log(np.random.rand()) < np.minimum(0, self.marg_lik - prev_marg_lik):
                self.breakpoints.remove(mid)
                self.seg_marg_liks.__delitem__(mid)
                self.seg_marg_liks[st] = seg_lik

                return st

            # don't accept transition; undo
            else:
                if flipped:
                    self.flip_hap(mid, en)
                self.marg_lik = prev_marg_lik

                return mid

        # don't extend segment. marginal likelihood won't change, since phasing
        # doesn't affect segment likelihoods
        # TODO: should it?
        else:
            return mid
    
    def flip_hap(self, st, en):
        """
        flips the haplotype of sites from st to en - 1 (Pythonic half indexing)
        """
        # note that loc indexing is closed, so we explicitly have to give it en - 1
        x = self.P.loc[st:(en - 1), "MAJ_COUNT"].copy()
        self.P.loc[st:(en - 1), "MAJ_COUNT"] = self.P.loc[st:(en - 1), "MIN_COUNT"]
        self.P.loc[st:(en - 1), "MIN_COUNT"] = x
        self.P.loc[st:(en - 1), "aidx"] = ~self.P.loc[st:(en - 1), "aidx"]
        self.P.loc[st:(en - 1), "flip"] += 1

    def t_probs(self, bdy1, bdy2, A1 = None, B1 = None, A2 = None, B2 = None):
        """
        Compute transition probabilities for segments bounded by bdy1 and bdy2
        """
        A1 = self.P.iloc[bdy1[0]:bdy1[1], self.min_idx].sum() if A1 is None else A1
        B1 = self.P.iloc[bdy1[0]:bdy1[1], self.maj_idx].sum() if B1 is None else B1
        brv1 = s.beta.rvs(A1 + 1, B1 + 1, size = 1000)

        A2 = self.P.iloc[bdy2[0]:bdy2[1], self.min_idx].sum() if A2 is None else A2
        B2 = self.P.iloc[bdy2[0]:bdy2[1], self.maj_idx].sum() if B2 is None else B2
        brv2 = s.beta.rvs(A2 + 1, B2 + 1, size = 1000)
 
        # if second segment was misphased
        brv3 = s.beta.rvs(B2 + 1, A2 + 1, size = 1000)

        #
        # probability of segment similarity

        # correct phasing
        p_gt = (brv1 > brv2).mean()
        prob_same = np.maximum(np.minimum(np.min(2*np.c_[p_gt, 1 - p_gt], 1), 1.0 - np.finfo(float).eps), np.finfo(float).eps)[0]

        # misphasing
        p_gt = (brv1 > brv3).mean()
        prob_same_mis = np.maximum(np.minimum(np.min(2*np.c_[p_gt, 1 - p_gt], 1), 1.0 - np.finfo(float).eps), np.finfo(float).eps)[0]

        #
        # probability of phase switch

        # haps = x/y, segs = 1/2, beta params. = A/B

        # seg 1
        x1_A = self.P.loc[(self.P.index >= bdy1[0]) & (self.P.index < bdy1[1]) & self.P["aidx"], "ALT_COUNT"].sum() + 1
        x1_B = self.P.loc[(self.P.index >= bdy1[0]) & (self.P.index < bdy1[1]) & self.P["aidx"], "REF_COUNT"].sum() + 1 

        y1_A = self.P.loc[(self.P.index >= bdy1[0]) & (self.P.index < bdy1[1]) & ~self.P["aidx"], "ALT_COUNT"].sum() + 1 
        y1_B = self.P.loc[(self.P.index >= bdy1[0]) & (self.P.index < bdy1[1]) & ~self.P["aidx"], "REF_COUNT"].sum() + 1 

        # seg 2
        x2_A = self.P.loc[(self.P.index >= bdy2[0]) & (self.P.index < bdy2[1]) & self.P["aidx"], "ALT_COUNT"].sum() + 1 
        x2_B = self.P.loc[(self.P.index >= bdy2[0]) & (self.P.index < bdy2[1]) & self.P["aidx"], "REF_COUNT"].sum() + 1 

        y2_A = self.P.loc[(self.P.index >= bdy2[0]) & (self.P.index < bdy2[1]) & ~self.P["aidx"], "ALT_COUNT"].sum() + 1 
        y2_B = self.P.loc[(self.P.index >= bdy2[0]) & (self.P.index < bdy2[1]) & ~self.P["aidx"], "REF_COUNT"].sum() + 1 

        lik_mis   = ss.betaln(x1_A + y1_B + y2_A + x2_B, y1_A + x1_B + x2_A + y2_B)
        lik_nomis = ss.betaln(x1_A + y1_B + x2_A + y2_B, y1_A + x1_B + y2_A + x2_B)

        # TODO: this could be a function of the actual SNP phasing 
        # overall misphase prob. may also be returned from EAGLE
        p_mis = 0.001

        # logsumexp
        m = np.maximum(lik_mis, lik_nomis)
        denom = m + np.log(np.exp(lik_mis - m)*p_mis + np.exp(lik_nomis - m)*(1 - p_mis))

        prob_misphase = np.exp(lik_mis + np.log(p_mis) - denom)

        return prob_same, prob_same_mis, prob_misphase

    def compute_all_cumsums(self):
        bpl = np.array(self.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]
        for st, en in bpl:
            self.cs_MAJ[st], self.cs_MIN[st], self.split_prob[st] = self.compute_cumsum(st, en)

    def compute_cumsum(self, st, en):
        # major
        cs_MAJ = np.zeros(en - st, dtype = np.int)
        cs_MAJ[0] = self.P.iat[st, self.maj_idx]
        for i in range(st + 1, en):
            cs_MAJ[i - st] = cs_MAJ[i - st - 1] + self.P.iat[i, self.maj_idx]
        # minor
        cs_MIN = np.zeros(en - st, dtype = np.int)
        cs_MIN[0] = self.P.iat[st, self.min_idx]
        for i in range(st + 1, en):
            cs_MIN[i - st] = cs_MIN[i - st - 1] + self.P.iat[i, self.min_idx]

        # marginal likelihoods
        ml = ss.betaln(cs_MAJ + 1, cs_MIN + 1) + ss.betaln(cs_MAJ[-1] - cs_MAJ + 1, cs_MIN[-1] - cs_MIN + 1)

        # logsumexp to get probabilities
        m = np.max(ml)
        split_prob = np.exp(ml - (m + np.log(np.exp(ml - m).sum())))

        return cs_MAJ, cs_MIN, split_prob
