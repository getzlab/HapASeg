import colorama
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc

class A_MCMC:
    def __init__(self, P, quit_after_burnin = False, n_iter = 100000):
        #
        # dataframe stuff
        self.P = P.copy().reset_index()
        self.P["aidx_orig"] = self.P["aidx"]

        self.min_idx = P.columns.get_loc("MIN_COUNT")
        self.maj_idx = P.columns.get_loc("MAJ_COUNT")
        self.alt_idx = P.columns.get_loc("ALT_COUNT")
        self.ref_idx = P.columns.get_loc("REF_COUNT")

        # number of times we've correct the phasing of a segment
        self.P["flip"] = 0

        #
        # config stuff

        # number of MCMC iterations
        self.n_iter = n_iter

        self.quit_after_burnin = quit_after_burnin

        #
        # chain state
        self.iter = 0
        self.burned_in = False

        #
        # breakpoint storage

        # breakpoints of last iteration
        self.breakpoints = sc.SortedSet(range(0, len(self.P)))
        #self.breakpoints = sc.SortedSet({0, self.MAX_SNP_IDX - 1})

        # count of all breakpoints ever created
        # breakpoint -> (number of times confirmed, number of times sampled)
        self.breakpoint_counter = np.zeros((len(self.P), 2), dtype = np.int)

        # list of all breakpoints at nth iteration
        self.breakpoint_list = []

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
          range(0, len(self.P)),
          ss.betaln(
            P.iloc[0:len(self.P), self.min_idx] + 1,
            P.iloc[0:len(self.P), self.maj_idx] + 1
          )
        ))
#        self.seg_marg_liks = sc.SortedDict(zip(
#          [0, self.MAX_SNP_IDX - 1],
#          [ss.betaln(
#            P.iloc[0:self.MAX_SNP_IDX, self.min_idx].sum() + 1,
#            P.iloc[0:self.MAX_SNP_IDX, self.maj_idx].sum() + 1
#          ), 0]
#        ))

        # total log marginal likelihood of all segments
        self.marg_lik = np.full(self.n_iter, np.nan)
        self.marg_lik[0] = np.array(self.seg_marg_liks.values()).sum()

    def run(self):
        for _ in range(0, self.n_iter):
            # perform a split and combine operation
            self.combine(np.random.choice(self.breakpoints[:-1]), force = False)
            self.split(b_idx = np.random.choice(len(self.breakpoints)))

            # if we're only running up to burnin, bail
            if self.quit_after_burnin and self.burned_in:
                print(colorama.Fore.GREEN + "Burned in [{st},{en}] in {n} iterations. n_bp = {n_bp}, lik = {lik}".format(
                  st = self.P["index"].iloc[0],
                  en = self.P["index"].iloc[-1],
                  n = self.iter,
                  n_bp = len(self.breakpoints),
                  lik = self.marg_lik[self.iter]
                ) + colorama.Fore.RESET)
                break

            # save set of breakpoints if burned in 
            if self.burned_in and not self.iter % 100:
                self.breakpoint_list.append(self.breakpoints.copy())

            # print status
            if not self.iter % 100:
                print("{color}[{st},{en}]\t{n}/{tot}\tn_bp = {n_bp}\tlik = {lik}".format(
                  st = self.P["index"].iloc[0],
                  en = self.P["index"].iloc[-1],
                  n = self.iter,
                  tot = self.n_iter,
                  n_bp = len(self.breakpoints),
                  lik = self.marg_lik[self.iter],
                  color = colorama.Fore.YELLOW if not self.burned_in else colorama.Fore.RESET
                ))

    def incr(self):
        self.iter += 1
        # TODO: use a faster method of computing rolling average
        if self.iter > 500:
            if np.diff(self.marg_lik[(self.iter - 500):self.iter]).mean() < 0:
                self.burned_in = True

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

        prob_misphase = self.prob_misphase(np.r_[st, mid], np.r_[mid, en])

#        # log PMF of proposal distribution q(s1+2|s1,s2)
#        with np.errstate(divide = "ignore"):
#            log_trans_probs = np.r_[
#              log_prob_same + np.log(1 - prob_misphase),                     # 0: extend seg, phase is correct
#              log_prob_same_mis + np.log(prob_misphase),                     # 1: extend seg, phase is wrong
#              np.log(-np.expm1(log_prob_same)) + np.log(1 - prob_misphase),  # 2: new segment, phase is correct
#              np.log(-np.expm1(log_prob_same_mis)) + np.log(prob_misphase),  # 3: new segment, phase is wrong
#            ]
#        log_trans_probs = np.maximum(log_trans_probs, np.finfo(float).min)
#
#        # draw from proposal distribution q(s1+2|s1,s2)
#        trans = np.random.choice(np.r_[0:4], p = np.exp(log_trans_probs))

        # flip phase
        flipped = False
        if np.random.rand() < prob_misphase:
            self.flip_hap(mid, en)
            flipped = True

        # M-H acceptance
        ML_split = self.seg_marg_liks[st] + self.seg_marg_liks[mid]

        ML_join = ss.betaln(
          self.P.loc[st:(en - 1), "MIN_COUNT"].sum() + 1,
          self.P.loc[st:(en - 1), "MAJ_COUNT"].sum() + 1
        )

        # proposal dist. ratio
        _, _, split_probs = self.compute_cumsum(st, en)
        # q(split)/q(join) = p(picking mid as breakpoint)/
        #                    p(picking first segment)
        log_q_rat = np.log(split_probs[mid - st]) - -np.log(len(self.breakpoints))

        # accept transition
        if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split + log_q_rat):
            self.breakpoints.remove(mid)
            self.seg_marg_liks.__delitem__(mid)
            self.seg_marg_liks[st] = ML_join

            self.incr()
            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1] - ML_split + ML_join
            if self.burned_in:
                self.incr_bp_counter(st = st, en = en)

            return st

        # don't accept transition; undo
        else:
            if flipped:
                self.flip_hap(mid, en)
            if self.burned_in:
                self.incr_bp_counter(st = st, mid = mid, en = en)

            self.incr()
            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]

            return mid

#        # extend segment
#        if trans <= 1:
#            prev_marg_lik = self.marg_lik
#            self.marg_lik -= self.seg_marg_liks[st]
#            self.marg_lik -= self.seg_marg_liks[mid]
#            seg_lik = ss.betaln(
#              self.P.loc[st:(en - 1), "MIN_COUNT"].sum() + 1,
#              self.P.loc[st:(en - 1), "MAJ_COUNT"].sum() + 1
#            )
#            self.marg_lik += seg_lik
#
#            # factor for proposal distribution q(s1,s2|s1+2) [breakpoint probability]
#            log_q_rat = 0
#            if not force:
#                _, _, split_probs = self.compute_cumsum(st, en)
#                # q(split)/q(join) = p(picking mid as breakpoint)*p(breaking)/
#                #                    p(picking first segment)*p(joining with subsequent)
#                log_q_rat = np.log(split_probs[mid - st]) + log_trans_probs[trans | 2] - \
#                  (-np.log(len(self.breakpoints)) + log_trans_probs[trans])
#
#            # accept transition
#            if force or np.log(np.random.rand()) < np.minimum(0, self.marg_lik - prev_marg_lik + log_q_rat):
#                self.breakpoints.remove(mid)
#                self.breakpoint_counter[mid] += np.r_[0, 1]
#                self.seg_marg_liks.__delitem__(mid)
#                self.seg_marg_liks[st] = seg_lik
#
#                return st
#
#            # don't accept transition; undo
#            else:
#                if flipped:
#                    self.flip_hap(mid, en)
#                self.marg_lik = prev_marg_lik
#                self.breakpoint_counter[mid] += np.r_[1, 1]
#
#                return mid
#
#        # don't extend segment. marginal likelihood won't change, since phasing
#        # doesn't affect segment likelihoods
#        # TODO: should it?
#        else:
#            self.breakpoint_counter[mid] += np.r_[1, 1]
#            return mid

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

    def t_probs(self, bdy1, bdy2, A1_ = None, B1_ = None, A2_ = None, B2_ = None, n_samp = None):
        """
        Compute transition probabilities for segments bounded by bdy1 and bdy2
        """

        n_samp = self.n_samp if n_samp is None else n_samp

        A1 = self.P.iloc[bdy1[0]:bdy1[1], self.min_idx].sum() if A1_ is None else A1_
        B1 = self.P.iloc[bdy1[0]:bdy1[1], self.maj_idx].sum() if B1_ is None else B1_
        brv1 = s.beta.rvs(A1 + 1, B1 + 1, size = n_samp)

        A2 = self.P.iloc[bdy2[0]:bdy2[1], self.min_idx].sum() if A2_ is None else A2_
        B2 = self.P.iloc[bdy2[0]:bdy2[1], self.maj_idx].sum() if B2_ is None else B2_
        brv2 = s.beta.rvs(A2 + 1, B2 + 1, size = n_samp)
 
        # if second segment was misphased
        brv3 = s.beta.rvs(B2 + 1, A2 + 1, size = n_samp)

        #
        # probability of segment similarity

        # if phasing is correct
        p_gt = (brv1 > brv2).mean()

        # use Gaussian approximation of beta distribution if overlap is too small
        if p_gt < 5/n_samp or p_gt > (n_samp - 5)/n_samp: # TODO: is this a good criterion?
            m1 = A1/(A1 + B1); m2 = A2/(A2 + B2)
            s1 = A1*B1/(A1 + B1)**3; s2 = A2*B2/(A2 + B2)**3

            log_prob_same = np.min(np.log(2) + np.c_[
              s.norm.logcdf(0, m1 - m2, np.sqrt(s1 + s2)),
              s.norm.logcdf(0, m2 - m1, np.sqrt(s1 + s2))
            ])
        # otherwise, use MC
        else:
            log_prob_same = np.min(np.log(2*np.c_[p_gt, 1 - p_gt]))

        # if phasing is incorrect
        p_gt = (brv1 > brv3).mean()

        # Gaussian approx.
        if p_gt < 5/n_samp or p_gt > (n_samp - 5)/n_samp: # TODO: is this a good criterion?
            m1 = A1/(A1 + B1); m2 = B2/(A2 + B2)
            s1 = A1*B1/(A1 + B1)**3; s2 = A2*B2/(A2 + B2)**3

            log_prob_same_mis = np.min(np.log(2) + np.c_[
              s.norm.logcdf(0, m1 - m2, np.sqrt(s1 + s2)),
              s.norm.logcdf(0, m2 - m1, np.sqrt(s1 + s2))
            ])
        # MC
        else:
            log_prob_same_mis = np.min(np.log(2*np.c_[p_gt, 1 - p_gt]))

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

        return log_prob_same, log_prob_same_mis, prob_misphase

    def prob_misphase(self, bdy1, bdy2):
        """
        Compute probability of misphase
        """

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

        return np.exp(lik_mis + np.log(p_mis) - denom)

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

        # prior
        # TODO: allow user to specify
#        if len(ml) > 1:
#            ml[-1] += np.log(0.9)
#            ml[:-1] += np.log(0.1) - np.log(len(ml) - 1)

        # logsumexp to get probabilities
        m = np.max(ml)
        split_prob = np.exp(ml - (m + np.log(np.exp(ml - m).sum())))

        return cs_MAJ, cs_MIN, split_prob

    def split(self, st = None, b_idx = None):
        if st is not None:
            st = self.breakpoints[self.breakpoints.bisect_left(st)]
        elif b_idx is not None:
            st = self.breakpoints[b_idx]
        else:
            raise ValueError("You must either specify nearest start coordinate or breakpoint index!")
        br = self.breakpoints.bisect_right(st)

        # we're trying to split something past the last segment
        if br >= len(self.breakpoints):
            return -1

        en = self.breakpoints[br]

        # compute split probabilities for this segment
        # TODO: memoize; use global self.cs_MAJ/cs_MIN/split_probs
        _, _, split_probs = self.compute_cumsum(st, en)

        b = np.random.choice(np.r_[0:len(split_probs)], p = split_probs)
        mid = b + st + 1

        # chosen split point is the segment end, so we're not splitting this segment
        if b == len(split_probs) - 1:
            if self.burned_in:
                self.incr_bp_counter(st = st, en = en)
            return

#        # otherwise, compute transition probabilities for split
#        # we use same transition probabilities as segment combining, but we'll
#        # switch them later
#        log_prob_same, log_prob_same_mis, prob_misphase = self.t_probs(np.r_[st, mid], np.r_[mid, en])
#
#        # log PMF of proposal distribution q(s1+2|s1,s2)
#        with np.errstate(divide = "ignore"):
#            log_trans_probs = np.r_[
#              log_prob_same + np.log(1 - prob_misphase),                     # 0: extend seg, phase is correct
#              log_prob_same_mis + np.log(prob_misphase),                     # 1: extend seg, phase is wrong
#              np.log(-np.expm1(log_prob_same)) + np.log(1 - prob_misphase),  # 2: new segment, phase is correct
#              np.log(-np.expm1(log_prob_same_mis)) + np.log(prob_misphase),  # 3: new segment, phase is wrong
#            ]
#        log_trans_probs = np.maximum(log_trans_probs, np.finfo(float).min)
#
#        trans = np.random.choice(np.r_[0:4], p = np.exp(log_trans_probs))

        prob_misphase = self.prob_misphase(np.r_[st, mid], np.r_[mid, en])

        # flip phase
        flipped = False
        if np.random.rand() < prob_misphase:
            self.flip_hap(mid, en)
            flipped = True

        # M-H acceptance
        seg_lik_1 = ss.betaln(
          self.P.loc[st:(mid - 1), "MIN_COUNT"].sum() + 1,
          self.P.loc[st:(mid - 1), "MAJ_COUNT"].sum() + 1
        )
        seg_lik_2 = ss.betaln(
          self.P.loc[mid:(en - 1), "MIN_COUNT"].sum() + 1,
          self.P.loc[mid:(en - 1), "MAJ_COUNT"].sum() + 1
        )

        ML_split = seg_lik_1 + seg_lik_2
        ML_join = self.seg_marg_liks[st]

        # q(join)/q(split) = p(picking first segment)/
        #                    p(picking first + second segment)*p(picking breakpoint)
        log_q_rat = -np.log(len(self.breakpoints)) - \
          (-np.log(len(self.breakpoints) - 1) + np.log(split_probs[b]))

        # accept transition
        if np.log(np.random.rand()) < np.minimum(0, ML_split - ML_join + log_q_rat):
            self.breakpoints.add(mid)
            self.seg_marg_liks[st] = seg_lik_1
            self.seg_marg_liks[mid] = seg_lik_2

            self.incr()
            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1] + ML_split - ML_join
            if self.burned_in:
                self.incr_bp_counter(st = st, mid = mid, en = en)

        # don't accept
        else:
            if flipped:
                self.flip_hap(mid, en)
            if self.burned_in:
                self.incr_bp_counter(st = st, en = en)

            self.incr()
            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]

#        # split segment
#        if trans > 1:
##            # p({s1,s2})/p({s1+s2}) = p(new segment)/p(extend segment)
##
##            p_same_3_4, p_same_mis_3_4, p_misphase_3_4 = self.t_probs(np.r_[mid, en], np.r_[st, mid])
##            p_diff_3_4 = np.log((1 - p_same_3_4)*(1 - p_misphase_3_4) + (1 - p_same_mis_3_4)*p_misphase_3_4)
##
##            p_same_1_2, p_same_mis_1_2, p_misphase_1_2 = self.t_probs(np.r_[mid, en], np.r_[st, mid])
##            p_diff_1_2 = np.log((1 - p_same_1_2)*(1 - p_misphase_1_2) + (1 - p_same_mis_1_2)*p_misphase_1_2)
##
##
##            p_same_1_23, p_same_mis_1_23, p_misphase_1_23 = self.t_probs(np.r_[mid, en], np.r_[st, en])
##            p_diff_1_23 = np.log((1 - p_same_1_23)*(1 - p_misphase_1_23) + (1 - p_same_mis_1_23)*p_misphase_1_23)
##
##            p_same_23_4, p_same_mis_23_4, p_misphase_23_4 = self.t_probs(np.r_[st, en], np.r_[st, mid])
##            p_diff_23_4 = np.log((1 - p_same_23_4)*(1 - p_misphase_23_4) + (1 - p_same_mis_23_4)*p_misphase_23_4)
##
##            p_diff_1_2 + np.log(np.maximum(trans_probs[2:], _EPS).sum()) + p_diff_3_4 - (p_diff_1_23 + p_diff_23_4)
##
##            MH_ratio = np.log(np.maximum(trans_probs[2:], _EPS).sum()) - np.log(np.maximum(trans_probs[0:2], _EPS).sum())
#
#            prev_marg_lik = self.marg_lik
#            self.marg_lik -= self.seg_marg_liks[st]
#            seg_lik_1 = ss.betaln(
#              self.P.loc[st:(mid - 1), "MIN_COUNT"].sum() + 1,
#              self.P.loc[st:(mid - 1), "MAJ_COUNT"].sum() + 1
#            )
#            seg_lik_2 = ss.betaln(
#              self.P.loc[mid:(en - 1), "MIN_COUNT"].sum() + 1,
#              self.P.loc[mid:(en - 1), "MAJ_COUNT"].sum() + 1
#            )
#            self.marg_lik += seg_lik_1 + seg_lik_2
#
#            # q(join)/q(split) = p(picking first segment)*p(joining with subsequent)/
#            #                    p(picking first + second segment)*p(picking breakpoint)*p(breaking)
#            log_q_rat = -np.log(len(self.breakpoints)) + log_trans_probs[trans & 1] - \
#              (-np.log(len(self.breakpoints) - 1) + np.log(split_probs[b]) + log_trans_probs[trans])
#
#            # accept transition
#            if np.log(np.random.rand()) < np.minimum(0, self.marg_lik - prev_marg_lik + log_q_rat):
#                self.breakpoints.add(mid)
#                self.breakpoint_counter[mid] += np.r_[1, 1]
#                self.seg_marg_liks[st] = seg_lik_1
#                self.seg_marg_liks[mid] = seg_lik_2
#
#            # don't accept; revert
#            else:
#                if flipped:
#                    self.flip_hap(mid, en)
#                self.marg_lik = prev_marg_lik
#                self.breakpoint_counter[mid] += np.r_[0, 1]
#
#        # don't split segment
#        else:
#            self.breakpoint_counter[mid] += np.r_[0, 1]

    def incr_bp_counter(self, st, en, mid = None):
        if mid is None:
            self.breakpoint_counter[st] += np.r_[1, 1]
            self.breakpoint_counter[(st + 1):en] += np.r_[0, 1]
        else:
            self.breakpoint_counter[st] += np.r_[1, 1]
            self.breakpoint_counter[(st + 1):mid] += np.r_[0, 1]
            self.breakpoint_counter[mid] += np.r_[1, 1]
            self.breakpoint_counter[(mid + 1):en] += np.r_[0, 1]

    def visualize(self):
        Ph = self.P.copy()
        CI = s.beta.ppf([0.05, 0.5, 0.95], Ph["MAJ_COUNT"][:, None] + 1, Ph["MIN_COUNT"][:, None] + 1)
        Ph[["CI_lo_hap", "median_hap", "CI_hi_hap"]] = CI

        plt.figure(); plt.clf()
        ax = plt.gca()

        # SNPs
        plt.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.5, color = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]][Ph["aidx_orig"].astype(np.int)])

#        # phase switches
#        o = 0
#        for i in Ph["flip"].unique():
#            if i == 0:
#                continue
#            plt.scatter(Ph.loc[Ph["flip"] == i, "pos"], o + np.zeros((Ph["flip"] == i).sum()))
#            o -= 0.01

        # breakpoints 
        bp_prob = self.breakpoint_counter[:, 0]/self.breakpoint_counter[:, 1]
        bp_idx = np.flatnonzero(bp_prob > 0)
        for i in bp_idx:
            col = 'k' if bp_prob[i] < 0.8 else 'm'
            alph = bp_prob[i]/2 if bp_prob[i] < 0.8 else bp_prob[i]
            plt.axvline(Ph.iloc[i, Ph.columns.get_loc("pos")], color = col, alpha = alph)
        ax2 = ax.twiny()
        ax2.set_xticks(Ph.iloc[self.breakpoints, Ph.columns.get_loc("pos")]);
        ax2.set_xticklabels(bp_idx);
        ax2.set_xlim(ax.get_xlim());
        ax2.set_xlabel("Breakpoint number in current MCMC iteration")

        # beta CI's weighted by breakpoints
        for bp_samp in self.breakpoint_list:
            bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]
            for st, en in bpl:
                ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], Ph.iloc[st:en, self.maj_idx].sum() + 1, Ph.iloc[st:en, self.min_idx].sum() + 1)
                ax.add_patch(mpl.patches.Rectangle((Ph.iloc[st, 1], ci_lo), Ph.iloc[en, 1] - Ph.iloc[st, 1], ci_hi - ci_lo, fill = True, facecolor = 'k', alpha = 0.01, zorder = 1000))

        ax.set_xticks(np.linspace(*plt.xlim(), 20));
        ax.set_xticklabels(Ph["pos"].searchsorted(np.linspace(*plt.xlim(), 20)));
        ax.set_xlabel("SNP index")
