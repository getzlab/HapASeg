import colorama
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

class MySortedList(sc.SortedList):
    # since the sorted list represents intervals, for debugging purposes it's
    # a lot easier to output them in columnar format:
    def __repr__(self):
        assert not len(self) % 2
        return str(np.array(self).reshape(-1, 2))

    def intervals(self):
        return np.array(self).reshape(-1, 2)

class A_MCMC:
    def __init__(self, P,
      quit_after_burnin = False,
      n_iter = 100000,
      ref_bias = 1.0,
      misphase_prior = 0.001,
      phase_correct = False
    ):
        #
        # dataframe stuff
        self.P = P.copy().reset_index()

        # column indices for iloc
        self.min_idx = self.P.columns.get_loc("MIN_COUNT")
        self.maj_idx = self.P.columns.get_loc("MAJ_COUNT")

        # factor by which to downscale all reference alleles, in order to
        # correct for bias against the alternate allele due to capture or alignment
        self.P["REF_COUNT"] *= ref_bias
        self.P["MAJ_COUNT"] = pd.concat([self.P.loc[self.P["aidx"], "ALT_COUNT"], self.P.loc[~self.P["aidx"], "REF_COUNT"]])
        self.P["MIN_COUNT"] = pd.concat([self.P.loc[self.P["aidx"], "REF_COUNT"], self.P.loc[~self.P["aidx"], "ALT_COUNT"]])

        # TODO: recompute CI's too? these are not actually used anywhere
        # TODO: we might want to have site-specific reference bias (inferred from post-burnin segs)

        # whether a SNP is included or pruned
        self.P["include"] = True

        # prior for pruning
        self.P["include_prior"] = self._set_prune_prior()

        # state of inclusion at nth iteration
        self.include = []

        #
        # config stuff

        # number of MCMC iterations
        self.n_iter = n_iter

        self.quit_after_burnin = quit_after_burnin

        self.misphase_prior = misphase_prior

        # whether to perform phasing correction iterations
        self.phase_correct = phase_correct

        # how many post-burnin samples to use to infer phase switches
        self.n_phase_correct_samples = 40

        #
        # chain state
        self.iter = 1
        self.burned_in = False

        # whether phase correction has been performed
        self.phase_correction_ready = False

        #
        # breakpoint storage

        # breakpoints of current iteration. initialize with each SNP belonging
        # to its own breakpoint.
        self.breakpoints = sc.SortedSet(range(0, len(self.P)))

        # count of all breakpoints ever created
        # breakpoint -> (number of times confirmed, number of times sampled)
        self.breakpoint_counter = np.zeros((len(self.P), 2), dtype = np.int)

        # list of all breakpoints at nth iteration
        self.breakpoint_list = []

        #
        # misphase interval storage

        # candidate intervals that were misphased
        self.B_ct = sp.dok_matrix((len(self.P), len(self.P)), dtype = np.int)

        # current state of interval assignments (relative to B_ct)
        self.F = MySortedList()

        # state of interval assignments at nth iteration
        self.phase_interval_list = []

        #
        # cumsum arrays for each segment

        # will be populated by compute_cumsums()
        # NOTE: not currently used for anything

        self.cs_MAJ = sc.SortedDict()
        self.cs_MIN = sc.SortedDict()

        # probability of picking a breakpoint
        self.split_prob = sc.SortedDict()

        #
        # marginal likelihoods

        # log marginal likelihoods for each segment
        # initialize with each SNP comprising its own segment.
        self.seg_marg_liks = sc.SortedDict(zip(
          range(0, len(self.P)),
          ss.betaln(
            self.P.iloc[0:len(self.P), self.min_idx] + 1,
            self.P.iloc[0:len(self.P), self.maj_idx] + 1
          )
        ))

        # total log marginal likelihood of all segments
        self.marg_lik = np.full(self.n_iter, np.nan)
        self.marg_lik[0] = np.array(self.seg_marg_liks.values()).sum()

    def _Piloc(self, st, en, col_idx):
        """
        Returns only SNPs flagged for inclusion within the range st:en
        """
        P = self.P.iloc[st:en, col_idx]
        return P.loc[self.P.iloc[st:en, self.P.columns.get_loc("include")]]

    def run(self):
        while self.iter < self.n_iter:
            # perform a split, combine, phase correct, or prune operation
            op = np.random.choice(4)
            if op == 0:
                if self.combine(np.random.choice(self.breakpoints[:-1]), force = False) == -1:
                    continue
            elif op == 1:
                if self.split(b_idx = np.random.choice(len(self.breakpoints))) == -1:
                    continue
            elif op == 2:
                if self.phase_correct and self.phase_correction_ready:
                    self.rephase()
                else:
                    continue
            elif op == 3:
                if np.random.rand() < 0.01:
                    self.prune()
                else:
                    continue

            # if we're only running up to burnin, bail
            if self.quit_after_burnin and self.burned_in:
                print(colorama.Fore.GREEN + "Burned in [{st},{en}] in {n} iterations. n_bp = {n_bp}, lik = {lik}".format(
                  st = self.P["index"].iloc[0],
                  en = self.P["index"].iloc[-1],
                  n = self.iter,
                  n_bp = len(self.breakpoints),
                  lik = self.marg_lik[self.iter]
                ) + colorama.Fore.RESET)
                return self

            # correct phases after some post-burnin iterations
            if not self.phase_correction_ready and self.phase_correct and \
              self.burned_in and len(self.breakpoint_list) >= 2*self.n_phase_correct_samples:
                self.correct_phases()
                self.phase_correction_ready = True

                # breakpoint list is liable to change after phase correction, so clear it
                self.breakpoint_list = []

            # save set of breakpoints, phase intervals, and prune states if burned in 
            if self.burned_in and not self.iter % 100:
                self.breakpoint_list.append(self.breakpoints.copy())
                self.include.append(self.P["include"].copy())
                if self.phase_correction_ready:
                    self.phase_interval_list.append(self.F.copy())

            # print status
            if not self.iter % 100:
                if self.burned_in:
                    color = colorama.Fore.MAGENTA if not self.phase_correction_ready else colorama.Fore.RESET
                else:
                    color = colorama.Fore.YELLOW
                print("{color}[{st},{en}]\t{n}/{tot}\tn_bp = {n_bp}\tlik = {lik}".format(
                  st = self.P["index"].iloc[0],
                  en = self.P["index"].iloc[-1],
                  n = self.iter,
                  tot = self.n_iter,
                  n_bp = len(self.breakpoints),
                  lik = self.marg_lik[self.iter],
                  color = color
                ))

            # check if we've burned in
            # TODO: use a faster method of computing rolling average
            if not self.burned_in and self.iter > 500:
                if np.diff(self.marg_lik[(self.iter - 500):self.iter]).mean() < 0:
                    self.burned_in = True

            self.iter += 1 

        return self

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

        # M-H acceptance
        ML_split = self.seg_marg_liks[st] + self.seg_marg_liks[mid]

        ML_join = ss.betaln(
          self._Piloc(st, en, self.min_idx).sum() + 1,
          self._Piloc(st, en, self.maj_idx).sum() + 1
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

            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1] - ML_split + ML_join
            if self.burned_in:
                self.incr_bp_counter(st = st, en = en)

            return st

        # don't accept transition; undo
        else:
            if self.burned_in:
                self.incr_bp_counter(st = st, mid = mid, en = en)

            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]

            return mid

    def flip_hap(self, st, en):
        """
        Flips the SNPs from st to en
        """

        x = self.P.iloc[st:en, self.maj_idx].copy()
        self.P.iloc[st:en, self.maj_idx] = self.P.iloc[st:en, self.min_idx]
        self.P.iloc[st:en, self.min_idx] = x

    def prob_misphase(self, bdy1, bdy2):
        """
        Compute probability of misphase
        """
        # TODO: change invocation to st, mid, en -- we don't need to correct
        #       phasing of noncontiguous segments

        # prior on misphasing probability
        p_mis = self.misphase_prior if np.isnan(self.P.loc[bdy1[1] - 1, "misphase_prob"]) else self.P.loc[bdy1[1] - 1, "misphase_prob"]
        if p_mis == 0:
            return -np.inf, 0

        # haps = x/y, segs = 1/2, beta params. = A/B

        # seg 1
        rng_idx = (self.P.index >= bdy1[0]) & (self.P.index < bdy1[1])

        idx = rng_idx & self.P["aidx"] & self.P["include"]
        x1_A = self.P.loc[idx, "ALT_COUNT"].sum() + 1
        x1_B = self.P.loc[idx, "REF_COUNT"].sum() + 1

        idx = rng_idx & ~self.P["aidx"] & self.P["include"]
        y1_A = self.P.loc[idx, "ALT_COUNT"].sum() + 1
        y1_B = self.P.loc[idx, "REF_COUNT"].sum() + 1

        # seg 2
        rng_idx = (self.P.index >= bdy2[0]) & (self.P.index < bdy2[1])

        idx = rng_idx & self.P["aidx"] & self.P["include"]
        x2_A = self.P.loc[idx, "ALT_COUNT"].sum() + 1
        x2_B = self.P.loc[idx, "REF_COUNT"].sum() + 1

        idx = rng_idx & ~self.P["aidx"] & self.P["include"]
        y2_A = self.P.loc[idx, "ALT_COUNT"].sum() + 1
        y2_B = self.P.loc[idx, "REF_COUNT"].sum() + 1

        lik_mis   = ss.betaln(x1_A + y1_B + y2_A + x2_B, y1_A + x1_B + x2_A + y2_B)
        lik_nomis = ss.betaln(x1_A + y1_B + x2_A + y2_B, y1_A + x1_B + y2_A + x2_B)

        # logsumexp
        m = np.maximum(lik_mis, lik_nomis)
        denom = m + np.log(np.exp(lik_mis - m)*p_mis + np.exp(lik_nomis - m)*(1 - p_mis))

        return lik_mis + np.log(p_mis) - denom, lik_nomis + np.log(1 - p_mis) - denom

    def correct_phases(self):
        """
        Compute potentially misphased intervals, given some segmentation samples
        """
        if not self.burned_in or len(self.breakpoint_list) == 0:
            raise RuntimeError("Breakpoint sample list must be populated (chain must be burned in)")

        #A_ct = sp.dok_matrix((len(self.P), len(self.P)), dtype = np.int)
        #B_ct = sp.dok_matrix((len(self.P), len(self.P)), dtype = np.int)

        for bp_idx in np.random.choice(len(self.breakpoint_list), self.n_phase_correct_samples, replace = False):
            bpl = np.array(self.breakpoint_list[bp_idx]); bpl = np.c_[bpl[:-1], bpl[1:]]

            p_mis = np.full(len(bpl) - 1, np.nan)
            p_A = np.full(len(bpl) - 1, np.nan)
            p_B = np.full(len(bpl) - 1, np.nan)

            V = np.full([len(bpl) - 1, 2], np.nan)
            B = np.zeros([len(bpl) - 1, 2], dtype = np.uint8)

            for i, (st, mid, _, en) in enumerate(np.c_[bpl[:-1], bpl[1:]]):
                p_mis, p_nomis = self.prob_misphase([st, mid], [mid, en])

                # TODO: memoize partial sums

                # prob. that left segment is on hap. A
                p_A1 = s.beta.logsf(0.5, self._Piloc(st, mid, self.min_idx).sum() + 1, self._Piloc(st, mid, self.maj_idx).sum() + 1)
                # prob. that right segment is on hap. A
                p_A2 = s.beta.logsf(0.5, self._Piloc(mid, en, self.min_idx).sum() + 1, self._Piloc(mid, en, self.maj_idx).sum() + 1)

                # prob. that left segment is on hap. B
                p_B1 = s.beta.logcdf(0.5, self._Piloc(st, mid, self.min_idx).sum() + 1, self._Piloc(st, mid, self.maj_idx).sum() + 1)
                # prob. that right segment is on hap. B
                p_B2 = s.beta.logcdf(0.5, self._Piloc(mid, en, self.min_idx).sum() + 1, self._Piloc(mid, en, self.maj_idx).sum() + 1)

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

            # join contiguous segments assigned to hap. B
            d = np.diff(BT, append = 0, prepend = 0)
            ctg_idx = np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1]
            b_segs_j = np.c_[bpl[ctg_idx[:, 0], 0], bpl[ctg_idx[:, 1], 1]]

#            # join contiguous segments assigned to hap. A
#            d = np.diff(1 - BT, append = 0, prepend = 0)
#            ctg_idx = np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1]
#            a_segs_j = np.c_[bpl[ctg_idx[:, 0], 0], bpl[ctg_idx[:, 1], 1]]

            # plot
            #for x in np.flatnonzero(BT):
            #    plt.plot(self.P.loc[bpl[x], "pos"], np.r_[j + 1, j + 1]*0.01)

            # record
            for x in b_segs_j:
                self.B_ct[x[0], x[1]] += 1
#            for x in a_segs_j:
#                A_ct[x[0], x[1]] += 1

#        # plot
#        for k, v in B_ct.items():
#            for _ in range(0, v):
#                plt.plot(self.P.iloc[np.r_[k], self.P.columns.get_loc("pos")], 0.2*np.random.rand()*np.r_[1, 1])

    # MCMC iteration that corrects a phase
    def rephase(self): # TODO: add parameters to force an interval?
        # TODO: prerequisite checks; has correct_phases() been run?
        choice = list(self.B_ct.keys())
        probs = np.r_[list(self.B_ct.values())]

        #
        # propose an interval to flip from B->A
        st, en = choice[np.random.choice(np.r_[0:len(choice)], p = probs/probs.sum())]

        #
        # check if this overlaps any other regions that were already flipped B->A.

        # any previously flipped regions contained within will be left alone

        # return range of flipped region array that [st, en) overlaps
        # TODO: rename this; f_o is a terrible name
        def f_o(st = st, en = en):
            st_idx = self.F.bisect_left(st + 1); st_idx -= st_idx % 2
            en_idx = self.F.bisect_right(en - 1); en_idx += en_idx % 2
            return slice(st_idx, en_idx)

        overlaps = np.array(self.F[f_o()]).reshape(-1, 2)
        o_S = sc.SortedSet({st, en})
        for o in overlaps:
            o_S.add(o[0])
            o_S.add(o[1])

        # somewhere we ought to assert that the length of self.F is even

        # get list of regions to flip
        flip_candidates = np.r_[o_S] # all possible regions to flip
        flip_idx = np.zeros(len(flip_candidates) - 1, dtype = np.bool) # index of regions that haven't been flipped yet
        A_flag = True # whether st:en consists entirely of regions that were flipped to A
        for i, (st_seg, en_seg) in enumerate(np.c_[flip_candidates[:-1], flip_candidates[1:]]):
            # this region was not already flipped B->A
            if not self.F[f_o(st_seg, en_seg)]:
                flip_idx[i] = True
                A_flag = False

        flips = np.c_[flip_candidates[:-1], flip_candidates[1:]][flip_idx, :]

        #
        # get full range of CNV breakpoints this region spans
        st_reg = self.breakpoints.bisect_left(o_S[0])
        en_reg = self.breakpoints.bisect_right(o_S[-1])
        breakpoints0 = sc.SortedSet(self.breakpoints[(st_reg - 1):(en_reg + 1)])

        #
        # get initial marginal likelihood of this configuration
        ML_orig = 0
        for b in breakpoints0[:-1]:
            ML_orig += self.seg_marg_liks[b]

        #
        # perform flips; update breakpoint list accordingly
        for st_seg, en_seg in flips:
            # if flip boundary corresponds to an extant breakpoint, remove it
            # (we will propose joining these segments after flip)
            if st_seg in breakpoints0:
                breakpoints0 -= {st_seg}
            # otherwise, add the flip boundary as a new breakpoint
            # (we will propose introducing a new segment after flip)
            else:
                breakpoints0.add(st_seg)
            if en_seg in breakpoints0:
                breakpoints0 -= {en_seg}
            else:
                breakpoints0.add(en_seg)

            self.flip_hap(st_seg, en_seg)

        #
        # if st:en is entirely assigned to A, try to flip it back to B (i.e. it was a false flip)
        if A_flag:
            if flip_candidates[0] in breakpoints0:
                breakpoints0 -= {flip_candidates[0]}
            else:
                breakpoints0.add(flip_candidates[0])
            if en_reg in breakpoints0:
                breakpoints0 -= {flip_candidates[-1]}
            else:
                breakpoints0.add(flip_candidates[-1])

            self.flip_hap(flip_candidates[0], flip_candidates[-1])

        #
        # get marginal likelihood post-flip and breakpoint adjustment
        bps = np.r_[breakpoints0]
        ML = 0
        for st_bp, en_bp in np.c_[bps[:-1], bps[1:]]:
            ML += ss.betaln(
              self._Piloc(st_bp, en_bp, self.min_idx).sum() + 1,
              self._Piloc(st_bp, en_bp, self.maj_idx).sum() + 1
            )

        #
        # probabilistically accept new configuration
        if np.log(np.random.rand()) < np.minimum(0, ML - ML_orig):
            #
            # update F array

            # we could have either flipped a region from B->A ...
            if not A_flag:
                for st_seg, en_seg in flips:
                    self.F.update([st_seg, en_seg])

            # ... or reverted a flip
            else:
                for p in self.F[f_o(flip_candidates[0], flip_candidates[-1])]:
                    self.F.remove(p)

            #
            # combine contiguous intervals in F array
            # TODO

            #
            # update breakpoint list and seg. marg. liks
            bps_to_del = list(self.breakpoints.islice(
              self.breakpoints.bisect_left(breakpoints0[0]),
              self.breakpoints.bisect_right(breakpoints0[-1])
            ))
            for x in bps_to_del:
                self.breakpoints.remove(x)
            self.breakpoints.update(breakpoints0)

            #
            # update seg. marg. liks
            # TODO: recomputing each sum (even if in the future we use memoization)
            #       is wasteful. intelligently pick which seg_marg_liks keys to update.
            for x in bps_to_del[:-1]:
                self.seg_marg_liks.__delitem__(x)
            for st_bp, en_bp in np.c_[bps[:-1], bps[1:]]:
                self.seg_marg_liks[st_bp] = ss.betaln(
                  self._Piloc(st_bp, en_bp, self.min_idx).sum() + 1,
                  self._Piloc(st_bp, en_bp, self.maj_idx).sum() + 1
                )

            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1] - ML_orig + ML

        #
        # revert
        else:
            # flip each region back
            for st_seg, en_seg in flips:
                self.flip_hap(st_seg, en_seg)
            if A_flag:
                self.flip_hap(flip_candidates[0], flip_candidates[-1])

            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]

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
            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]
            return

        # M-H acceptance
        seg_lik_1 = ss.betaln(
          self._Piloc(st, mid, self.min_idx).sum() + 1,
          self._Piloc(st, mid, self.maj_idx).sum() + 1
        )
        seg_lik_2 = ss.betaln(
          self._Piloc(mid, en, self.min_idx).sum() + 1,
          self._Piloc(mid, en, self.maj_idx).sum() + 1
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

            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1] + ML_split - ML_join
            if self.burned_in:
                self.incr_bp_counter(st = st, mid = mid, en = en)

        # don't accept
        else:
            if self.burned_in:
                self.incr_bp_counter(st = st, en = en)

            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]

    def prune(self):
        incl_cols = self.P.columns.get_indexer(["include", "include_prior"])

        # we will incrementally update this
        self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]

        # prune SNPs within every breakpoint
        bpl = np.array(self.breakpoints); bpl = np.c_[bpl[:-1], bpl[1:]]
        for st, en in bpl:
            # don't prune short segments
            if en - st <= 2:
                continue

            T = self.P.iloc[st:en, np.r_[self.min_idx, self.maj_idx, incl_cols]]
            I = T.loc[T["include"]]
            E = T.loc[~T["include"]]

            A_inc_s = I["MIN_COUNT"].sum()
            B_inc_s = I["MAJ_COUNT"].sum()

            #
            # generate proposal dist from posterior ratios:

            # 1. probability to exclude SNPs
            # q_i = seg(A - A_i, B - B_i) + garbage(A_i, B_i) + (1 - include prior_i)
            #       - (seg(A, B) + (include prior_i))
            r_exc = ss.betaln(
              A_inc_s - I["MIN_COUNT"] + 1,
              B_inc_s - I["MAJ_COUNT"] + 1
            ) + ss.betaln(I["MIN_COUNT"] + 1, I["MAJ_COUNT"] + 1) \
              + np.log(1 - I["include_prior"]) \
              - (ss.betaln(A_inc_s + 1, B_inc_s + 1) + np.log(I["include_prior"]))

            # 2. probability to include SNPs (that were previously excluded)
            # q_i = seg(A + A_i, B + B_i) + (include prior_i)
            #       - (seg(A, B) + garbage(A_i, B_i) + (1 - include prior_i))
            r_inc = ss.betaln(
              A_inc_s + E["MIN_COUNT"] + 1,
              B_inc_s + E["MAJ_COUNT"] + 1
            ) + np.log(E["include_prior"]) \
              - (ss.betaln(A_inc_s + 1, B_inc_s + 1) + \
                ss.betaln(E["MIN_COUNT"] + 1, E["MAJ_COUNT"] + 1) + \
                np.log(1 - E["include_prior"]))

            r_cat = pd.concat([r_inc, r_exc]).sort_index()

            # normalize posterior ratios to get proposal distribution
            r_e = np.exp(r_cat - r_cat.max())
            q = r_e/r_e.sum()

            # draw from proposal
            choice_idx = np.random.choice(T.index, p = q)

            #
            # compute probability of proposing the reverse jump
            T_star = T.loc[[choice_idx]]

            # reverse jump excludes
            if not T_star.iat[0, T_star.columns.get_loc("include")]:
                A_inc_s_star = I["MIN_COUNT"].sum() + T_star["MIN_COUNT"].values
                B_inc_s_star = I["MAJ_COUNT"].sum() + T_star["MAJ_COUNT"].values

                I_star = pd.concat([I, T_star]).sort_index()
                E_star = E.drop(T_star.index)

            # reverse jump includes
            else:
                A_inc_s_star = I["MIN_COUNT"].sum() - T_star["MIN_COUNT"].values
                B_inc_s_star = I["MAJ_COUNT"].sum() - T_star["MAJ_COUNT"].values

                I_star = I.drop(T_star.index)
                E_star = pd.concat([E, T_star]).sort_index()

            # regardless, code for computing q_star is the same
            r_exc_star = ss.betaln(
              A_inc_s_star - I_star["MIN_COUNT"] + 1,
              B_inc_s_star - I_star["MAJ_COUNT"] + 1
            ) + ss.betaln(I_star["MIN_COUNT"] + 1, I_star["MAJ_COUNT"] + 1) \
              + np.log(1 - I_star["include_prior"]) \
              - (ss.betaln(A_inc_s_star + 1, B_inc_s_star + 1) + np.log(I_star["include_prior"]))

            r_inc_star = ss.betaln(
              A_inc_s_star + E_star["MIN_COUNT"] + 1,
              B_inc_s_star + E_star["MAJ_COUNT"] + 1
            ) + np.log(E_star["include_prior"]) \
              - (ss.betaln(A_inc_s_star + 1, B_inc_s_star + 1) + \
                ss.betaln(E_star["MIN_COUNT"] + 1, E_star["MAJ_COUNT"] + 1) + \
                np.log(1 - E_star["include_prior"]))

            r_cat_star = pd.concat([r_inc_star, r_exc_star]).sort_index()

            r_e_star = np.exp(r_cat_star - r_cat_star.max())
            q_star = (r_e_star/r_e_star.sum())[T_star.index].values

            # proposal ratio term
            q_rat = np.log(q_star) - np.log(q[choice_idx])

            # accept via Metropolis
            if np.log(np.random.rand()) < np.minimum(0, (r_cat[choice_idx] + q_rat).item()):
                # update inclusion flag
                self.P.at[choice_idx, "include"] = ~self.P.at[choice_idx, "include"]

                # update marginal likelihoods
                T.at[choice_idx, "include"] = ~T.at[choice_idx, "include"]

                self.marg_lik[self.iter] -= self.seg_marg_liks[st]
                self.seg_marg_liks[st] = ss.betaln(
                  T.loc[T["include"], "MIN_COUNT"].sum() + 1,
                  T.loc[T["include"], "MAJ_COUNT"].sum() + 1,
                )
                self.marg_lik[self.iter] += self.seg_marg_liks[st]

                # account for SNPs sent to "garbage" in likelihood (they are
                # effectively their own segments)
                self.marg_lik[self.iter] += (1 if ~self.P.at[choice_idx, "include"] else -1)* \
                  ss.betaln(
                    self.P.at[choice_idx, "MIN_COUNT"] + 1,
                    self.P.at[choice_idx, "MAJ_COUNT"] + 1
                  )

                # TODO: update segment partial sums (when we actually use these)

    def _set_prune_prior(self):
        if all([x in self.P.columns for x in ["REF_COUNT_N", "ALT_COUNT_N"]]):
            # TODO: also account for het site panel
            return np.diff(s.beta.cdf([0.4, 0.6], self.P["ALT_COUNT_N"].values[:, None] + 1, self.P["REF_COUNT_N"].values[:, None] + 1), 1)
        else:
            return 0.9

    def incr_bp_counter(self, st, en, mid = None):
        if mid is None:
            self.breakpoint_counter[st] += np.r_[1, 1]
            self.breakpoint_counter[(st + 1):en] += np.r_[0, 1]
        else:
            self.breakpoint_counter[st] += np.r_[1, 1]
            self.breakpoint_counter[(st + 1):mid] += np.r_[0, 1]
            self.breakpoint_counter[mid] += np.r_[1, 1]
            self.breakpoint_counter[(mid + 1):en] += np.r_[0, 1]

    def visualize(self, show_CIs = False):
        Ph = self.P.copy()
        CI = s.beta.ppf([0.05, 0.5, 0.95], Ph["MIN_COUNT"][:, None] + 1, Ph["MAJ_COUNT"][:, None] + 1)
        Ph[["CI_lo_hap", "median_hap", "CI_hi_hap"]] = CI

        plt.figure(); plt.clf()
        ax = plt.gca()

        # SNPs
        ax.scatter(Ph["pos"], Ph["median_hap"], color = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]][Ph["aidx"].astype(np.int)], alpha = 0.5, s = 4)
        if show_CIs:
            ax.errorbar(Ph["pos"], y = Ph["median_hap"], yerr = np.c_[Ph["median_hap"] - Ph["CI_lo_hap"], Ph["CI_hi_hap"] - Ph["median_hap"]].T, fmt = 'none', alpha = 0.5, color = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]][Ph["aidx"].astype(np.int)])

        # mask excluded SNPs
        ax.scatter(Ph["pos"], Ph["median_hap"], color = 'k', alpha = 1 - pd.concat(self.include, axis = 1).mean(1).values)

        # breakpoints 
#        bp_prob = self.breakpoint_counter[:, 0]/self.breakpoint_counter[:, 1]
#        bp_idx = np.flatnonzero(bp_prob > 0)
#        for i in bp_idx:
#            col = 'k' if bp_prob[i] < 0.8 else 'm'
#            alph = bp_prob[i]/2 if bp_prob[i] < 0.8 else bp_prob[i]
#            ax.axvline(Ph.iloc[i, Ph.columns.get_loc("pos")], color = col, alpha = alph)
#        ax2 = ax.twiny()
#        ax2.set_xticks(Ph.iloc[self.breakpoints, Ph.columns.get_loc("pos")]);
#        ax2.set_xticklabels(bp_idx);
#        ax2.set_xlim(ax.get_xlim());
#        ax2.set_xlabel("Breakpoint number in current MCMC iteration")

        # beta CI's weighted by breakpoints
        # flip current rephases back to baseline
        for st, en in self.F.intervals():
            # code excised from flip_hap
            x = Ph.iloc[st:en, self.maj_idx].copy()
            Ph.iloc[st:en, self.maj_idx] = Ph.iloc[st:en, self.min_idx]
            Ph.iloc[st:en, self.min_idx] = x

        pos_col = Ph.columns.get_loc("pos")
        for bp_samp, pi_samp, inc_samp in zip(self.breakpoint_list, self.phase_interval_list, self.include):
            # flip everything according to sample
            for st, en in pi_samp.intervals():
                # TODO: can replace with flip_hap()?
                x = Ph.iloc[st:en, self.maj_idx].copy()
                Ph.iloc[st:en, self.maj_idx] = Ph.iloc[st:en, self.min_idx]
                Ph.iloc[st:en, self.min_idx] = x

            # SNPs TODO: plot only those that flipped, in a diff. color?
            #ax.scatter(Ph["pos"], Ph["median_hap"], color = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]][Ph["aidx"].astype(np.int)], alpha = 0.5, s = 4)

            bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]
            for st, en in bpl:
                Phi = Ph.iloc[st:en]; Phi = Phi.loc[inc_samp]
                ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], Phi.iloc[:, self.min_idx].sum() + 1, Phi.iloc[:, self.maj_idx].sum() + 1)
                ax.add_patch(mpl.patches.Rectangle((Ph.iloc[st, pos_col], ci_lo), Ph.iloc[en, pos_col] - Ph.iloc[st, pos_col], ci_hi - ci_lo, fill = True, facecolor = 'k', alpha = 1/len(self.breakpoint_list), zorder = 1000))

            # flip everything back
            for st, en in pi_samp.intervals():
                # TODO: can replace with flip_hap()?
                x = Ph.iloc[st:en, self.maj_idx].copy()
                Ph.iloc[st:en, self.maj_idx] = Ph.iloc[st:en, self.min_idx]
                Ph.iloc[st:en, self.min_idx] = x

        # 50:50 line
        ax.axhline(0.5, color = 'k', linestyle = ":")

        ax.set_xticks(np.linspace(*plt.xlim(), 20));
        ax.set_xticklabels(Ph["pos"].searchsorted(np.linspace(*plt.xlim(), 20)));
        ax.set_xlabel("SNP index")
        ax.set_ylim([0, 1])
