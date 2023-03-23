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
      wgs = False
    ):
        self.wgs = wgs
        #
        # dataframe stuff
        self.P = P.copy().reset_index()

        # factor by which to downscale all reference alleles, in order to
        # correct for bias against the alternate allele due to capture or alignment
        self._set_ref_bias(ref_bias)

        # column indices for iloc
        self.min_idx = self.P.columns.get_loc("MIN_COUNT")
        self.maj_idx = self.P.columns.get_loc("MAJ_COUNT")
        
        self.min_arr = self.P.iloc[:, self.min_idx].astype(int).values
        self.maj_arr = self.P.iloc[:, self.maj_idx].astype(int).values

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

        #
        # chain state
        self.iter = 1
        self.burned_in = False

        #
        # breakpoint storage

        # breakpoints of current iteration. initialize with each SNP belonging
        # to its own breakpoint.
        self.breakpoints = sc.SortedSet(range(0, len(self.P) + 1))

        # count of all breakpoints ever created
        # breakpoint -> (number of times confirmed, number of times sampled)
        self.breakpoint_counter = np.zeros((len(self.P), 2), dtype = np.int)

        # list of all breakpoints at nth iteration
        self.breakpoint_list = []

        # MLE breakpoint
        self.breakpoints_MLE = None

        #
        # cumsum arrays for each segment

        # will be populated by compute_cumsums()
        # NOTE: not currently used for anything

        self.cs_MAJ = sc.SortedDict()
        self.cs_MIN = sc.SortedDict()

        self._set_betahyp()
        
        # marginal likelihoods

        # log marginal likelihoods for each segment
        # initialize with each SNP comprising its own segment.
        self.seg_marg_liks = sc.SortedDict(zip(
          range(0, len(self.P)),
          ss.betaln(
            self.P.iloc[0:len(self.P), self.min_idx] + 1 + self.betahyp,
            self.P.iloc[0:len(self.P), self.maj_idx] + 1 + self.betahyp
          )
        ))

        # total log marginal likelihood of all segments
        self.marg_lik = np.full(self.n_iter, np.nan)
        self.marg_lik[0] = np.array(self.seg_marg_liks.values()).sum()

    # beta smoothing hyperparameter depends on modality
    # TODO: estimate this based on SNP density?
    def _set_betahyp(self):
        if self.wgs:
            self.betahyp = (self.P["REF_COUNT"] + self.P["ALT_COUNT"]).mean()/4.0
        else:
            self.betahyp = 0

    def _Piloc(self, st, en, col_idx, incl_idx = None):
        """
        Returns only SNPs flagged for inclusion within the range st:en
        """
        P = self.P.iloc[st:en, col_idx]
        return P.loc[
          self.P.iloc[st:en, self.P.columns.get_loc("include")] if incl_idx is None else incl_idx
        ]

    def run(self):
        while self.iter < self.n_iter:
            # perform a split or combine
            op = np.random.choice(2)
            if op == 0:
                if self.combine(np.random.choice(self.breakpoints[:-1]), force = False) == -1:
                    continue
            elif op == 1:
                if self.split(b_idx = np.random.choice(len(self.breakpoints))) == -1:
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

            # save MLE breakpoint if we've burned in
            if self.burned_in:
                if self.marg_lik[self.iter] > self.marg_lik[self.iter - 1]:
                    self.breakpoints_MLE = self.breakpoints.copy()

            # print status
            if not self.iter % 100:
                if self.burned_in:
                    color = colorama.Fore.RESET
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

            # check if we've burned in -- chain is oscillating around some
            # optimium (and thus mean differences between marginal likelihoods might
            # be slightly negative)
            # TODO: use a faster method of computing rolling average
            if not self.burned_in and self.iter > 1000:
                if np.diff(self.marg_lik[(self.iter - 1000):self.iter]).mean() < 0:
                    self.burned_in = True
                # contingency if we've unambiguously converged on an optimum and chain has not moved at all
                # exit early to save time
                if (np.diff(self.marg_lik[(self.iter - 1000):self.iter]) == 0).all():
                    self.breakpoints_MLE = self.breakpoints.copy()
                    print(colorama.Fore.GREEN + "Chain has unambiguously converged on an optimum; stopping early in {n} iterations. n_bp = {n_bp}, lik = {lik}".format(
                      n = self.iter,
                      n_bp = len(self.breakpoints),
                      lik = self.marg_lik[self.iter]
                    ) + colorama.Fore.RESET)
                    return self

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
          self.min_arr[st:en].sum() + 1 + self.betahyp,
          self.maj_arr[st:en].sum() + 1 + self.betahyp
        )

        # proposal dist. ratio
        _, _, split_probs = self.compute_cumsum(st, en)
        # q(split)/q(join) = p(picking mid as breakpoint)/
        #                    p(picking first segment)
        log_q_rat = np.log(split_probs[mid - st - 1]) - -np.log(len(self.breakpoints))

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

    def compute_cumsum(self, st, en):
        cs_MAJ = self.maj_arr[st:en].cumsum()
        cs_MIN = self.min_arr[st:en].cumsum()
        # marginal likelihoods
        ml = ss.betaln(cs_MAJ + 1 + self.betahyp, cs_MIN + 1 + self.betahyp) + ss.betaln(cs_MAJ[-1] - cs_MAJ + 1 + self.betahyp, cs_MIN[-1] - cs_MIN + 1 + self.betahyp)

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

        # chosen split point is either (a) the segment end
        # we won't split this segment
        if b == len(split_probs) - 1:
            if self.burned_in:
                self.incr_bp_counter(st = st, en = en)
            self.marg_lik[self.iter] = self.marg_lik[self.iter - 1]
            return

        # M-H acceptance
        seg_lik_1 = ss.betaln(
          self.min_arr[st:mid].sum() + 1 + self.betahyp,
          self.maj_arr[st:mid].sum() + 1 + self.betahyp
        )
        seg_lik_2 = ss.betaln(
          self.min_arr[mid:en].sum() + 1 + self.betahyp,
          self.maj_arr[mid:en].sum() + 1 + self.betahyp
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
              A_inc_s - I["MIN_COUNT"] + 1 + self.betahyp,
              B_inc_s - I["MAJ_COUNT"] + 1 + self.betahyp
            ) + ss.betaln(I["MIN_COUNT"] + 1 + self.betahyp, I["MAJ_COUNT"] + 1 + self.betahyp) \
              + np.log(1 - I["include_prior"]) \
              - (ss.betaln(A_inc_s + 1 + self.betahyp, B_inc_s + 1 + self.betahyp) + np.log(I["include_prior"]))

            # 2. probability to include SNPs (that were previously excluded)
            # q_i = seg(A + A_i, B + B_i) + (include prior_i)
            #       - (seg(A, B) + garbage(A_i, B_i) + (1 - include prior_i))
            r_inc = ss.betaln(
              A_inc_s + E["MIN_COUNT"] + 1 + self.betahyp,
              B_inc_s + E["MAJ_COUNT"] + 1 + self.betahyp
            ) + np.log(E["include_prior"]) \
              - (ss.betaln(A_inc_s + 1 + self.betahyp, B_inc_s + 1 + self.betahyp) + \
                ss.betaln(E["MIN_COUNT"] + 1 + self.betahyp, E["MAJ_COUNT"] + 1 + self.betahyp) + \
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
              A_inc_s_star - I_star["MIN_COUNT"] + 1 + self.betahyp,
              B_inc_s_star - I_star["MAJ_COUNT"] + 1 + self.betahyp
            ) + ss.betaln(I_star["MIN_COUNT"] + 1 + self.betahyp, I_star["MAJ_COUNT"] + 1 + self.betahyp) \
              + np.log(1 - I_star["include_prior"]) \
              - (ss.betaln(A_inc_s_star + 1 + self.betahyp, B_inc_s_star + 1 + self.betahyp) + np.log(I_star["include_prior"]))

            r_inc_star = ss.betaln(
              A_inc_s_star + E_star["MIN_COUNT"] + 1 + self.betahyp,
              B_inc_s_star + E_star["MAJ_COUNT"] + 1 + self.betahyp
            ) + np.log(E_star["include_prior"]) \
              - (ss.betaln(A_inc_s_star + 1 + self.betahyp, B_inc_s_star + 1 + self.betahyp) + \
                ss.betaln(E_star["MIN_COUNT"] + 1 + self.betahyp, E_star["MAJ_COUNT"] + 1 + self.betahyp) + \
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
                  T.loc[T["include"], "MIN_COUNT"].sum() + 1 + self.betahyp,
                  T.loc[T["include"], "MAJ_COUNT"].sum() + 1 + self.betahyp,
                )
                self.marg_lik[self.iter] += self.seg_marg_liks[st]

                # account for SNPs sent to "garbage" in likelihood (they are
                # effectively their own segments)
                self.marg_lik[self.iter] += (1 if ~self.P.at[choice_idx, "include"] else -1)* \
                  ss.betaln(
                    self.P.at[choice_idx, "MIN_COUNT"] + 1 + self.betahyp,
                    self.P.at[choice_idx, "MAJ_COUNT"] + 1 + self.betahyp
                  )

                # TODO: update segment partial sums (when we actually use these)

    def _set_prune_prior(self):
        if all([x in self.P.columns for x in ["REF_COUNT_N", "ALT_COUNT_N"]]):
            # TODO: also account for het site panel
            return np.diff(s.beta.cdf([0.4, 0.6], self.P["ALT_COUNT_N"].values[:, None] + 1, self.P["REF_COUNT_N"].values[:, None] + 1), 1)
        else:
            return 0.9

    def _set_ref_bias(self, ref_bias): 
        # factor by which to downscale all reference alleles, in order to
        # correct for bias against the alternate allele due to capture or alignment
        self.ref_bias = ref_bias
        self.P["REF_COUNT"] *= self.ref_bias
        self.P["MAJ_COUNT"] = pd.concat([self.P.loc[self.P["aidx"], "ALT_COUNT"], self.P.loc[~self.P["aidx"], "REF_COUNT"]])
        self.P["MIN_COUNT"] = pd.concat([self.P.loc[self.P["aidx"], "REF_COUNT"], self.P.loc[~self.P["aidx"], "ALT_COUNT"]])

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
        plt.figure(figsize = [16, 4]); plt.clf()
        ax = plt.gca()

        # SNPs
        ax.scatter(self.P["pos"], self.P["median_hap"], color = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]][self.P["aidx"].astype(np.int)], alpha = 0.5, s = 4, marker = '.')
        if show_CIs:
            ax.errorbar(self.P["pos"], y = self.P["median_hap"], yerr = np.c_[self.P["median_hap"] - self.P["CI_lo_hap"], self.P["CI_hi_hap"] - self.P["median_hap"]].T, fmt = 'none', alpha = 0.1, ecolor = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]][self.P["aidx"].astype(np.int)])

        # mask excluded SNPs
        # ax.scatter(Ph["pos"], Ph["median_hap"], color = 'k', alpha = 1 - pd.concat(self.include, axis = 1).mean(1).values)

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

        bpl = self.breakpoints if self.breakpoints_MLE is None else self.breakpoints_MLE
        bpl = np.array(bpl); bpl = np.c_[bpl[0:-1], bpl[1:]]

        pos_col = self.P.columns.get_loc("pos")
        for st, en in bpl:
            ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], self.P.iloc[st:en, self.maj_idx].sum() + 1, self.P.iloc[st:en, self.min_idx].sum() + 1)
            ax.add_patch(mpl.patches.Rectangle((self.P.iloc[st, pos_col], ci_lo), 
                         self.P.iloc[min(en, len(self.P) - 1), pos_col] - self.P.iloc[st, pos_col],
                         ci_hi - ci_lo, fill = True, facecolor = 'lime', alpha = 0.4, zorder = 1000))

        # 50:50 line
        ax.axhline(0.5, color = 'k', linestyle = ":")

        ax.set_xticks(np.linspace(*plt.xlim(), 20));
        ax.set_xticklabels(self.P["pos"].searchsorted(np.linspace(*plt.xlim(), 20)));
        ax.set_xlabel("SNP index")
        ax.set_ylim([0, 1])

        ax.set_title(f"{self.P.iloc[0]['chr']}:{self.P.iloc[0]['pos']}-{self.P.iloc[-1]['pos']}")

        plt.tight_layout()
