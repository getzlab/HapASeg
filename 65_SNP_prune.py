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

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.pickle")
H = allelic_segs.iloc[0]["results"]
self = H
self.P["include"] = True
self.P["include_prior"] = 0.9

bp_idx = 10
bpl = np.array(self.breakpoint_list[bp_idx]); bpl = np.c_[bpl[:-1], bpl[1:]]

plt.figure(1); plt.clf()
_, axs = plt.subplots(10, 2, num = 1)

incl_cols = self.P.columns.get_indexer(["include", "include_prior"])

for st, en in bpl:
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

#    A = self.P.iloc[st:en, np.r_[self.min_idx, incl_cols]]
#    A_inc = A.loc[A["include"]]; A_exc = A.loc[~A["include"]]
#    B = self.P.iloc[st:en, np.r_[self.maj_idx, incl_cols]];
#    B_inc = B.loc[B["include"]]; B_exc = B.loc[~B["include"]]
#
#    A_inc_s = A_inc["MIN_COUNT"].sum()
#    A_exc_s = A_exc["MIN_COUNT"].sum()
#    B_inc_s = B_inc["MAJ_COUNT"].sum()
#    B_exc_s = B_exc["MAJ_COUNT"].sum()
#
#    # generate proposal dist from likelihood ratios:
#    # q_i =   seg(A - A_i, B - B_i) + garbage(A_i, B_i) + (1 - include prior)
#    #       - (seg(A, B) + (include prior))
#    p_inc = ss.betaln(
#      A_inc_s - A_inc["MIN_COUNT"] + 1,
#      B_inc_s - B_inc["MAJ_COUNT"] + 1
#    ) + ss.betaln(A_inc["MIN_COUNT"] + 1, B_inc["MAJ_COUNT"] + 1) \
#      + np.log(1 - A_inc["include_prior"]) \
#      - (ss.betaln(A_inc_s + 1, B_inc_s + 1) + np.log(A_inc["include_prior"]))
#    # q_i = seg(A, B) + (include prior)
#    #       - (seg(-A_i, -B_i) + garbage(A_i, B_i) + (1 - include prior)
#    p_exc = ss.betaln(A_inc_s + A_exc + 1, B_exc_s + B_exc + 1) + np.log(A_exc["include_prior"]) \
#      - (ss.betaln(A_exc_s + 1, B_exc_s + 1))
#
#ss.betaln(
#      A_exc_s - A_exc["MIN_COUNT"] + 1,
#      B_exc_s - B_exc["MAJ_COUNT"] + 1
#    ) + ss.betaln(A_exc["MIN_COUNT"], B_exc["MAJ_COUNT"]) \
#      + np.log(1 - A_exc["include_prior"]) \
#      - (ss.betaln(A_exc_s, B_exc_s) + np.log(A_exc["include_prior"]))
#    p_e = np.exp(p - p.max())
#    q = p_e/p_e.sum()
#
#    seg_idx = np.random.choice(len(q), p = q)
        # update marginal likelihoods
        T.at[choice_idx, "include"] = ~T.at[choice_idx, "include"]

        ML_orig = self.seg_marg_liks[st]
        self.seg_marg_liks[st] = ss.betaln(
          T.loc[T["include"], "MIN_COUNT"].sum() + 1,
          T.loc[T["include"], "MAJ_COUNT"].sum() + 1,
        )
        self.marg_lik[self.iter] = self.marg_lik[self.iter - 1] - ML_orig + self.seg_marg_liks[st]

        # TODO: update segment partial sums (when we actually use these)
