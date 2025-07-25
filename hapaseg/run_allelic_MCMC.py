import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc
from itertools import chain

from . import A_MCMC

from capy import mut

class AllelicMCMCRunner:
    def __init__(self,
      allele_counts,
      chromosome_intervals,
      c,
      n_iter = 20000,
      phase_correct = True,
      misphase_prior = 0.001,
    ):
        self.client = c
        self.P = allele_counts
        self.P_shared = c.scatter(allele_counts)
        self.chr_int = chromosome_intervals

        self.n_iter = n_iter
        self.phase_correct = phase_correct
        self.misphase_prior = misphase_prior

        self.ref_bias = np.nan

        # make ranges
        t = mut.map_mutations_to_targets(self.P, self.chr_int, inplace = False)
        self.groups = t.groupby(t).apply(lambda x : [x.index.min(), x.index.max()]).to_frame(name = "bdy")
        self.groups["ranges"] = self.groups["bdy"].apply(lambda x : np.r_[x[0]:x[1]:5000, x[1]])
        self.chunks = pd.DataFrame(
          np.vstack([
            np.hstack(np.broadcast_arrays(k, np.c_[y[0:-1], y[1:]]))
            for k, y in self.groups["ranges"].iteritems()
          ])[:, 1:],
          columns = ["arm", "start", "end"]
        )

    @staticmethod
    def _run_on_chunks(rng, P):
        H = A_MCMC(P.iloc[rng], quit_after_burnin = True)
        return H.run()

    def run_all(self, chunks = None):
        #
        # scatter across chunks. for each range, run until burnin
        chunks = [slice(*x) for x in self.chunks[["start", "end"]].values] if chunks is None else chunks

        futures = self.client.map(self._run_on_chunks, chunks, P = self.P_shared)
        self.chunks["results"] = self.client.gather(futures)

        #
        # compute reference bias
        X = []
        j = 0
        for chunk in self.chunks["results"]:
            bpl = np.array(chunk.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]

            for i, (st, en) in enumerate(bpl):
                x = chunk.P.iloc[st:en].groupby("allele_A")[["MIN_COUNT", "MAJ_COUNT"]].sum()
                x["idx"] = i + j
                X.append(x)

            j += len(chunk.P)

        X = pd.concat(X)
        g = X.groupby("idx").size() == 2
        Y = X.loc[X["idx"].isin(g[g].index)]

        f = np.zeros([len(Y)//2, 100])
        l = np.zeros([len(Y)//2, 2])
        for i, (_, g) in enumerate(Y.groupby("idx")):
            f[i, :] = s.beta.rvs(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1, size = 100)/s.beta.rvs(g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1, size = 100)
            l[i, :] = np.r_[
              ss.betaln(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1),
              ss.betaln(g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1),
            ]

        # weight mean by negative log marginal likelihoods
        # take smaller of the two likelihoods to account for power imbalance
        w = np.min(-l, 1, keepdims = True)

        self.ref_bias = (f*w).sum()/(100*w.sum())

        #
        # concatenate burned in chunks for each arm
        H = [None]*len(self.chunks["arm"].unique())
        for i, (arm, A) in enumerate(self.chunks.groupby("arm")):
            # concatenate allele count dataframes
            H[i] = A_MCMC(
              pd.concat([x.P for x in A["results"]], ignore_index = True),
              n_iter = self.n_iter,
              phase_correct = self.phase_correct,
              misphase_prior = self.misphase_prior,
              ref_bias = self.ref_bias
            )

            # replicate constructor steps to define initial breakpoint set and
            # marginal likelihood dict
            breakpoints = [None]*len(A)
            H[i].seg_marg_liks = sc.SortedDict()
            for j, (_, _, start, _, r) in enumerate(A.itertuples()):
                start -= A["start"].iloc[0]
                breakpoints[j] = np.array(r.breakpoints) + start
                for k, v in r.seg_marg_liks.items():
                    H[i].seg_marg_liks[k + start] = v
            H[i].breakpoints = sc.SortedSet(np.hstack(breakpoints))

            H[i].marg_lik = np.full(H[i].n_iter, np.nan)
            H[i].marg_lik[0] = np.array(H[i].seg_marg_liks.values()).sum()

        Hs = self.client.scatter(H)

        #
        # run full MCMC on each arm
        arm_futures = self.client.map(lambda x : x.run(), Hs)
        self.groups["results"] = self.client.gather(arm_futures)
        self.groups = self.groups.join(self.chr_int, how = "right")

        return self.groups
