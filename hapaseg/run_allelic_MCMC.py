import numpy as np
from itertools import chain

from . import A_MCMC

from capy import mut

class AllelicMCMCRunner:
    def __init__(self, allele_counts, chromosome_intervals, c):
        self.client = c
        self.P = allele_counts
        self.P_shared = c.scatter(allele_counts)
        self.chr_int = chromosome_intervals

        # make ranges
        t = mut.map_mutations_to_targets(allele_counts, chromosome_intervals, inplace = False)
        self.groups = t.groupby(t).apply(lambda x : [x.index.min(), x.index.max()]).to_frame(name = "bdy")
        self.groups["ranges"] = self.groups["bdy"].apply(lambda x : np.r_[x[0]:x[1]:5000, x[1]])
        self.chunks = list(chain(*[[slice(*x) for x in np.c_[y[0:-1], y[1:]]] for y in self.groups["ranges"]]))

    @staticmethod
    def run(rng, P):
        H = A_MCMC(P.iloc[rng], quit_after_burnin = True)
        H.run()
        return H

    def run_all(self, chunks = None):
        #
        # scatter across chunks. for each range, run until burnin
        chunks = self.chunks if chunks is None else chunks

        futures = self.client.map(self.run, chunks, P = self.P_shared)
        results = self.client.gather(futures)

        #
        # concatenate burned in chunks; run again for full number of iterations

        # TODO: run in parallel for each arm

        # concat P dataframes
        H = A_MCMC(self.P)
        H.P = pd.concat([x.P for x in results], ignore_index = True)

        # replicate constructor steps
        H.P["index"] = range(0, len(H.P))

        # concat breakpoint lists
        breakpoints = [None]*len(chunk_bdy)
        H.seg_marg_liks = sc.SortedDict()
        for i, (r, b) in enumerate(zip(results_g, chunk_bdy[:, 0])):
            breakpoints[i] = np.array(r.breakpoints) + b
            for k, v in r.seg_marg_liks.items():
                H.seg_marg_liks[k + b] = v
        H.breakpoints = sc.SortedSet(np.hstack(breakpoints))

        H.marg_lik = np.full(H.n_iter, np.nan)
        H.marg_lik[0] = np.array(H.seg_marg_liks.values()).sum()

        H.run()
