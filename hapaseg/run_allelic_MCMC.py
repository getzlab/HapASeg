from . import A_MCMC

class AllelicMCMCRunner:
    def __init__(self, P, c):
        self.client = c
        self.P = P
        self.P_shared = c.scatter(P)

    @staticmethod
    def run(rng, P):
        H = A_MCMC(P.iloc[rng], quit_after_burnin = True)
        H.run()
        return H

    def run_all(self, ranges):
        #
        # scatter across ranges. for each range, run until burnin
        futures = self.client.map(self.run, ranges, P = self.P_shared)
        results = self.client.gather(futures)

        #
        # concatenate burned in ranges; run again for full number of iterations

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
