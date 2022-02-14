import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.special as ss
import sortedcontainers as sc

from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB

from capy import seq

colors = mpl.cm.get_cmap("tab20").colors

# helper LogSumExp implementation
def LSE(x):
    lmax = np.max(x)
    return lmax + np.log(np.exp(x - lmax).sum())

# This wrapper class allows for multiple segmentation samples to be used as 
# seeds for consecutive CDP runs. Following the clustering of the first
# segmentation sample, subsequent CDP runs use the clustering of the previous run
# as a prior
class Coverage_DP:
    def __init__(self,
            segmentation_draws,
            beta,
            cov_df):
        self.segmentation_draws = segmentation_draws
        self.beta = beta
        self.cov_df = cov_df

        # number of seg samples to use and draws from each DP to take
        self.num_samples = None
        self.num_draws = None

        # Coverage DP run object for each seg sample
        self.DP_runs = None

        self.bins_to_clusters = None
        self.segs_to_clusters = None

    def run_dp(self, num_samples=-1, num_draws=50):

        if num_samples == -1:
            self.num_samples = self.segmentation_draws.shape[1]
        else:
            self.num_samples = num_samples
        self.num_draws = num_draws

        self.DP_runs = [None] * self.num_samples
        prior_run = None
        count_prior_sum = None

        # TODO: load segmentation samples randomly
        self.bins_to_clusters = []
        self.segs_to_clusters = []
        for samp in range(self.num_samples):
            print('starting sample {}'.format(samp))
            self.cov_df['segment_ID'] = self.segmentation_draws[:, samp].astype(int)

            DP_runner = Run_Cov_DP(self.cov_df.copy(), self.beta, prior_run, count_prior_sum)
            self.DP_runs[samp] = DP_runner
            draws, count_prior_sum = DP_runner.run(self.num_draws, samp)
            self.bins_to_clusters.append(draws)
            prior_run = DP_runner

    def visualize_DP_run(self, run_idx, save_path):
        if run_idx > len(self.DP_runs):
            raise ValueError('DP run index out of range')
        
        cov_dp = self.DP_runs[run_idx]
        cur = 0
        f, axs = plt.subplots(1, figsize = (25,10))
        for c in cov_dp.cluster_dict.keys():
            clust_start = cur
            for seg in cov_dp.cluster_dict[c]:
                len_seg = len(cov_dp.segment_r_list[seg])
                axs.scatter(np.r_[cur:len_seg+cur], np.exp(np.log(cov_dp.segment_r_list[seg]) - (cov_dp.segment_C_list[seg] @ cov_dp.beta).flatten()))
                cur += len_seg
            axs.add_patch(mpl.patches.Rectangle((clust_start,0), cur-clust_start, 500, fill=True, alpha=0.15, color = colors[c % 10]))
        plt.savefig(save_path)

# This class implements the actual DP clustering
# for now input will be coverage_df with global segment id column
class Run_Cov_DP:
    def __init__(self, cov_df, beta, prior_run=None, count_prior_sum=None):
        self.cov_df = cov_df
        self.seg_id_col = self.cov_df.columns.get_loc('segment_ID')
        self.beta = beta

        self.num_segments = self.cov_df.iloc[:, self.seg_id_col].max() + 1
        self.segment_r_list = [None] * self.num_segments
        self.segment_C_list = [None] * self.num_segments
        self.segment_counts = np.zeros(self.num_segments, dtype=int)
        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1

        self.cluster_counts = sc.SortedDict({})
        self.unassigned_segs = sc.SortedList(np.r_[0:self.num_segments])
        self.cluster_dict = sc.SortedDict({})
        self.cluster_MLs = sc.SortedDict({})
        self.greylist_segments = sc.SortedSet({})
        self.cluster_LLs = sc.SortedDict({})

        self.prior_clusters = None
        self.prior_r_list = None
        self.prior_C_list = None
        self.count_prior_sum = None
        # for saving init clusters
        self.init_clusters = None

        self._init_segments()
        self._init_clusters(prior_run, count_prior_sum)

        # containers for saving the MCMC trace
        self.clusters_to_segs = []
        self.bins_to_clusters = []
        self.ll_history = []
        self.alpha = 0.1

    # initialize each segment object with its data
    def _init_segments(self):
        for ID, seg_df in self.cov_df.groupby('segment_ID'):
            self.segment_r_list[ID] = seg_df['covcorr'].values
            if 'C_len' in self.cov_df.columns:
                self.segment_C_list[ID] = np.c_[np.log(seg_df["C_len"]), seg_df["C_RT_z"], seg_df["C_GC_z"]]
            else:
                self.segment_C_list[ID] = np.c_[seg_df["C_RT_z"], seg_df["C_GC_z"]]
            self.segment_counts[ID] = len(self.segment_r_list[ID])

    def _init_clusters(self, prior_run, count_prior_sum):
        # if first iteration then add first segment to first cluster
        if prior_run is None:
            self.cluster_counts[0] = self.segment_counts[0]
            self.unassigned_segs.discard(0)
            self.cluster_dict[0] = sc.SortedSet([0])
            self.cluster_MLs[0] = self._ML_cluster([0])
            # next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = 1
        else:
            #otherwise we initialize the prior clusters
            self.prior_clusters = prior_run.cluster_dict.copy()
            self.prior_r_list = prior_run.segment_r_list.copy()
            self.prior_C_list = prior_run.segment_C_list.copy()
            self.count_prior_sum = count_prior_sum.copy()
            self.next_cluster_index = np.r_[self.prior_clusters.keys()].max() + 1
            self.clust_prior_ML = None
    
    @staticmethod
    def ll_nbinom(r, mu, C, beta, lepsi):
        r = r.flatten()
        epsi = np.exp(lepsi)
        bc = (C @ beta).flatten()
        exp = np.exp(mu + bc).flatten()
        return (ss.gammaln(r + epsi) - ss.gammaln(r + 1) - ss.gammaln(epsi) +
                (r * (mu + bc - np.log(epsi + exp))) +
                (epsi * np.log(epsi / (epsi + exp)))).sum()

    # main worker function for computing marginal likelihoods of clusters
    def _ML_cluster(self, cluster_set):
        # aggregate r and C arrays
        r = np.hstack([self.segment_r_list[i] for i in cluster_set])
        C = np.concatenate([self.segment_C_list[i] for i in cluster_set])
        mu_opt, lepsi_opt, H_opt = self.stats_optimizer(r, C, ret_hess=True)
        ll_opt = self.ll_nbinom(r, mu_opt, C, self.beta, lepsi_opt)

        # print('clust set: ', cluster_set, 'll: ', ll_opt, 'lap approx: ', self._get_laplacian_approx(H_opt))
        return ll_opt + self._get_laplacian_approx(H_opt)

    # computes the log likelihood for a set of segments comprising a cluster
    def _LL_cluster(self, cluster_set):
        # aggregate r and C arrays
        r = np.hstack([self.segment_r_list[i] for i in cluster_set])
        C = np.concatenate([self.segment_C_list[i] for i in cluster_set])
        mu_opt, lepsi_opt = self.stats_optimizer(r, C, ret_hess=False)
        ll_opt = self.ll_nbinom(r, mu_opt, C, self.beta, lepsi_opt)

        return ll_opt

    # computes the ML of a cluster with some clusters optionally containing 
    # segments from previous iterations (i.e. prior clustering)
    def _ML_cluster_prior(self, cluster_set, new_segIDs=None):
        # aggregate r and C arrays
        if new_segIDs is not None:
            r_new = np.hstack([self.segment_r_list[s] for s in new_segIDs])
            C_new = np.concatenate([self.segment_C_list[s] for s in new_segIDs])
            r = np.r_[np.hstack([self.prior_r_list[i] for i in cluster_set]), r_new]
            C = np.r_[np.concatenate([self.prior_C_list[i] for i in cluster_set]), C_new]
        else:
            r = np.hstack([self.prior_r_list[i] for i in cluster_set])
            C = np.concatenate([self.prior_C_list[i] for i in cluster_set])

        mu_opt, lepsi_opt, H_opt = self.stats_optimizer(r, C, ret_hess=True)
        ll_opt = self.ll_nbinom(r, mu_opt, C, self.beta, lepsi_opt)
        return ll_opt + self._get_laplacian_approx(H_opt)

    @staticmethod
    def _get_laplacian_approx(H):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-H))) / 2

    # returns optimal NB parameter values, along with optionally the hessian
    # from the MLE point
    def stats_optimizer(self, r, C, ret_hess=False):
        offset = (C @ self.beta).flatten()
        exog = np.ones(r.shape[0])
        sNB = statsNB(r, exog, offset=offset)
        res = sNB.fit(disp=0)
        if ret_hess:
            return res.params[0], -np.log(res.params[1]), sNB.hessian(res.params)
        else:
            return res.params[0], -np.log(res.params[1])

    #TODO: switch to new DP prior
    def DP_merge_prior(self, cur_cluster):
        cur_index = self.cluster_counts.index(cur_cluster)
        cluster_vals = np.array(self.cluster_counts.values())
        N = cluster_vals.sum()
        M = cluster_vals[cur_index]
        DP_prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            if i != cur_index:
                DP_prior_results[i] = ss.loggamma(M + nc) + ss.loggamma(N + self.alpha - M) - (
                            ss.loggamma(nc) + ss.loggamma(N + self.alpha))
            else:
                # the prior prob of remaining in the current cluster is the same as for joining a new cluster
                DP_prior_results[i] = ss.gammaln(M) + np.log(self.alpha) + ss.gammaln(N + self.alpha - M) - ss.gammaln(
                    N + self.alpha)
        return DP_prior_results

    def DP_tuple_split_prior(self, seg_id):
        cur_cluster = self.cluster_assignments[seg_id]
        seg_size = self.segment_counts[seg_id]
        cluster_vals = np.array(self.cluster_counts.values())

        if cur_cluster > -1:
            # exclude the points were considering moving from the dp calculation
            # if the segment was already in a cluster
            cur_index = self.cluster_counts.index(cur_cluster)
            cluster_vals[cur_index] -= seg_size

        N = cluster_vals.sum()

        loggamma_N_alpha = ss.loggamma(N + self.alpha)
        loggamma_N_alpha_seg = ss.loggamma(N + self.alpha - seg_size)
        DP_prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            DP_prior_results[i] = ss.loggamma(seg_size + nc) + loggamma_N_alpha_seg - (ss.loggamma(nc) + loggamma_N_alpha)

        # the prior prob of starting a new cluster
        DP_prior_new = ss.gammaln(seg_size) + np.log(self.alpha) + loggamma_N_alpha_seg - loggamma_N_alpha
        return np.r_[DP_prior_results, DP_prior_new]

    # if we have prior assignments from the last iteration we can use those clusters to probalistically assign
    # each segment into a old cluster
    def initial_prior_assignment(self, count_prior):
        for segID in range(self.num_segments):
            # compute MLs of segment joining each prior cluster with the current r and C for the segID and old r and C
            # lists for the previous segmentation.
            BC = np.r_[
                [self._ML_cluster_prior(self.prior_clusters[c], [segID]) for c in self.prior_clusters.keys()]]
            S = self._ML_cluster([segID])
            C = np.r_[self.clust_prior_ML.values()]

            # prior liklihood ratios
            P_l = BC - (S + C)
            # get count prior
            ccp = count_prior / count_prior.sum()

            # posterior numerator
            num = P_l + np.log(ccp)
            num = np.nan_to_num(num)
            num -= num.max()

            # probabilitically choose a cluster
            probs = np.exp(num) / np.exp(num).sum()
            idx = np.r_[self.prior_clusters.keys()]
            choice = np.random.choice(idx, p=probs)

            # make assignment
            self.cluster_assignments[segID] = choice
            if choice not in self.cluster_counts:
                self.cluster_counts[choice] = 1
                self.cluster_dict[choice] = sc.SortedSet({})
            else:
                self.cluster_counts[choice] += 1

            self.cluster_dict[choice].add(segID)
            self.cluster_MLs[choice] = self._ML_cluster(self.cluster_dict[choice])
        self.init_clusters = sc.SortedDict({k: v.copy() for k, v in self.cluster_dict.items()})

    def run(self, n_iter, sample_num=0):

        burned_in = False
        all_assigned = False

        n_it = 0
        n_it_last = 0

        if self.prior_clusters:
            self.clust_prior_ML = sc.SortedDict(
                {k: self._ML_cluster_prior(self.prior_clusters[k]) for k in self.prior_clusters.keys()})
            count_prior = np.r_[[v for (k, v) in self.count_prior_sum.items() if
                                 k in self.prior_clusters.keys()]] / sample_num
            self.initial_prior_assignment(count_prior)
        
        white_segments = set(range(self.num_segments)) - self.greylist_segments
        while len(self.bins_to_clusters) < n_iter:
    
            # status update
            if not n_it % 250 and self.prior_clusters is None:
                print("n unassigned: {}".format(len(self.unassigned_segs)))

            # start couting for burn in
            if not n_it % 100:
                if not all_assigned and (self.cluster_assignments[white_segments] > -1).all():
                    all_assigned = True
                    n_it_last = n_it
                    #start keeping track of the total likelihood after all have been assigned
                    for ID in self.cluster_counts.keys():
                        self.cluster_LLs[ID] = self._LL_cluster(self.cluster_dict[ID])
                
                # burn in based on ll changes after all are assigned
                if not burned_in and all_assigned and n_it - n_it_last > 500:
                    if np.diff(np.r_[self.ll_history[-500:]]).mean() <= 0:
                        print('burnin')
                        burned_in = True
                        n_it_last = n_it

            # pick either a segment or a cluster at random

            # pick segment
            if np.random.rand() < 0.5:
                if len(self.unassigned_segs) > 0 and len(
                        self.unassigned_segs) / self.num_segments < 0.1 and np.random.rand() < 0.5:
                    segID = np.random.choice(self.unassigned_segs)
                else:
                    segID = np.random.choice(white_segments)
                
                # get cluster assignment of S
                clustID = self.cluster_assignments[segID]
                # compute ML of AB = Cs (cached)
                if clustID == -1:
                    ML_AB = 0
                else:
                    ML_AB = self.cluster_MLs[clustID]

                # compute ML of A = Cs - S
                if clustID == -1:
                    ML_A = 0
                # if cluster is empty without S ML is also 0
                elif self.cluster_counts[clustID] == 1:
                    ML_A = 0
                else:
                    ML_A = self._ML_cluster(self.cluster_dict[clustID].difference([segID]))

                # compute ML of S on its own
                ML_S = self._ML_cluster([segID])

                # compute ML of every other cluster C = Ck, k != s (cached)
                # for now were also allowing it to chose to stay in current cluster

                ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])

                # compute ML of every cluster if S joins
                ML_BC = np.array([self._ML_cluster(self.cluster_dict[k].union([segID]))
                                  for k in self.cluster_counts.keys()])
                # likelihood ratios of S joining each other cluster S -> Ck
                ML_rat_BC = ML_A + ML_BC - (ML_AB + ML_C)

                # if cluster is unassigned we set the ML ratio to 1 for staying in its own cluster

                if clustID > -1:
                    ML_rat_BC[list(self.cluster_counts.keys()).index(clustID)] = 0

                # compute ML of S starting a new cluster
                ML_new = ML_A + ML_S - ML_AB

                ML_rat = np.r_[ML_rat_BC, ML_new]

                prior_diff = []
                clust_prior_p = 1
                # use prior cluster information if available
                if self.prior_clusters:
                    # divide prior into three sections:
                    # * clusters in prior not currently active (if picked, will open a new cluster with that ID)
                    # * clusters in prior currently active (if picked, will weight that cluster's posterior probability)
                    # * currently active clusters not in the prior (if picked, would weight cluster's posterior
                    #   probability with prior probability of making brand new cluster)

                    # not currently active
                    prior_diff = self.prior_clusters.keys() - self.cluster_counts.keys()

                    # currently active clusters in prior
                    prior_com = self.prior_clusters.keys() & self.cluster_counts.keys()

                    # currently active clusters not in prior
                    prior_null = self.cluster_counts.keys() - self.prior_clusters.keys()

                    # order of prior vector:
                    # [-1 (totally new cluster), <prior_diff>, <prior_com + prior_null>]
                    prior_idx = np.r_[
                        np.r_[[self.prior_clusters.index(x) for x in prior_diff]],
                        np.r_[[self.prior_clusters.index(x) if x in self.prior_clusters else -1 for x in
                               (prior_com | prior_null | {self.next_cluster_index})]]
                    ].astype(int)

                    prior_MLs = np.r_[[self._ML_cluster_prior(self.prior_clusters[self.prior_clusters.keys()[idx]],
                                                              [segID]) if idx != -1 else 0 for idx in prior_idx]] - (
                                        self._ML_cluster([segID]) + np.r_[[self.clust_prior_ML[
                                                                               self.prior_clusters.keys()[
                                                                                   idx]] if idx != -1 else - self._ML_cluster(
                                    [segID]) for idx in prior_idx]])
                    clust_prior_p = np.maximum(
                        np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                    # expand MLs to account for multiple new clusters
                    ML_rat = np.r_[np.full(len(prior_diff), ML_rat[-1]), ML_rat]
                # DP prior based on clusters sizes
                count_prior = np.r_[
                    [np.log(count_prior[self.prior_clusters.index(x)]) for x in
                     prior_diff], self.DP_tuple_split_prior(segID)]

                # construct transition probability distribution and draw from it
                MLs_max = (ML_rat + count_prior).max()
                choice_p = np.exp(ML_rat + count_prior - MLs_max + np.log(clust_prior_p)) / np.exp(
                    ML_rat + count_prior - MLs_max + np.log(clust_prior_p)).sum()

                if np.isnan(choice_p.sum()):
                    print('skipping iteration {} due to nan. picked segment {}'.format(n_it, segID))
                    n_it += 1
                    continue
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                choice = np.r_[-np.r_[prior_diff] - 1,
                               self.cluster_counts.keys(), self.next_cluster_index][choice_idx]
                choice = int(choice)

                # if choice == next_cluster_idx then start a brand new cluster with this new index
                # if choice < 0 then we create a new cluster with the index of the old cluster
                if choice == self.next_cluster_index or choice < 0:
                    old_clust = False
                    if choice < 0:
                        # since its an old cluster we correct its index
                        choice = -(choice + 1)
                        old_clust = True

                    if clustID > -1:
                        # if the segment used to occupy a cluster by itself, do nothing if its a new cluster
                        # need to rename it if its a old cluster
                        if self.cluster_counts[clustID] == 1:
                            if old_clust:
                                # rename clustID to choice
                                self.cluster_counts[choice] = self.cluster_counts[clustID]
                                self.cluster_dict[choice] = self.cluster_dict[clustID].copy()
                                self.cluster_assignments[segID] = choice
                                self.cluster_MLs[choice] = self.cluster_MLs[clustID]
                                
                                # delete obsolete cluster
                                del self.cluster_counts[clustID]
                                del self.cluster_dict[clustID]
                                del self.cluster_MLs[clustID]
                                
                                if all_assigned:
                                    self.cluster_LLs[choice] = self._LL_cluster[clustID].copy()
                                    del self.cluster_LLs[clustID]
                            n_it += 1
                            continue
                        else:
                            # if seg was previously assigned remove it from previous cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_dict[clustID].discard(segID)

                            self.cluster_MLs[clustID] = ML_A
                            if all_assigned:
                                self.cluster_LLs[clustID] = self._LL_cluster(self.cluster_dict[clustID]) 
                    else:
                        self.unassigned_segs.discard(segID)

                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] = self.segment_counts[segID]
                    self.cluster_dict[choice] = sc.SortedSet([segID])
                    self.cluster_MLs[choice] = ML_S
                    self.next_cluster_index += 1
                    if all_assigned:
                        self.cluster_LLs[choice] = self._LL_cluster(self.cluster_dict[choice])
                else:
                    # if remaining in same cluster, skip
                    if clustID == choice:
                        n_it += 1
                        continue

                    # joining existing cluster

                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] += self.segment_counts[segID]
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]
                    if all_assigned:
                        self.cluster_LLs[choice] = self._LL_cluster(self.cluster_dict[choice])

                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if self.cluster_counts[clustID] == 1:
                            del self.cluster_counts[clustID]
                            del self.cluster_dict[clustID]
                            del self.cluster_MLs[clustID]
                            if all_assigned:
                                del self.cluster_LLs[clustID]
                        else:
                            # otherwise update former cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
                            if all_assigned:
                                self.cluster_LLs[clustID] = self._LL_cluster(self.cluster_dict[clustID])
                    else:
                        self.unassigned_segs.discard(segID)

            # pick cluster to merge
            else:
                # it only makes sense to try joining two clusters if there are at least two of them!
                if len(self.cluster_counts) < 2:
                    n_it += 1
                    continue

                clust_pick = np.random.choice(self.cluster_dict.keys())
                clust_pick_segs = np.r_[self.cluster_dict[clust_pick]].astype(int)
                # get ML of this cluster merged with each of the other existing clusters
                ML_join = [
                    self._ML_cluster(self.cluster_dict[i].union(clust_pick_segs)) if i != clust_pick else
                    self.cluster_MLs[i] for i in self.cluster_dict.keys()]
                # we need to compare this ML with the ML of leaving the target cluster and the picked cluster on their own
                ML_split = np.array(self.cluster_MLs.values()) + self.cluster_MLs[clust_pick]
                ML_split[self.cluster_MLs.keys().index(clust_pick)] = self.cluster_MLs[clust_pick]
                ML_rat = np.array(ML_join) - ML_split

                prior_diff = []
                clust_prior_p = 1
                if self.prior_clusters:
                    prior_diff = self.prior_clusters.keys() - self.cluster_counts.keys()
                    # currently active clusters in prior
                    prior_com = self.prior_clusters.keys() & self.cluster_counts.keys()
                    # currently active clusters not in prior
                    prior_null = self.cluster_counts.keys() - self.prior_clusters.keys()
                    # order of prior vector:
                    # [-1 (totally new cluster), <prior_diff>, <prior_com + prior_null>]
                    prior_idx = np.r_[
                        np.r_[[self.prior_clusters.index(x) for x in prior_diff]],
                        np.r_[[self.prior_clusters.index(x) if x in self.prior_clusters else -1 for x in
                               (prior_com | prior_null)]]
                    ].astype(int)

                    prior_MLs = np.r_[[self._ML_cluster_prior(self.prior_clusters[self.prior_clusters.keys()[idx]], clust_pick_segs) if idx != -1 else 0 for idx in prior_idx]] - (self.cluster_MLs[clust_pick] + np.r_[
                                    [self.clust_prior_ML[self.prior_clusters.keys()[idx]] if idx != -1 else -self.cluster_MLs[clust_pick] for idx in prior_idx]])

                    clust_prior_p = np.maximum(
                        np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                    # expand MLs to account for multiple new merge clusters--which have liklihood = cluster staying as is = 0
                    ML_rat = np.r_[np.full(len(prior_diff), 0), ML_rat]
                    # DP prior based on clusters sizes now with no alpha
                count_prior = np.r_[
                    [np.log(count_prior[self.prior_clusters.index(x)]) for x in prior_diff],
                    self.DP_merge_prior(clust_pick)]

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()
                if np.isnan(choice_p.sum()):
                    print("skipping iteration {} due to nan".format(n_it))
                    n_it += 1
                    continue

                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                choice = np.r_[-np.r_[prior_diff] - 1, self.cluster_counts.keys()][choice_idx]
                choice = int(choice)

                if choice != clust_pick:
                    if choice < 0:
                        # we're merging into an old cluster, which we do by simply reindexing that cluster
                        choice = -(choice + 1)
                        # rename clustID to choice
                        self.cluster_counts[choice] = self.cluster_counts[clust_pick]
                        self.cluster_dict[choice] = self.cluster_dict[clust_pick].copy()
                        self.cluster_assignments[clust_pick_segs] = choice
                        self.cluster_MLs[choice] = self.cluster_MLs[clust_pick]
                        
                        if all_assigned:
                            self.cluster_LLs[choice] = self.cluster_LLs[clust_pick].copy()
                            del self.cluster_LLs[clust_pick]
                        
                        # delete obsolete cluster
                        del self.cluster_counts[clust_pick]
                        del self.cluster_dict[clust_pick]
                        del self.cluster_MLs[clust_pick]
                    else:
                        # we need to merge clust_pick and choice_idx which we do by merging to the cluster with more
                        # segments
                        tup = (clust_pick, choice)
                        larger_cluster = np.argmax([self.cluster_counts[clust_pick], self.cluster_counts[choice]])
                        merged_ID = tup[larger_cluster]
                        vacatingID = tup[int(not larger_cluster)]

                        # move vacatingID to newclustID
                        # update new cluster with additional segments
                        vacating_segs = self.cluster_dict[vacatingID]
                        self.cluster_assignments[vacating_segs] = merged_ID
                        self.cluster_counts[merged_ID] += self.cluster_counts[vacatingID]
                        self.cluster_dict[merged_ID] = self.cluster_dict[merged_ID].union(vacating_segs)
                        self.cluster_MLs[merged_ID] = ML_join[list(self.cluster_counts.keys()).index(choice)]
                        
                        if all_assigned:
                            self.cluster_LLs[merged_ID] = self._LL_cluster(self.cluster_dict[merged_ID])
                            del self.cluster_LLs[vacatingID]
                        # delete last cluster
                        del self.cluster_counts[vacatingID]
                        del self.cluster_dict[vacatingID]
                        del self.cluster_MLs[vacatingID]


            # save draw after burn in for every n_seg / (n_clust / 2) iterations
            if burned_in and n_it - n_it_last > self.num_segments / (len(self.cluster_counts) * 2):
            # if burned_in and n_it - n_it_last > self.num_segments:
                self.bins_to_clusters.append(self.cluster_assignments.copy())
                self.clusters_to_segs.append(self.cluster_dict.copy())
                n_it_last = n_it

            n_it += 1
            if all_assigned:
                self.ll_history.append(np.r_[self.cluster_LLs.values()].sum())
        # add the cluster counts from the last draw to the running total
        if self.count_prior_sum is None:
            self.count_prior_sum = self.cluster_counts

        else:
            for k, v in self.cluster_counts.items():
                if k in self.count_prior_sum.keys():
                    self.count_prior_sum[k] += v
                else:
                    self.count_prior_sum[k] = v

        # return the clusters from the last draw and the counts
        return self.bins_to_clusters, self.count_prior_sum
