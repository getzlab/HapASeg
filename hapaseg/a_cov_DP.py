import colorama
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc
import h5py

from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB

from capy import seq

colors = mpl.cm.get_cmap("tab20").colors


def LSE(x):
    lmax = np.max(x)
    return lmax + np.log(np.exp(x - lmax).sum())


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

            DP_runner = Run_Cov_DP(self.cov_df, self.beta, prior_run, count_prior_sum)
            self.DP_runs[samp] = DP_runner
            draws, count_prior_sum = DP_runner.run(self.num_draws, samp)
            self.bins_to_clusters.append(draws)
            prior_run = DP_runner

    def visualize_DP_run(self, run_idx, save_path):
        if run_idx > len(self.DP_runs):
            raise ValueError('DP run index out of range')

        cov_dp = self.DP_runs[run_idx]
        cur = 0
        f, axs = plt.subplots(1, figsize=(25, 10))
        for c in cov_dp.cluster_dict.keys():
            clust_start = cur
            for seg in cov_dp.cluster_dict[c]:
                len_seg = len(cov_dp.segment_r_list[seg])
                axs.scatter(np.r_[cur:len_seg + cur], np.exp(
                    np.log(cov_dp.segment_r_list[seg]) - (cov_dp.segment_C_list[seg] @ cov_dp.beta).flatten()))
                cur += len_seg
            axs.add_patch(mpl.patches.Rectangle((clust_start, 0), cur - clust_start, 500, fill=True, alpha=0.15,
                                                color=colors[c % 10]))
        plt.savefig(save_path)
# for now input will be coverage_df with global segment id column
class Run_Cov_DP:
    def __init__(self, cov_df, beta, coverage_prior=True, seed_all_clusters = True, prior_run=None, count_prior_sum=None):
        self.cov_df = cov_df
        self.beta = beta
        self.seed_all_clusters = seed_all_clusters
        self.num_segments = len(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        self.segment_r_list = [None] * self.num_segments
        self.segment_V_list = [None] * self.num_segments
        self.segment_sigma_list = [None] * self.num_segments
        self.segment_counts = np.zeros(self.num_segments, dtype=int)
        self.segment_cov_bins = np.zeros(self.num_segments, dtype=int)
        self.segment_allele = np.zeros(self.num_segments, dtype=int)
        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1
        
        self.cluster_counts = sc.SortedDict({})
        self.unassigned_segs = sc.SortedList(np.r_[0:self.num_segments])
        self.cluster_dict = sc.SortedDict({})
        self.cluster_MLs = sc.SortedDict({})
        self.greylist_segments = sc.SortedSet({})
        self.cluster_datapoints = sc.SortedDict({})

        self.prior_clusters = None
        self.prior_r_list = None
        self.prior_C_list = None
        self.count_prior_sum = None
        self.ML_total_history = []
        self.DP_total_history = []
        self.MLDP_total_history = []
        self.tmp_ml_jump_history = []        

        self.cdict_history = []

        self.coverage_prior = coverage_prior
        # for saving init clusters
        self.init_clusters = None
    
        self._init_segments()
        self._init_clusters(prior_run, count_prior_sum)

        # containers for saving the MCMC trace_cov_dp
        self.clusters_to_segs = []
        self.bins_to_clusters = []
        self.draw_indices = []
        self.alpha = 0.5

    # initialize each segment object with its data
    def _init_segments(self):
        fallback_counts = sc.SortedDict({})
        for ID, (name, grouped) in enumerate(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw'])):
            mu = grouped['cov_DP_mu'].values[0]
            sigma = grouped['cov_DP_sigma'].values[0]
            group_len = len(grouped)
            if group_len > 10:
                major, minor =  (grouped['maj_count'].sum(), grouped['min_count'].sum())
            else:
                ADP_clust = grouped['allelic_cluster'].values[0]
                if ADP_clust in fallback_counts:
                    major, minor = fallback_counts[ADP_clust]
                else:
                    filt = self.cov_df.loc[self.cov_df.allelic_cluster == ADP_clust]
                    major, minor = filt['maj_count'].sum(), filt['min_count'].sum()
                    fallback_counts[ADP_clust] = (major, minor)
                    
            allele = grouped.allele.values[0]
            self.segment_allele[ID] = allele
            if group_len < 3:
                self.greylist_segments.add(ID)
            
            if allele== -1:
                f = minor / (minor + major)
                a=minor;b=major
            else:
                f =  major / (minor + major)
                a=major;b=minor
            r = np.array(np.exp(s.norm.rvs(mu, np.sqrt(sigma), size=group_len)) * s.beta.rvs(a,b, size=group_len))

            C = np.c_[np.log(grouped["C_len"]), grouped["C_RT_z"], grouped["C_GC_z"]]
            x = grouped['covcorr']

            V = (np.exp(s.norm.rvs(mu, np.sqrt(sigma), size=10000)) * s.beta.rvs(a,b, size=10000)).var()
            
            if np.sqrt(V) > 4.5:
                self.greylist_segments.add(ID) 
            self.segment_V_list[ID] = V
            self.segment_r_list[ID] = r 
            self.segment_cov_bins[ID] = group_len
            if self.coverage_prior:
                self.segment_counts[ID] = group_len
            else:
                self.segment_counts[ID] = 1

    def _init_clusters(self, prior_run, count_prior_sum):
        [self.unassigned_segs.discard(s) for s in self.greylist_segments]
        if not self.seed_all_clusters:
            first = (set(range(self.num_segments)) - self.greylist_segments)[0]
            clusterID = 0
            self.cluster_counts[0] = self.segment_counts[0]
            self.unassigned_segs.discard(0)
            self.cluster_dict[0] = sc.SortedSet([first])
            self.cluster_MLs[0] = self._ML_cluster_from_list([first])
            self.cluster_assignments[first] = 0
            self.cluster_datapoints[0] = self.segment_r_list[first].copy()
            #next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = 1
        else: 
            for i in set(range(self.num_segments)) - self.greylist_segments:
                self.cluster_counts[i] = self.segment_counts[i]
                self.unassigned_segs.discard(i)
                self.cluster_dict[i] = sc.SortedSet([i])
                self.cluster_MLs[i] = self._ML_cluster_from_list([i])
                self.cluster_assignments[i] = i
                self.cluster_datapoints[i] = self.segment_r_list[i].copy()
            #next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = i+1
    
    # datapoint array generation methods
    
    # generate cluster from list of segment IDs
    def _cluster_gen_from_list(self, cluster_list):
        r_lst = []
        for s in cluster_list:
            r_lst.append(self.segment_r_list[s])
        r = np.hstack(r_lst)
        return r
    
    def _cluster_gen_add_one(self, clusterID, segID):
        return np.concatenate([ self.cluster_datapoints[clusterID], self.segment_r_list[segID]], axis=0)
    
    #assumes the datapoints are ordered by segment ID
    def _cluster_gen_remove_one(self, clusterID, segID):
        cur = self.cluster_datapoints[clusterID]
        seg_ind = self.cluster_dict[clusterID].index(segID)
        st = self.segment_cov_bins[:seg_ind].sum()
        en = st + self.segment_cov_bins[segID]
        return np.concatenate([cur[:st], cur[en:]], axis=0)

    def _cluster_gen_merge(self, clust_A, clust_B):
        return np.concatenate([self.cluster_datapoints[clust_A], self.cluster_datapoints[clust_B]], axis=0)
    
    def _ML_cluster_from_r(self, r):
        alpha = 50
        beta = alpha/2 * 45
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def _ML_cluster_from_list(self, cluster_list):
        r = self._cluster_gen_from_list(cluster_list)
        alpha = 50
        beta = alpha/2 * 45
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)
    
    def _ML_cluster_add_one(self, clusterID, segID):
        r = self._cluster_gen_add_one(clusterID, segID)
        alpha = 50
        beta = alpha/2 * 45
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)
    
    def _ML_cluster_remove_one(self, clusterID, segID):
        r = self._cluster_gen_remove_one(clusterID, segID)
        alpha = 50
        beta = alpha/2 * 45
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def _ML_cluster_merge(self, clust_A, clust_B): 
        r = self._cluster_gen_merge(clust_A, clust_B)
        alpha = 50
        beta = alpha/2 * 45
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def ML_normalgamma(self, x, mu0, kappa0, alpha0, beta0):
        #for now x_mean is the same as mu0
        x_mean = mu0
        n = len(x)

        mu_n = (kappa0*mu0 + n * x_mean) / (kappa0 + n)
        kappa_n = kappa0 + n
        alpha_n = alpha0 + n/2
        beta_n = beta0 + 0.5 * ((x - x_mean)**2).sum() + kappa0 * n * (x_mean - mu0)**2 / 2*(kappa0 + n)

        return ss.loggamma(alpha_n) - ss.loggamma(alpha0) + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n) + np.log(kappa0 / kappa_n) / 2 - n * np.log(2*np.pi) / 2

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

    def optimize_gaussian(self, r):
        mu = r.mean()
        sigma2 = r.var()
        N = len(r)

        sq_difs_sum = ((r - mu) ** 2).sum()

        # hess computations
        mu_mu = - N / sigma2
        mu_sigma2 = - 1 / N * (r - mu).sum()
        sigma2_sigma2 = N / (2 * sigma2 ** 2) - sq_difs_sum / (sigma2 ** 3)

        ll = -(N / 2) * (np.log(2 * np.pi * sigma2)) - sq_difs_sum / (2 * sigma2)
        H = np.r_[np.c_[mu_mu, mu_sigma2], np.c_[mu_sigma2, sigma2_sigma2]]

        laplacian = self._get_laplacian_approx(H)
        return ll + laplacian
    def save_ML_total(self):
        ML_tot = np.r_[self.cluster_MLs.values()].sum()
        self.ML_total_history.append(ML_tot)
        
        num_clusts = len(self.cluster_dict)
        N = sum(self.cluster_counts.values())
        DP_tot = num_clusts * np.log(self.alpha) + sum([ss.gammaln(na) for na in self.cluster_counts.values()]) - ss.gammaln(self.alpha + N) + ss.gammaln(self.alpha)
        self.DP_total_history.append(DP_tot)
        self.MLDP_total_history.append(ML_tot + DP_tot)

    def DP_merge_prior(self, cur_cluster):
        cur_index = self.cluster_counts.index(cur_cluster)
        cluster_vals = np.array(self.cluster_counts.values())
        N = cluster_vals.sum()
        M = cluster_vals[cur_index]
        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            if i != cur_index:
                prior_results[i] = ss.loggamma(M + nc) + ss.loggamma(N + self.alpha - M) - (ss.loggamma(nc) + ss.loggamma(N + self.alpha))
            else:
                #the prior prob of remaining in the current cluster is the same as for joining a new cluster
                prior_results[i] = ss.gammaln(M) + np.log(self.alpha) + ss.gammaln(N + self.alpha - M) - ss.gammaln(N + self.alpha)
        return prior_results

    def DP_tuple_split_prior(self, seg_id):
        cur_cluster = self.cluster_assignments[seg_id]
        seg_size = len(self.segment_r_list[seg_id])
        cluster_vals = np.array(self.cluster_counts.values())
        
        if cur_cluster > -1:
            # exclude the points were considering moving from the dp calculation
            # if the tuple was already in a cluster
            cur_index = self.cluster_counts.index(cur_cluster)
            cluster_vals[cur_index] -= seg_size
        
        N = cluster_vals.sum()

        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            prior_results[i] = ss.loggamma(seg_size + nc) + ss.loggamma(N + self.alpha - seg_size) - (ss.loggamma(nc) + ss.loggamma(N + self.alpha))
   
            #the prior prob of starting a new cluster
            prior_new = ss.gammaln(seg_size) + np.log(self.alpha) + ss.gammaln(N + self.alpha - seg_size) - ss.gammaln(N + self.alpha)
        return np.r_[prior_results, prior_new]

    # since were taking the ratio we can remove the final two terms:
    # ss.gammaln(self.alpha + N - M) - ss.gammaln(self.alpha + N)
    # from both split and stay
    def DP_split_prior(self, split_A_segs, split_B_segs):
        n_a = self.segment_cov_bins[split_A_segs].sum()
        n_b = self.segment_cov_bins[split_B_segs].sum()
        M = n_a + n_b
        split = 2 * np.log(self.alpha) + ss.gammaln(n_a) + ss.gammaln(n_b)
        stay = np.log(self.alpha) + ss.gammaln(M - 1)
        return split - stay

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
        # while n_it < n_iter:
        while len(self.bins_to_clusters) < n_iter:
            
            self.save_ML_total()
            # status update
            if not n_it % 250 and self.prior_clusters is None:
                print("n unassigned: {}".format(len(self.unassigned_segs)))
            
            # start couting for burn in
            if not n_it % 100:
                self.cdict_history.append(self.cluster_dict.copy())
                if not all_assigned and (self.cluster_assignments[white_segments] > -1).all():
                    all_assigned = True
                    n_it_last = n_it

                # burn in after n_seg / n_clust iteration
                if not burned_in and all_assigned and n_it - n_it_last > max(2000, self.num_segments):
                    if np.diff(np.r_[self.MLDP_total_history[-2000:]]).mean() <= 0:
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
                elif len(self.cluster_dict[clustID]) == 1:
                    ML_A = 0
                else:
                    #ML_A = self._ML_cluster(self.cluster_dict[clustID].difference([segID]))
                    ML_A = self._ML_cluster_remove_one(clustID, segID)
                # compute ML of S on its own
                ML_S = self._ML_cluster_from_list([segID])

                # compute ML of every other cluster C = Ck, k != s (cached)
                # for now were also allowing it to chose to stay in current cluster

                ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])

                # compute ML of every cluster if S joins
                #ML_BC = np.array([self._ML_cluster(self.cluster_dict[k].union([segID]))
                #                  for k in self.cluster_counts.keys()])
                ML_BC = np.array([self._ML_cluster_add_one(k, segID) for k in self.cluster_counts.keys()])
                
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
                #count_prior = np.r_[
                #    [count_prior[self.prior_clusters.index(x)] for x in
                #     prior_diff], self.cluster_counts.values(), self.segment_counts[segID] * self.alpha]
                #count_prior /= count_prior.sum()

                # currently we do not support prior draws here
                # construct transition probability distribution and draw from it
                log_count_prior = self.DP_tuple_split_prior(segID)
                MLs_max = (ML_rat + log_count_prior).max()
                choice_p = np.exp(ML_rat + log_count_prior - MLs_max + np.log(clust_prior_p)) / np.exp(
                    ML_rat + log_count_prior - MLs_max + np.log(clust_prior_p)).sum()
                if np.isnan(choice_p.sum()):
                    print('skipping iteration {} due to nan. picked segment {}'.format(n_it, segID))
                    n_it += 1
                    continue
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )
                #self.tmp_ml_jump_history.append((ML_rat + log_count_prior - MLs_max)[choice_idx])

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
                        if len(self.cluster_dict[clustID]) == 1:
                            if old_clust:
                                # rename clustID to choice
                                self.cluster_counts[choice] = self.cluster_counts[clustID].copy()
                                self.cluster_dict[choice] = self.cluster_dict[clustID].copy()
                                self.cluster_assignments[segID] = choice
                                self.cluster_MLs[choice] = self.cluster_MLs[clustID]
                                # delete obsolete cluster
                                del self.cluster_counts[clustID]
                                del self.cluster_dict[clustID]
                                del self.cluster_MLs[clustID]
                            n_it += 1
                            continue
                        else:
                            # seg was previously assigned so remove it from previous cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_datapoints[clustID] = self._cluster_gen_remove_one(clustID, segID)
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
                    else:
                        self.unassigned_segs.discard(segID)

                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] = self.segment_counts[segID]
                    self.cluster_dict[choice] = sc.SortedSet([segID])
                    self.cluster_datapoints[choice] = self.segment_r_list[segID].copy()
                    self.cluster_MLs[choice] = ML_S
                    self.next_cluster_index += 1
                else:
                    # if remaining in same cluster, skip
                    if clustID == choice:
                        #   print('{} staying in current cluster'.format(clustID))
                        n_it += 1
                        continue

                    # joining existing cluster

                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] += self.segment_counts[segID]
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]
                    #TODO possibly change to faster sorted insertion
                    self.cluster_datapoints[choice] = self._cluster_gen_from_list(self.cluster_dict[choice])
                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if len(self.cluster_dict[clustID]) == 1:
                            del self.cluster_counts[clustID]
                            del self.cluster_dict[clustID]
                            del self.cluster_MLs[clustID]
                            del self.cluster_datapoints[clustID]
                        else:
                            # otherwise update former cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_datapoints[clustID] = self._cluster_gen_remove_one(clustID, segID)
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
                    else:
                        self.unassigned_segs.discard(segID)

            # pick cluster to merge or split
            else:
                # it only makes sense to try joining two clusters if there are at least two of them!
                if len(self.cluster_counts) < 2:
                    n_it += 1
                    #self.tmp_ml_jump_history.append(0)
                    continue

                clust_pick = np.random.choice(self.cluster_dict.keys())
                clust_pick_segs = np.r_[self.cluster_dict[clust_pick]].astype(int)

                # half the time we'll propose splitting this cluster
                if np.random.rand() < 0.5:
                    # if theres only one tuple then we cant split
                    if len(clust_pick_segs) == 1:
                        n_it += 1
                        self.tmp_ml_jump_history.append(0)
                        continue

                    #find the best place to split these tuples based on their datapoint means
                    seg_means = np.array([self.segment_r_list[i].mean() for i in clust_pick_segs])
                    sort_indices = np.argsort(seg_means)
                    sorted_vals = seg_means[sort_indices]

                    #abs_dif = []
                    #for i in np.r_[1:len(sorted_vals)]:
                    #    abs_dif.append(abs(sorted_vals[:i].mean() - sorted_vals[i:].mean()))
                    #abs_dif = np.array(abs_dif)
                    tot_list = []
                    stay_ml = self.cluster_MLs[clust_pick]
                    
                    sorted_segs = clust_pick_segs[sort_indices]
                    sorted_datapoints = self._cluster_gen_from_list(sorted_segs)
                    sorted_lens = self.segment_cov_bins[sorted_segs]
                    datapoint_ind = sorted_lens[0]

                    search_inds = np.r_[1:len(sorted_vals)]
                    for i in search_inds:
                        A_list = sorted_segs[:i]
                        B_list = sorted_segs[i:]
                        
                        A_r = sorted_datapoints[:datapoint_ind]
                        B_r = sorted_datapoints[datapoint_ind:]
                        ML_rat = self._ML_cluster_from_r(A_r) + self._ML_cluster_from_r(B_r) - stay_ml
                        dp_prior_rat = self.DP_split_prior(A_list, B_list)
                        ML_tot = ML_rat + dp_prior_rat
                        tot_list.append(ML_tot)
                        
                        datapoint_ind += sorted_lens[i]
                    tot_list = np.array(tot_list)
                    tot_max = tot_list.max()
                    choice_p = np.exp(tot_list - tot_max) / np.exp(tot_list - tot_max).sum()
                    split_ind = np.random.choice(len(tot_list), p=choice_p)
                    
                    #self.tmp_ml_jump_history.append(tot_list[split_ind] - tot_max)
                    A_list = sorted_segs[:split_ind + 1]
                    B_list = sorted_segs[split_ind + 1:]

                    # add ML ratios to get the likelihood of splitting
                    ML_tot = tot_list[split_ind]

                    # we split with probabilty equal to this likelihood
                    # to avoid overflow
                    if ML_tot >=0:
                        split_prob = 1
                    else:
                        split_prob = np.exp(ML_tot)
                    if np.random.rand() < split_prob:
                        # split these clusters
                        #print("splitting cluster {}".format(clust_pick)) 
                        
                        #update cluster pick to include only segments from list A
                        self.cluster_counts[clust_pick] = sum(self.segment_counts[A_list])
                        self.cluster_dict[clust_pick] = sc.SortedSet(A_list)
                        self.cluster_MLs[clust_pick] = self._ML_cluster_from_list(A_list)
                        self.cluster_datapoints[clust_pick] = self._cluster_gen_from_list(sorted(A_list))

                        # create new cluster with next available index and add segments from list B
                        self.cluster_assignments[B_list] = self.next_cluster_index
                        self.cluster_counts[self.next_cluster_index] = sum(self.segment_counts[B_list])
                        self.cluster_dict[self.next_cluster_index] = sc.SortedSet(B_list)
                        self.cluster_MLs[self.next_cluster_index] = self._ML_cluster_from_list(B_list)
                        self.cluster_datapoints[self.next_cluster_index] = self._cluster_gen_from_list(sorted(B_list))
                        self.next_cluster_index += 1

                #otherwise we'll propose a merge
                else:
                    # get ML of this cluster merged with each of the other existing clusters
                    #ML_join = [
                    #    self._ML_cluster(self.cluster_dict[i].union(clust_pick_segs)) if i != clust_pick else
                    #    self.cluster_MLs[i] for i in self.cluster_dict.keys()]
                    
                    ML_join = [self._ML_cluster_merge(i, clust_pick) if i != clust_pick else 
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

                        prior_MLs = np.r_[[self._ML_cluster_prior(self.prior_clusters[self.prior_clusters.keys()[idx]],
                                                                  clust_pick_segs) if idx != -1 else 0 for idx in
                                           prior_idx]] - (self.cluster_MLs[clust_pick] + np.r_[
                            [self.clust_prior_ML[self.prior_clusters.keys()[idx]] if idx != -1 else -self.cluster_MLs[
                                clust_pick] for idx in prior_idx]])

                        # print('prior_MLs', prior_MLs)
                        clust_prior_p = np.maximum(
                            np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                        # expand MLs to account for multiple new merge clusters--which have liklihood = cluster staying as is = 0
                        ML_rat = np.r_[np.full(len(prior_diff), 0), ML_rat]
                        # DP prior based on clusters sizes now with no alpha
                    #print('cluster ML_rat', ML_rat)
                    #will need to change this when we incorporate muliple smaples
                    count_prior = np.r_[
                        [count_prior[self.prior_clusters.index(x)] for x in prior_diff], self.DP_merge_prior(clust_pick)]

                   # construct transition probability distribution and draw from it
                    MLs_max = (ML_rat + count_prior).max()
                    choice_p = np.exp(ML_rat + count_prior - MLs_max + np.log(clust_prior_p)) / np.exp(
                        ML_rat + count_prior - MLs_max + np.log(clust_prior_p)).sum()
                    # print('clust_choice_p', choice_p)
                    if np.isnan(choice_p.sum()):
                        print("skipping iteration {} due to nan".format(n_it))
                        print(ML_rat)
                        print(count_prior)
                        print(choice_p)
                        n_it += 1
                        continue

                    choice_idx = np.random.choice(
                        np.r_[0:len(ML_rat)],
                        p=choice_p
                    )
                    #self.tmp_ml_jump_history.append((ML_rat + count_prior - MLs_max)[choice_idx])
                    # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                    choice = np.r_[-np.r_[prior_diff] - 1, self.cluster_counts.keys()][choice_idx]
                    choice = int(choice)

                    if choice != clust_pick:
                        #print('cluster {} merging with {}'.format(clust_pick, choice_idx))
                        if choice < 0:
                            # we're merging into an old cluster, which we do by simply reindexing that cluster
                            choice = -(choice + 1)
                            # rename clustID to choice
                            self.cluster_counts[choice] = self.cluster_counts[clust_pick].copy()
                            self.cluster_dict[choice] = self.cluster_dict[clust_pick].copy()
                            self.cluster_assignments[clust_pick_segs] = choice
                            self.cluster_MLs[choice] = self.cluster_MLs[clust_pick]
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
                            self.cluster_counts[merged_ID] += self.segment_counts[vacating_segs].sum()
                            self.cluster_dict[merged_ID] = self.cluster_dict[merged_ID].union(vacating_segs)
                            self.cluster_MLs[merged_ID] = ML_join[choice_idx]
                            self.cluster_datapoints[merged_ID] = self._cluster_gen_from_list(self.cluster_dict[merged_ID])
                            
                            # delete last cluster
                            del self.cluster_counts[vacatingID]
                            del self.cluster_dict[vacatingID]
                            del self.cluster_MLs[vacatingID]
                            del self.cluster_datapoints[vacatingID]

            # save draw after burn in for every n_seg / (n_clust / 2) iterations
            # if burned_in and n_it - n_it_last > self.num_segments / (len(self.cluster_counts) * 2):
            if burned_in and n_it - n_it_last > self.num_segments:
                self.bins_to_clusters.append(self.cluster_assignments.copy())
                self.clusters_to_segs.append(self.cluster_dict.copy())
                self.draw_indices.append(n_it)
                n_it_last = n_it

            n_it += 1

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
