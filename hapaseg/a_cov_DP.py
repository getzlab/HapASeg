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

colors = mpl.cm.get_cmap("tab10").colors


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
    def __init__(self, cov_df, beta, seed_clusters=True, prior_run=None, count_prior_sum=None):
        self.cov_df = cov_df
        self.a_imbalances = a_imbalances
        self.seg_id_col = self.cov_df.columns.get_loc('a_cov_segID')
        self.beta = beta

        self.num_segments = self.cov_df.iloc[:, self.seg_id_col].max() + 1
        self.segment_r_list = [None] * self.num_segments
        self.segment_C_list = [None] * self.num_segments
        self.segment_sigma_list = [None] * self.num_segments
        self.segment_counts_list = [None] * self.num_segments
        self.segment_allele = [g['allele'].values[0] for i, g in self.cov_df.groupby('a_cov_segID')]
        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1

        self.cluster_counts = sc.SortedDict({})
        self.unassigned_segs = sc.SortedList(np.r_[0:self.num_segments])
        self.cluster_dict = sc.SortedDict({})
        self.cluster_MLs = sc.SortedDict({})
        self.greylist_segments = sc.SortedSet({})

        self.prior_clusters = None
        self.prior_r_list = None
        self.prior_C_list = None
        self.count_prior_sum = None
        self.ML_total_history = []

        # for saving init clusters
        self.init_clusters = None
    
        self._init_segments()
        self._init_clusters(prior_run, count_prior_sum, seed_cluters=seed_clusters)

        # containers for saving the MCMC trace
        self.clusters_to_segs = []
        self.bins_to_clusters = []

        self.alpha = 0.1

    # initialize each segment object with its data
    def _init_segments(self):
        for ID, grouped in self.cov_df.groupby('a_cov_segID'):
            mu = grouped['cov_DP_mu'].values[0]
            major, minor =  (grouped['seg_maj_count'].values[0], grouped['seg_min_count'].values[0])
            if len(grouped) < 3:
                self.greylist_segments.add(ID)
            if grouped.allele.values[0] == -1:
                r = np.exp(mu)  * minor / (minor + major)
            else:
                r = np.exp(mu)  * major / (minor + major)
            self.segment_r_list[ID] = r 

    def _init_clusters(self, prior_run, count_prior_sum, seed_clusters=True):
        [self.unassigned_segs.discard(s) for s in self.greylist_segments]
            # for now only support single iteration
        if not seed_clusters:
            first = (set(range(self.num_segments)) - self.greylist_segments)[0]
            clusterID = 0
            self.cluster_counts[0] = 1
            self.unassigned_segs.discard(0)
            self.cluster_dict[0] = sc.SortedSet([first])
            self.cluster_MLs[0] = self._ML_cluster([first])
            #next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = 1
        else
            clusterID = 0
            for name, grouped in self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele']):
                segIDs = sc.SortedSet(grouped['a_cov_segID'].values) - self.greylist_segments
                if len(segIDs) == 0:
                    continue
                self.cluster_dict[clusterID] = segIDs
                self.cluster_assignments[segIDs] = clusterID
                self.cluster_counts[clusterID] = len(segIDs)
                self.cluster_MLs[clusterID] = self._ML_cluster(segIDs)
                clusterID += 1
            self.next_cluster_index = clusterID
            self.unassigned_segs.clear()

    def _ML_cluster(self, cluster_set):
        r_lst = []
        for s in cluster_set:
            r_seg = self.segment_r_list[s]
            r_lst.append(r_seg)
        r = np.hstack(r_lst)

        return self.ML_normalgamma(r, 100, 0.000001, 10, 1)

    def ML_normalgamma(self, x, mu0, kappa0, alpha0, beta0):
        x_mean = x.mean()
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
        # print('prior assigned')
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
                if not all_assigned and (self.cluster_assignments[white_segments] > -1).all():
                    all_assigned = True
                    n_it_last = n_it

                # burn in after n_seg / n_clust iteration
                if not burned_in and all_assigned and n_it - n_it_last > self.num_segments / len(self.cluster_counts):
                    print('burnin')
                    burned_in = True

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
                # print('seg_pick:', segID, clustID)
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
                # print(ML_rat_BC)
                if clustID > -1:
                    ML_rat_BC[list(self.cluster_counts.keys()).index(clustID)] = 0

                # compute ML of S starting a new cluster
                ML_new = ML_A + ML_S - ML_AB

                ML_rat = np.r_[ML_rat_BC, ML_new]
                # if self.prior_clusters:
                # print('ML_A', ML_A)
                # print('ML_AB', ML_AB)
                # print('ML_C', ML_C)
                # print('C now', [self._ML_cluster(self.cluster_dict[c]) for c in self.cluster_dict.keys()])
                # print('ML_BC', ML_BC)

               # print('ml_rat_seg: ', ML_rat)

                prior_diff = []
                clust_prior_p = 1
                # use prior cluster information if available
                if self.prior_clusters:
                    # print('segID: ', segID, 'cur cluster: ', clustID)
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
                    # print('prior_diff', prior_diff)
                    # print('prior_idx', prior_id
                    # print('prior BC', [self._ML_cluster_prior(self.prior_clusters[self.prior_clusters.keys()[idx]], [segID]) for idx in prior_idx])
                    # print('prior C', [self.clust_prior_ML[self.prior_clusters.keys()[idx]]for idx in prior_idx] )
                    prior_MLs = np.r_[[self._ML_cluster_prior(self.prior_clusters[self.prior_clusters.keys()[idx]],
                                                              [segID]) if idx != -1 else 0 for idx in prior_idx]] - (
                                        self._ML_cluster([segID]) + np.r_[[self.clust_prior_ML[
                                                                               self.prior_clusters.keys()[
                                                                                   idx]] if idx != -1 else - self._ML_cluster(
                                    [segID]) for idx in prior_idx]])
                    # print('prior_MLs', prior_MLs)
                    clust_prior_p = np.maximum(
                        np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                    #  print('clust_prior_p: ', clust_prior_p)
                    # expand MLs to account for multiple new clusters
                    ML_rat = np.r_[np.full(len(prior_diff), ML_rat[-1]), ML_rat]
                #   print('ML_rat:', ML_rat)
                # DP prior based on clusters sizes
                count_prior = np.r_[
                    [count_prior[self.prior_clusters.index(x)] for x in
                     prior_diff], self.cluster_counts.values(), self.alpha]
                count_prior /= count_prior.sum()

                # if self.prior_clusters:
                # print('count prior: ', count_prior)
                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()
                # print(choice_p)
                if np.isnan(choice_p.sum()):
                    print('skipping iteration {} due to nan. picked segment {}'.format(n_it, segID))
                    n_it += 1
                    continue
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # if self.prior_clusters:
                #   print(choice_p)
            # print(choice_idx)
                # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                choice = np.r_[-np.r_[prior_diff] - 1,
                               self.cluster_counts.keys(), self.next_cluster_index][choice_idx]
                choice = int(choice)
                # print('choice: ', choice)
                # print(self.cluster_counts)
                # print(np.r_[-np.r_[prior_diff] - 1, self.cluster_counts.keys(), self.next_cluster_index])
                # if choice == next_cluster_idx then start a brand new cluster with this new index
                # if choice < 0 then we create a new cluster with the index of the old cluster
                if choice == self.next_cluster_index or choice < 0:
                    old_clust = False
                    if choice < 0:
                        # since its an old cluster we correct its index
                        choice = -(choice + 1)
                        old_clust = True
                        # print('starting new old cluster')
                    # else:
                    # print('starting new clust')

                    # print('{} starting new cluster {}'.format(segID, choice_idx))
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
                            n_it += 1
                            continue
                        else:
                            # if seg was previously assigned remove it from previous cluster
                            self.cluster_counts[clustID] -= 1
                            self.cluster_dict[clustID].discard(segID)

                            self.cluster_MLs[clustID] = ML_A
                    else:
                        self.unassigned_segs.discard(segID)

                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] = 1
                    self.cluster_dict[choice] = sc.SortedSet([segID])
                    self.cluster_MLs[choice] = ML_S
                    self.next_cluster_index += 1
                else:
                    # if remaining in same cluster, skip
                    if clustID == choice:
                        #   print('{} staying in current cluster'.format(clustID))
                        n_it += 1
                        continue

                    # joining existing cluster
                    # print('{} joining cluster {}'.format(segID, choice_idx))

                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] += 1
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]

                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if self.cluster_counts[clustID] == 1:
                            del self.cluster_counts[clustID]
                            del self.cluster_dict[clustID]
                            del self.cluster_MLs[clustID]

                        else:
                            # otherwise update former cluster
                            self.cluster_counts[clustID] -= 1
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
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
                # print('clust_pick:', clust_pick)
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
                count_prior = np.r_[
                    [count_prior[self.prior_clusters.index(x)] for x in prior_diff], self.cluster_counts.values()]
                count_prior /= (count_prior.sum() + self.alpha)

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()
                # print('clust_choice_p', choice_p)
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
                    # print('cluster {} merging with {}'.format(clust_pick, choice_idx))h
                    if choice < 0:
                        # we're merging into an old cluster, which we do by simply reindexing that cluster
                        choice = -(choice + 1)
                        # rename clustID to choice
                        self.cluster_counts[choice] = self.cluster_counts[clust_pick]
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
                        self.cluster_counts[merged_ID] += len(vacating_segs)
                        self.cluster_dict[merged_ID] = self.cluster_dict[merged_ID].union(vacating_segs)
                        self.cluster_MLs[merged_ID] = ML_join[choice_idx]

                        # delete last cluster
                        del self.cluster_counts[vacatingID]
                        del self.cluster_dict[vacatingID]
                        del self.cluster_MLs[vacatingID]

            # save draw after burn in for every n_seg / (n_clust / 2) iterations
            # if burned_in and n_it - n_it_last > self.num_segments / (len(self.cluster_counts) * 2):
            if burned_in and n_it - n_it_last > self.num_segments:
                self.bins_to_clusters.append(self.cluster_assignments.copy())
                self.clusters_to_segs.append(self.cluster_dict.copy())
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
