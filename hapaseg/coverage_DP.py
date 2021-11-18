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


# for now input will be coverage_df with global segment id column
class Run_Cov_DP:
    def __init__(self, cov_df, beta, prior_run=None, count_prior_sum=None):
        self.cov_df = cov_df
        self.seg_id_col = self.cov_df.columns.get_loc('segment_ID')
        self.beta = beta

        self.num_segments = self.cov_df.segment_ID.max() + 1
        self.segment_r_list = [None] * self.num_segments
        self.segment_C_list = [None] * self.num_segments
        self._init_segments()

        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1

        self.cluster_counts = sc.SortedDict({})
        self.unassigned_segs = sc.SortedList(np.r_[0:self.num_segments])
        self.cluster_dict = sc.SortedDict({})
        self.cluster_MLs = sc.SortedDict({})

        self.prior_clusters = None
        self.prior_r_list = None
        self.prior_C_list = None
        self.count_prior_sum = None
        # for saving init clusters
        self.init_clusters = None
        # if first iteration then add first segment to first cluster
        if prior_run is None:
            self.cluster_counts[0] = 1
            self.unassigned_segs.discard(0)
            self.cluster_dict[0] = sc.SortedSet([0])
            self.cluster_MLs[0] = self._ML_cluster([0])
            # next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = 1
        else:
            self.prior_clusters = prior_run.cluster_dict.copy()
            self.prior_r_list = prior_run.segment_r_list.copy()
            self.prior_C_list = prior_run.segment_C_list.copy()
            self.count_prior_sum = count_prior_sum.copy()
            self.next_cluster_index = np.r_[self.prior_clusters.keys()].max() + 1
            self.clust_prior_ML = None
            # print('prior clust', self.prior_clusters)
        # containers for saving the MCMC trace
        self.clusters_to_segs = []
        self.bins_to_clusters = []

        self.alpha = 0.1

    # initialize each segment object with its data
    def _init_segments(self):
        for ID, seg_df in self.cov_df.groupby('segment_ID'):
            self.segment_r_list[ID] = seg_df['covcorr'].values
            self.segment_C_list[ID] = np.c_[np.log(seg_df["C_len"]), seg_df["C_RT_z"], seg_df["C_GC_z"]]

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

    def stats_optimizer(self, r, C, ret_hess=False):
        endog = np.exp(np.log(r) - (C @ self.beta).flatten())
        exog = np.ones(r.shape[0])
        sNB = statsNB(endog, exog)
        res = sNB.fit(disp=0)

        if ret_hess:
            return res.params[0], -np.log(res.params[1]), sNB.hessian(res.params)
        else:
            return res.params[0], -np.log(res.params[1])

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

    def run(self, n_iter, sample_num):

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

        # while n_it < n_iter:
        while len(self.bins_to_clusters) < n_iter:

            # status update
            if not n_it % 250 and self.prior_clusters is None:
                print("n unassigned: {}".format(len(self.unassigned_segs)))

            # start couting for burn in
            if not n_it % 100:
                if not all_assigned and (self.cluster_assignments > -1).all():
                    all_assigned = True
                    n_it_last = n_it

                # burn in after n_seg / n_clust iteration
                if not burned_in and all_assigned and n_it - n_it_last > self.num_segments / len(self.cluster_counts):
                    burned_in = True

            # pick either a segment or a cluster at random

            # pick segment
            if np.random.rand() < 0.5:
                if len(self.unassigned_segs) > 0 and len(
                        self.unassigned_segs) / self.num_segments < 0.1 and np.random.rand() < 0.5:
                    segID = np.random.choice(self.unassigned_segs)
                else:
                    segID = np.random.choice(range(self.num_segments))

                # get cluster assignment of S
                clustID = self.cluster_assignments[segID]
                # print(segID, clustID)
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

                # print('ml_rat: ', ML_rat)

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
                    # print('prior_idx', prior_idx)
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
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # if self.prior_clusters:
                #   print(choice_p)
                #  print(choice_idx)
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

                # get ML of this cluster merged with each of the other existing clusters
                ML_join = [
                    self._ML_cluster(self.cluster_dict[i].union(clust_pick_segs)) if i != clust_pick else
                    self.cluster_MLs[i] for i in self.cluster_dict.keys()]
                ML_rat = np.array(ML_join) - np.array(self.cluster_MLs.values())

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

                    # print('prior_MLs', prior_MLs)
                    clust_prior_p = np.maximum(
                        np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                    # expand MLs to account for multiple new merge clusters--which have liklihood = cluster staying as is = 0
                    ML_rat = np.r_[np.full(len(prior_diff), 0), ML_rat]
                    # DP prior based on clusters sizes now with no alpha

                count_prior = np.r_[
                    [count_prior[self.prior_clusters.index(x)] for x in prior_diff], self.cluster_counts.values()]
                count_prior /= (count_prior.sum() + self.alpha)

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()
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
