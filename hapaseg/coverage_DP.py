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

from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB

from capy import seq

def LSE(x):
    lmax = np.max(x)
    return lmax + np.log(np.exp(x - lmax).sum())

# segment object for keeping track of r and C arrays
# class Segment:
#     def __init__(self, r, C):
#         self.r = r
#         self.C = C
#
# class Cluster:
#     def __init__(self, ):
#         self.size = 0
#         self.r = np.array()
#         self.C = np.array()
#
#     def add_seg(self, seg):
#         self.r = np.r_[self.r, seg.r]
#         self.C = np.r_[self.C, seg.C]

# num_samples = number of segmentation samples to use
# num draws = number of thinned DP draws to take after burn in for each sample

class Coverage_DP:
    def __init__(self, segmentation_h5):
        self.segmentation_draws = segmentation_h5['segment_IDs']
        self.beta = segmentation_h5['beta']
        self.cov_df = pd.read_hf(segmentation_h5, 'cov_df')

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

        prior_clusts = None
        count_prior_sum = None

        # TODO: load segmentation samples randomly
        self.bins_to_clusters = []
        self.segs_to_clusters = []
        for samp in range(self.num_samples):
            self.cov_df['segment_ID'] = self.segmentation_draws[:, samp]

            DP_runner = Run_Cov_DP(self.cov_df, self.beta, prior_clusts, count_prior_sum)
            draws, prior_clusts, count_prior_sum = DP_runner.run(self.num_draws, samp)
            self.segs_to_clusters.append(draws)

# for now input will be coverage_df with global segment id column
class Run_Cov_DP:
    def __init__(self, cov_df, beta, prior_clusters=None, count_prior_sum=None):
        self.cov_df = cov_df
        self.seg_id_col = self.cov_df.columns.get_loc('segment_ID')
        self.beta = beta

        self.num_segments = self.cov_df.segment_ID.max() + 1
        self.segment_r_list = [None] * self.num_segments
        self.segment_C_list = [None] * self.num_segments
        self._init_segments()

        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1

        self.cluster_counts = sc.SortedDict()
        self.unassigned_segs = sc.SortedList(np.r_[0:self.num_segments])
        self.cluster_dict = sc.SortedDict()
        self.cluster_MLs = sc.SortedDict()

        # if first iteration then add first segment to first cluster
        if prior_clusters is None:
            self.cluster_counts[0] = 1
            self.unassigned_segs.discard(0)
            self.cluster_dict[0] = sc.SortedSet([0])
            self.cluster_MLs[0] = self._ML_cluster([0])
            # next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = 1
        else:
            self.prior_clusters = prior_clusters.copy()
            self.count_prior_sum = count_prior_sum.copy()
            self.next_cluster_index = prior_clusters.keys().max() + 1
            self.clust_prior_ML = None

        # containers for saving the MCMC trace
        self.clusters_to_segs = []
        self.segs_to_clusters = []

        self.alpha = 0.1

    # initialize each segment object with its data
    def _init_segments(self):
        for ID, seg_df in self.cov_df.groupby('segment_ID'):
            self.segment_r_list[ID] = seg_df['covcorr']
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
        
        #print('clust set: ', cluster_set, 'll: ', ll_opt, 'lap approx: ', self._get_laplacian_approx(H_opt))
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

            # compute MLs of segment joining each prior cluster
            BC = np.r_[[self._ML_cluster(self.prior_clusters[c].union([segID]))
                        for c in self.prior_clusters.keys()]]
            S = self._ML_cluster([segID])
            C = self.clust_prior_ML.values()

            #prior liklihood ratios
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
            self.cluster_MLs[choice] = BC[idx == choice]
        print('prior assigned')
        print(self.cluster_dict)

    def run(self, n_iter, sample_num):

        self.burned_in = False

        n_it = 0
        n_it_last = 0

        if self.prior_clusters:
            count_prior = self.count_prior_sum / sample_num
            self.clust_prior_ML = sc.SortedDict({k: self._ML_cluster(self.prior_clusters[k])
                                                 for k in self.prior_clusters.keys()})
            self.initial_prior_assignment(count_prior)

        while n_it < n_iter:
        # while len(self.segs_to_clusters) < n_iter:

            # status update
            if not n_it % 100:
                print("n unassigned: {}".format(len(self.unassigned_segs)))

            # pick either a segment or a cluster at random
            
            # pick segment
            if np.random.rand() < 0.5:
                if len(self.unassigned_segs) > 0 and len(self.unassigned_segs)/self.num_segments < 0.1 and np.random.rand() < 0.5:
                    segID = np.random.choice(self.unassigned_segs)
                else:
                    segID = np.random.choice(range(self.num_segments))
                
                # get cluster assignment of S
                clustID = self.cluster_assignments[segID]
                #print(segID, clustID)
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
            #        print(clustID, segID)
             #       print(self.cluster_dict[clustID].difference([segID]))
                    ML_A = self._ML_cluster(self.cluster_dict[clustID].difference([segID]))

                # compute ML of S on its own
                ML_S = self._ML_cluster([segID])

                # compute ML of every other cluster C = Ck, k != s (cached)
                # for now were also allowing it to chose to stay in current cluster

                ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])
                
                # compute ML of every cluster if S joins
                ML_BC = np.array([self._ML_cluster(self.cluster_dict[k].union([segID]))
                                  for k in self.cluster_counts.keys()])
                #print('ml_A: ', ML_A)
                #print('ml_AB: ', ML_AB)
                #print('ml_BC: ', ML_BC)
                #print('ml_C: ', ML_C)
                #print('ml_s: ', ML_S)
                # likelihood ratios of S joining each other cluster S -> Ck
                ML_rat_BC = ML_A + ML_BC - (ML_AB + ML_C)

                # if cluster is unassigned we set the ML ratio to 1 for staying in its own cluster
                #print(ML_rat_BC)
                if clustID > -1:
                    ML_rat_BC[clustID] = 0
                
                # compute ML of S starting a new cluster
                ML_new = ML_A + ML_S - ML_AB

                ML_rat = np.r_[ML_rat_BC, ML_new]
                #print('ml_rat: ', ML_rat)

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
                               (prior_com | prior_null)]]
                    ]

                    prior_MLs = np.r_[[self._ML_cluster(self.prior_clusters[idx].union([segID])) if idx != -1 else 1
                                       for idx in prior_idx]] - (self._ML_cluster([segID]) +
                                                                 np.r_[[self._ML_cluster(self.prior_clusters[idx])
                                                                        if idx != -1 else - self._ML_cluster([segID])
                                                                        for idx in prior_idx]])

                    clust_prior_p = np.maximum(
                        np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                    # expand MLs to account for multiple new clusters
                    ML_rat = np.r_[np.full(len(prior_diff), ML_rat[-1]), ML_rat]

                # DP prior based on clusters sizes
                count_prior = np.r_[
                    [count_prior[x] for x in prior_diff], self.clust_counts.values(), self.alpha]
                count_prior /= count_prior.sum()

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                choice = np.r_[-np.r_[prior_diff] - 1,
                               self.clust_counts.keys(), self.next_cluster_index][choice_idx]


                # if choice == next_cluster_idx then start a brand new cluster with this new index
                # if choice < 0 then we create a new cluster with the index of the old cluster
                if choice == self.next_cluster_index or choice < 0:
                    old_clust = False
                    if choice < 0:
                        # since its an old cluster we correct its index
                        choice = -(choice + 1)
                        old_clust = True

                    #print('{} starting new cluster {}'.format(segID, choice_idx))
                    if clustID > -1:
                        # if the segment used to occupy a cluster by itself, do nothing if its a new cluster
                        # need to rename it if its a old cluster
                        if self.cluster_counts[clustID] == 1:
                            if old_clust:
                                # rename clustID to choice
                                self.cluster_counts[choice] = self.cluster_counts[clustID]
                                self.cluster_dict[choice] = self.cluster_dict[clustID]
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

                else:
                    # if remaining in same cluster, skip
                    if clustID == choice_idx:
                     #   print('{} staying in current cluster'.format(clustID))
                        n_it += 1
                        continue

                    # joining existing cluster
                    #print('{} joining cluster {}'.format(segID, choice_idx))
                    
                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] += 1
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[choice - 1]

                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if self.cluster_counts[clustID] == 1:
                            del self.cluster_counts[last_clust]
                            del self.cluster_dict[last_clust]
                            del self.cluster_MLs[last_clust]

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
                if len(self.clust_counts) < 2:
                    n_it += 1
                    continue

                clust_pick = np.random.choice(self.cluster_dict.keys())

                # get ML of this cluster merged with each of the other existing clusters
                ML_join = [self._ML_cluster(self.cluster_dict[i].union(self.cluster_dict[clust_pick]))
                           if i != clust_pick else self.cluster_MLs[i] for i in self.cluster_dict.keys()]
                ML_ratio = np.array(ML_join) - np.array(self.cluster_MLs.values())
                p_ct = np.array(self.cluster_counts.values())
                p_ct = p_ct / (self.alpha + p_ct.sum())

                MLs_max = ML_ratio.max()
                choice_p = np.exp(ML_ratio - MLs_max + np.log(p_ct)) / np.exp(
                    ML_ratio - MLs_max + np.log(p_ct)).sum()
                choice_idx = np.random.choice(
                    np.r_[1:len(ML_ratio) + 1],
                    p=choice_p
                )

                if choice_idx != clust_pick:
                    
                    #print('cluster {} merging with {}'.format(clust_pick, choice_idx))
                    
                    # we need to merge clust_pick and choice_idx which we do by merging to the cluster with more
                    # segments and then swapping the last cluster into the empty index (unless it was already last)
                    tup = (clust_pick, choice_idx)
                    larger_cluster = np.argmax([self.cluster_counts[clust_pick], self.cluster_counts[choice_idx]])
                    merged_ID = tup[larger_cluster]
                    vacatingID = tup[int(not larger_cluster)]

                    # move vacatingID to newclustID
                    # update new cluster with additional segments
                    vacating_segs = self.cluster_dict[vacatingID]
                    self.cluster_assignments[vacating_segs] = merged_ID
                    self.cluster_counts[merged_ID] += len(vacating_segs)
                    self.cluster_dict[merged_ID] = self.cluster_dict[merged_ID].union(vacating_segs)
                    self.cluster_MLs[merged_ID] = ML_join[choice_idx - 1]

                    # now we need to deal with the vacated cluster
                    last_clust = max(self.cluster_counts.keys())
                    if vacatingID != last_clust:
                        # if its not the last cluster we need to swap the last cluster with the vacated one

                        # move last cluster to empty cluster idx
                        self.cluster_counts[vacatingID] = self.cluster_counts[last_clust]
                        self.cluster_dict[vacatingID] = self.cluster_dict[last_clust].copy()
                        self.cluster_MLs[vacatingID] = self.cluster_MLs[last_clust]
                        self.cluster_assignments[self.cluster_assignments == last_clust] = vacatingID

                    # either way we delete last cluster
                    del self.cluster_counts[last_clust]
                    del self.cluster_dict[last_clust]
                    del self.cluster_MLs[last_clust]

                #else:
                    # we've chosen to stay in the same cluster
                    #print('cluster {} decided not to merge'.format(clust_pick))
            
            n_it += 1
