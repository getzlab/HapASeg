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

# for now input will be coverage_df with global segment id column
class Cov_DP:
    def __init__(self, cov_df, beta):
        self.cov_df = cov_df
        self.seg_id_col = self.cov_df.columns.get_loc('segment_ID')
        self.beta = beta

        self.num_segments = self.cov_df.segment_ID.max() + 1
        #self.segment_list = [None] * self.num_segments
        self.segment_r_list = [None] * self.num_segments
        self.segment_C_list = [None] * self.num_segments
        self._init_segments()

        # initialize all but first segment to be unassigned
        self.cluster_assignments = np.ones(self.num_segments) * -1
        self.cluster_assignments[0] = 1
        self.cluster_counts = sc.SortedDict({1: 1})
        self.unassigned_segs = sc.Sorted_list(self.cluster_assignments[self.cluster_assignments == -1])
        self.cluster_dict = sc.SortedDict({1: sc.SortedSet([0])})
        self.cluster_MLs = sc.SortedDict({1, self._ML_cluster([0])})

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

        return ll_opt + self._get_laplacian_approx(H_opt)

    @staticmethod
    def _get_laplacian_approx(H):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-H))) / 2

    def stats_optimizer(self, r, C, ret_hess=False):
        endog = np.exp(np.log(r - (C @ self.beta).flatten()))
        exog = np.ones(r.shape[0])
        sNB = statsNB(endog, exog)
        res = sNB.fit(disp=0)

        if ret_hess:
            return res.params[0], -np.log(res.params[1]), sNB.hessian(res.params)
        else:
            return res.params[0], -np.log(res.params[1])

    def run(self, n_iter=50):

        self.burned_in = False

        n_it = 0
        n_it_last = 0

        while len(self.segs_to_clusters) < n_iter:

            # status update
            if not n_it % 100:
                print("n unassigned: {}".format(len(self.unassigned_segs)))

            # pick either a segment or a cluster at random

            # pick segment
            if np.random.rand() < 1:
                segID = np.random.choice(range(self.num_segments))
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
                    ML_A = self._ML_cluster(self.cluster_dict[clustID].difference(clustID))

                # compute ML of S on its own
                ML_S = self._ML_cluster([clustID])

                # compute ML of every other cluster C = Ck, k != s (cached)
                # for now were also allowing it to chose to stay in current cluster
                #ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items() if ID != clustID])
                ML_C = np.array([ML if ID in self.cluster_MLs for (ID, ML) in self.cluster_MLs.items()])

                # compute ML of every cluster if S joins
                ML_BC = np.array([self._ML_cluster(self.cluster_dict[k].union(clustID))
                                  for k in self.cluster_counts.keys()])

                # likelihood ratios of S joining each other cluster S -> Ck
                ML_rat_BC = ML_A + ML_BC - (ML_AB + ML_C)

                # if cluster is unassigned we set the ML ratio to 1 for staying in its own cluster
                if clustID > -1:
                    ML_rat_BC[clustID] = 1

                # compute ML of S starting a new cluster
                ML_new = ML_A + ML_S - ML_AB

                ML_rat = np.r_[ML_rat_BC, ML_new]

                # compute prior distribution
                c_counts = [v for k, v in self.cluster_counts.items() if k != clustID]
                c_s = self.cluster_counts[clustID]
                p_ct = c_counts / (self.alpha + c_counts.sum() + c_s)
                pct_new = self.alpha / (self.alpha + c_counts.sum() + c_s)
                p_ct = np.r_[p_ct, pct_new]

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(p_ct)) / np.exp(
                    ML_rat - MLs_max + np.log(p_ct)).sum()
                choice_idx = np.random.choice(
                    np.r_[1:len(ML_rat)],
                    p=choice_p
                )

                # idx num_clusters + 1 is equivalent to starting a new cluster
                if choice_idx == len(self.cluster_dict.keys()) + 1:
                    print('starting new cluster')
                    # check if cluster was previously assigned
                    if clustID > -1:
                        # if the segment used to occupy a cluster by itself, do nothing
                        if self.cluster_counts[clustID] == 1:
                            continue

                        # if seg was previously assigned remove it from previous cluster
                        self.cluster_counts[clustID] -= 1
                        self.unassigned_segs.discard(segID)
                        self.cluster_dict[clustID].discard(segID)

                        self.cluster_MLs[choice_idx] = ML_A
                    else:
                        self.unassigned_segs.discard(segID)

                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice_idx
                    self.cluster_counts[choice_idx] = 1
                    self.cluster_dict[choice_idx] = sc.SortedSet([segID])
                    self.cluster_MLs[choice_idx] = ML_S

                else:
                    # joining existing cluster
                    print('joining cluster {}'.format(choice_idx))
                    self.cluster_assignments[segID] = choice_idx
                    self.cluster_counts[choice_idx] += 1
                    self.cluster_dict[choice_idx].add(segID)
                    self.cluster_MLs[choice_idx] = ML_BC[choice_idx]

                    # if seg was previously assigned remove it from previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed and index
                        # will be taken by the last cluster idx
                        if self.cluster_counts[clustID] == 1:
                            last_clust = max(self.cluster_counts.keys())
                            if last_clust != clustID:
                                # move last cluster to empty cluster idx
                                self.cluster_counts[clustID] = self.cluster_counts[last_clust]
                                self.cluster_dict[clustID] = self.cluster_dict[last_clust].copy()
                                self.cluster_MLs[clustID] = self.cluster_MLs[last_clust]

                            del self.cluster_counts[last_clust]
                            del self.cluster_dict[last_clust]
                            del self.cluster_MLs[last_clust]

                        else:
                            # otherwise update former cluster
                            self.cluster_counts[clustID] -= 1
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[choice_idx] = ML_A
                    else:
                        self.unassigned_segs.discard(segID)

            n_it += 1