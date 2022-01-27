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
import pickle
import os

from capy import seq, mut

from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB

colors = mpl.cm.get_cmap("tab20").colors


def LSE(x):
    lmax = np.max(x)
    return lmax + np.log(np.exp(x - lmax).sum())

# method for concatenating dp draws into a single large df to be used by the acdp
def generate_acdp_df(SNP_path, # path to SNP df
                 CDP_path, # path to CDP runner pickle object
                 ADP_path, # path to npz ADP result
                 ADP_draw_index=-1): # index of ADP draw used in Coverage MCMC

    SNPs = pd.read_pickle(SNP_path)
    SNPs["chr"], SNPs["pos"] = seq.gpos2chrpos(SNPs["gpos"])
    ADP_clusters = np.load(ADP_path)
    phases = ADP_clusters["snps_to_phases"][ADP_draw_index]
    SNPs.iloc[phases, [0, 1]] = SNPs.iloc[phases, [1, 0]]

    with open(CDP_path, 'rb') as f:
        dp_pickle = pickle.load(f)

    # currently uses the last sample from every DP draw
    draw_dfs = []
    for draw_num, dp_run in enumerate(dp_pickle.DP_runs):
        print('concatenating dp run ', draw_num)
        a_cov_seg_df = dp_run.cov_df.copy()

        # add minor and major allele counts for each bin to the cov_seg_df here to allow for beta draws on the fly for each segment
        a_cov_seg_df['min_count'] = 0
        a_cov_seg_df['maj_count'] = 0
        min_col_idx = a_cov_seg_df.columns.get_loc('min_count')
        maj_col_idx = a_cov_seg_df.columns.get_loc('maj_count')

        SNPs["cov_tidx"] = mut.map_mutations_to_targets(SNPs, a_cov_seg_df, inplace=False)

        for idx, group in SNPs.groupby('cov_tidx').indices.items():
            minor, major = SNPs.iloc[group, [0, 1]].sum()
            a_cov_seg_df.iloc[int(idx), [min_col_idx, maj_col_idx]] = minor, major

        # add dp cluster annotations
        a_cov_seg_df['cov_DP_cluster'] = -1

        segs_to_clusts = dp_run.bins_to_clusters[-1]
        for seg in range(len(segs_to_clusts)):
            a_cov_seg_df.loc[a_cov_seg_df['segment_ID'] == seg, 'cov_DP_cluster'] = segs_to_clusts[seg]

        # adding cluster mus and sigmas to df for each tuple
        # falls back to CDP mu and sigma if the tuple is too small (less than 10 coverage bins)
        a_cov_seg_df['cov_DP_mu'] = 0
        a_cov_seg_df['cov_DP_sigma'] = 0

        for adp, cdp in a_cov_seg_df.groupby(['allelic_cluster', 'cov_DP_cluster']).indices:
            acdp_clust = a_cov_seg_df.loc[
                (a_cov_seg_df.cov_DP_cluster == cdp) & (a_cov_seg_df.allelic_cluster == adp)]
            if len(acdp_clust) < 10:
                acdp_clust = a_cov_seg_df.loc[a_cov_seg_df.cov_DP_cluster == cdp]
            r = acdp_clust.covcorr.values
            C = np.c_[np.log(acdp_clust['C_len'].values), acdp_clust['C_RT_z'].values, acdp_clust['C_GC_z'].values]
            endog = r
            exog = np.ones(r.shape)
            sNB = statsNB(endog, exog, offset = (C @ dp_pickle.beta).flatten())
            res = sNB.fit(disp=0)
            mu = res.params[0]
            a_cov_seg_df.loc[
                (a_cov_seg_df.cov_DP_cluster == cdp) & (a_cov_seg_df.allelic_cluster == adp), 'cov_DP_mu'] = mu
            H = sNB.hessian(res.params)
            # variance of the mu posterior is taken as the inverse of the hessian component for mu
            mu_sigma = np.linalg.inv(-H)[0, 0]
            a_cov_seg_df.loc[(a_cov_seg_df.cov_DP_cluster == cdp) & (
                        a_cov_seg_df.allelic_cluster == adp), 'cov_DP_sigma'] = mu_sigma

        # add next_g for ease of plotting down the line
        a_cov_seg_df["next_g"] = np.r_[a_cov_seg_df.iloc[1:]["start_g"], 2880794554]

        # duplicate segments to account for second allele
        num_bins = len(a_cov_seg_df)
        a_cov_seg_df = a_cov_seg_df.reset_index(drop=True)
        a_cov_seg_df = a_cov_seg_df.append(a_cov_seg_df)
        a_cov_seg_df = a_cov_seg_df.reset_index(drop=True)
        a_cov_seg_df['allele'] = 0
        allele_col_idx = a_cov_seg_df.columns.get_loc('allele')

        # minor
        a_cov_seg_df.iloc[:num_bins, allele_col_idx] = -1
        # major
        a_cov_seg_df.iloc[num_bins:, allele_col_idx] = 1

        a_cov_seg_df['dp_draw'] = draw_num
        draw_dfs.append(a_cov_seg_df)

    print('completed ACDP dataframe generation')
    return pd.concat(draw_dfs), dp_run.beta

class AllelicCoverage_DP:
    def __init__(self, cov_df, beta, allelic_segs_path, seed_all_clusters=True):
        self.cov_df = cov_df
        self.beta = beta
        self.allelic_segs_path = allelic_segs_path
        self.seed_all_clusters = seed_all_clusters

        self.num_segments = len(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        self.segment_r_list = [None] * self.num_segments
        self.segment_V_list = np.zeros(self.num_segments)
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

        # inverse gamma hyper parameter default values -- will be set later based on tuples
        self.ig_alpha = 100
        self.ig_ssd = 30

        self._init_segments()
        self._init_clusters()

        # containers for saving the MCMC trace_cov_dp
        self.clusters_to_segs = []
        self.bins_to_clusters = []
        self.draw_indices = []

        self.alpha = 0.5

    # initialize each segment object with its data
    def _init_segments(self):
        # keep a table of reads for each allelic cluster to fallback on if the tuple has too few bins (<10)
        fallback_counts = sc.SortedDict({})
        for ID, (name, grouped) in enumerate(
                self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw'])):
            mu = grouped['cov_DP_mu'].values[0]
            sigma = grouped['cov_DP_sigma'].values[0]
            group_len = len(grouped)
            if group_len > 10:
                major, minor = (grouped['maj_count'].sum(), grouped['min_count'].sum())
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

            if allele == -1:
                a = minor
                b = major
            else:
                a = major
                b = minor
            r = np.array(np.exp(s.norm.rvs(mu, np.sqrt(sigma), size=group_len)) * s.beta.rvs(a, b, size=group_len))

            V = (np.exp(s.norm.rvs(mu, np.sqrt(sigma), size=10000)) * s.beta.rvs(a, b, size=10000)).var()

            # blacklist segments with very high variance
            if np.sqrt(V) > 15:
                self.greylist_segments.add(ID)

            self.segment_V_list[ID] = V
            self.segment_r_list[ID] = r
            self.segment_cov_bins[ID] = group_len
            self.segment_counts[ID] = group_len

        # go back through segments and greylist ones with high variance
        greylist_mask = np.ones(self.num_segments, dtype=bool)
        greylist_mask[self.greylist_segments] = False
        cutoff = np.quantile(self.segment_V_list[greylist_mask], 0.80)
        self.ig_ssd = self.segment_V_list[greylist_mask].mean()
        self.ig_alpha = self.segment_cov_bins[greylist_mask].mean()
        print(cutoff)
        for i in set(range(self.num_segments)) - self.greylist_segments:
            if self.segment_V_list[i] > cutoff:
                self.greylist_segments.add(i)

    def _init_clusters(self):
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
        return np.concatenate([self.cluster_datapoints[clusterID], self.segment_r_list[segID]], axis=0)

    # assumes the datapoints are ordered by segment ID
    def _cluster_gen_remove_one(self, clusterID, segID):
        cur = self.cluster_datapoints[clusterID]
        seg_ind = self.cluster_dict[clusterID].index(segID)
        st = self.segment_cov_bins[:seg_ind].sum()
        en = st + self.segment_cov_bins[segID]
        return np.concatenate([cur[:st], cur[en:]], axis=0)

    def _cluster_gen_merge(self, clust_A, clust_B):
        return np.concatenate([self.cluster_datapoints[clust_A], self.cluster_datapoints[clust_B]], axis=0)

    def _ML_cluster_from_r(self, r):
        alpha = self.ig_alpha
        beta = alpha / 2 * self.ig_ssd
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def _ML_cluster_from_list(self, cluster_list):
        r = self._cluster_gen_from_list(cluster_list)
        alpha = self.ig_alpha
        beta = alpha / 2 * self.ig_ssd
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def _ML_cluster_add_one(self, clusterID, segID):
        r = self._cluster_gen_add_one(clusterID, segID)
        alpha = self.ig_alpha
        beta = alpha / 2 * self.ig_ssd
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def _ML_cluster_remove_one(self, clusterID, segID):
        r = self._cluster_gen_remove_one(clusterID, segID)
        alpha = self.ig_alpha
        beta = alpha / 2 * self.ig_ssd
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    def _ML_cluster_merge(self, clust_A, clust_B):
        r = self._cluster_gen_merge(clust_A, clust_B)
        alpha = self.ig_alpha
        beta = alpha / 2 * self.ig_ssd
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    # worker function for normal-gamma distribution log Marginal Likelihood
    def ML_normalgamma(self, x, mu0, kappa0, alpha0, beta0):
        # for now x_mean is the same as mu0
        x_mean = mu0
        n = len(x)

        mu_n = (kappa0*mu0 + n * x_mean) / (kappa0 + n)
        kappa_n = kappa0 + n
        alpha_n = alpha0 + n/2
        beta_n = beta0 + 0.5 * ((x - x_mean)**2).sum() + kappa0 * n * (x_mean - mu0)**2 / 2*(kappa0 + n)

        return ss.loggamma(alpha_n) - ss.loggamma(alpha0) + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n) + np.log(kappa0 / kappa_n) / 2 - n * np.log(2*np.pi) / 2

    def save_ML_total(self):
        ML_tot = np.r_[self.cluster_MLs.values()].sum()
        self.ML_total_history.append(ML_tot)

        num_clusts = len(self.cluster_dict)
        N = sum(self.cluster_counts.values())
        DP_tot = num_clusts * np.log(self.alpha) + sum(
            [ss.gammaln(na) for na in self.cluster_counts.values()]) - ss.gammaln(self.alpha + N) + ss.gammaln(
            self.alpha)
        self.DP_total_history.append(DP_tot)
        self.MLDP_total_history.append(ML_tot + DP_tot)

    # functions for computing the DP priors for each cluster action

    def DP_merge_prior(self, cur_cluster):
        cur_index = self.cluster_counts.index(cur_cluster)
        cluster_vals = np.array(self.cluster_counts.values())
        N = cluster_vals.sum()
        M = cluster_vals[cur_index]
        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            if i != cur_index:
                prior_results[i] = ss.loggamma(M + nc) + ss.loggamma(N + self.alpha - M) - (
                            ss.loggamma(nc) + ss.loggamma(N + self.alpha))
            else:
                # the prior prob of remaining in the current cluster is the same as for joining a new cluster
                prior_results[i] = ss.gammaln(M) + np.log(self.alpha) + ss.gammaln(N + self.alpha - M) - ss.gammaln(
                    N + self.alpha)
        return prior_results

    def DP_tuple_split_prior(self, seg_id):
        cur_cluster = self.cluster_assignments[seg_id]
        seg_size = self.segment_cov_bins[seg_id]
        cluster_vals = np.array(self.cluster_counts.values())

        if cur_cluster > -1:
            # exclude the points were considering moving from the dp calculation
            # if the tuple was already in a cluster
            cur_index = self.cluster_counts.index(cur_cluster)
            cluster_vals[cur_index] -= seg_size

        N = cluster_vals.sum()

        loggamma_N_alpha = ss.loggamma(N + self.alpha)
        loggamma_N_alpha_seg = ss.loggamma(N + self.alpha - seg_size)
        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            prior_results[i] = ss.loggamma(seg_size + nc) + loggamma_N_alpha_seg - (ss.loggamma(nc) + loggamma_N_alpha)

        # the prior prob of starting a new cluster
        prior_new = ss.gammaln(seg_size) + np.log(self.alpha) + loggamma_N_alpha_seg - loggamma_N_alpha
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

    # for assigning greylisted segments after the clustering is complete
    def assign_greylist(self):
        # keep a copy of this since it will remain static
        ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])
        # make a deep copy of the cluster dict since we dont want assignment of greylisted clusters to affect subsequent assignments
        greylist_added_dict = sc.SortedDict({k: v.copy() for k, v in self.cluster_dict.items()})

        for segID in self.greylist_segments:

            # compute ML of every cluster if S joins
            ML_BC = np.array([self._ML_cluster_add_one(k, segID) for k in self.cluster_counts.keys()])

            # likelihood ratios of S joining each other cluster S -> Ck
            ML_rat = ML_BC - ML_C

            # currently we do not support prior draws here
            # construct transition probability distribution and draw from it
            log_count_prior = self.DP_tuple_split_prior(segID)[:-1]
            MLs_max = (ML_rat + log_count_prior).max()
            choice_p = np.exp(ML_rat + log_count_prior - MLs_max) / np.exp(
                ML_rat + log_count_prior - MLs_max).sum()
            choice_idx = np.random.choice(
                np.r_[0:len(ML_rat)],
                p=choice_p
            )
            choice = np.r_[self.cluster_dict.keys()][choice_idx]
            choice = int(choice)
            greylist_added_dict[choice].add(segID)

        # now set cluster dict to the one with the greylisted items assigned
        self.cluster_dict = greylist_added_dict

    def run(self, n_iter, sample_num=0):

        burned_in = False
        all_assigned = False

        n_it = 0
        n_it_last = 0

        white_segments = set(range(self.num_segments)) - self.greylist_segments

        while len(self.bins_to_clusters) < n_iter:

            self.save_ML_total()
            # status update
            if not n_it % 250 and self.prior_clusters is None:
                print("n unassigned: {}".format(len(self.unassigned_segs)))

            # start couting for burn in
            if not n_it % 100:
                # self.cdict_history.append(self.cluster_dict.copy())
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
                    ML_A = self._ML_cluster_remove_one(clustID, segID)
                # compute ML of S on its own
                ML_S = self._ML_cluster_from_list([segID])

                # compute ML of every other cluster C = Ck, k != s (cached)
                # for now were also allowing it to chose to stay in current cluster

                ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])

                # compute ML of every cluster if S joins
                ML_BC = np.array([self._ML_cluster_add_one(k, segID) for k in self.cluster_counts.keys()])

                # likelihood ratios of S joining each other cluster S -> Ck
                ML_rat_BC = ML_A + ML_BC - (ML_AB + ML_C)

                # if cluster is unassigned we set the ML ratio to 1 for staying in its own cluster
                if clustID > -1:
                    ML_rat_BC[list(self.cluster_counts.keys()).index(clustID)] = 0

                # compute ML of S starting a new cluster
                ML_new = ML_A + ML_S - ML_AB

                ML_rat = np.r_[ML_rat_BC, ML_new]

                # currently we do not support prior draws here
                # construct transition probability distribution and draw from it
                log_count_prior = self.DP_tuple_split_prior(segID)
                MLs_max = (ML_rat + log_count_prior).max()
                choice_p = np.exp(ML_rat + log_count_prior - MLs_max) / np.exp(
                    ML_rat + log_count_prior - MLs_max).sum()

                if np.isnan(choice_p.sum()):
                    print('skipping iteration {} due to nan. picked segment {}'.format(n_it, segID))
                    n_it += 1
                    continue
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                choice = np.r_[self.cluster_counts.keys(), self.next_cluster_index][choice_idx]
                choice = int(choice)
                # if choice == next_cluster_idx then start a brand new cluster with this new index
                # if choice < 0 then we create a new cluster with the index of the old cluster
                if choice == self.next_cluster_index:
                    if clustID > -1:
                        # if the segment used to occupy a cluster by itself, do nothing if its a new cluster
                        if len(self.cluster_dict[clustID]) == 1:
                            n_it += 1
                            continue
                        else:
                            # otherwise seg was previously assigned so remove it from previous cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_datapoints[clustID] = self._cluster_gen_remove_one(clustID, segID)
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
                    else:
                        # if it wasn't previously assigned we need to remove it from the unassigned list
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
                        n_it += 1
                        continue

                    # joining existing cluster

                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] += self.segment_counts[segID]
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]
                    # TODO possibly change to faster sorted insertion
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
                    continue

                clust_pick = np.random.choice(self.cluster_dict.keys())
                clust_pick_segs = np.r_[self.cluster_dict[clust_pick]].astype(int)

                # half the time we'll propose splitting this cluster
                if np.random.rand() < 0.5:
                    # if theres only one tuple then we cant split
                    if len(clust_pick_segs) == 1:
                        n_it += 1
                        continue

                    # find the best place to split these tuples based on their datapoint means
                    seg_means = np.array([self.segment_r_list[i].mean() for i in clust_pick_segs])
                    sort_indices = np.argsort(seg_means)
                    sorted_vals = seg_means[sort_indices]

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

                    A_list = sorted_segs[:split_ind + 1]
                    B_list = sorted_segs[split_ind + 1:]

                    # add ML ratios to get the likelihood of splitting
                    ML_tot = tot_list[split_ind]

                    # we split with probability equal to this likelihood
                    # to avoid overflow with large positive lls
                    if ML_tot >= 0:
                        split_prob = 1
                    else:
                        split_prob = np.exp(ML_tot)
                    if np.random.rand() < split_prob:
                        # split these clusters

                        # update cluster pick to include only segments from list A
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

                # otherwise we'll propose a merge
                else:
                    # get ML of this cluster merged with each of the other existing clusters

                    ML_join = [self._ML_cluster_merge(i, clust_pick) if i != clust_pick else
                               self.cluster_MLs[i] for i in self.cluster_dict.keys()]
                    # we need to compare this ML with the ML of leaving the target cluster and the picked cluster on their own
                    ML_split = np.array(self.cluster_MLs.values()) + self.cluster_MLs[clust_pick]
                    ML_split[self.cluster_MLs.keys().index(clust_pick)] = self.cluster_MLs[clust_pick]
                    ML_rat = np.array(ML_join) - ML_split

                    count_prior = np.r_[self.DP_merge_prior(clust_pick)]

                    # construct transition probability distribution and draw from it
                    MLs_max = (ML_rat + count_prior).max()
                    choice_p = np.exp(ML_rat + count_prior - MLs_max) / np.exp(
                        ML_rat + count_prior - MLs_max).sum()

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

                    choice = np.r_[self.cluster_counts.keys()][choice_idx]
                    choice = int(choice)

                    if choice != clust_pick:
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
                        self.cluster_datapoints[merged_ID] = self._cluster_gen_from_list(
                            self.cluster_dict[merged_ID])

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

        #assign the greylisted segments
        self.assign_greylist()

        # return the clusters from the last draw and the counts
        return self.clusters_to_segs

    # by default uses last sample
    def visualize_ACDP(self, save_path):
        # precompute the fallback ACDP numbers
        ADP_dict = {}
        for ADP, group in self.cov_df.loc[self.cov_df.dp_draw == 0].groupby('allelic_cluster'):
            ADP_dict[ADP] = (group['maj_count'].sum(), group['min_count'].sum())

        # compute chr ends for plotting
        allelic_segs = pd.read_pickle(self.allelic_segs_path)
        chrbdy = allelic_segs.dropna().loc[:, ["start", "end"]]
        chr_ends = chrbdy.loc[chrbdy["start"] != 0, "end"].cumsum()

        # helper function for plotting
        def _scatter_apply(_x, _minor, _major):
            _f = np.zeros(len(x))
            _f[x.allele == -1] = _minor / (_minor + _major)
            _f[x.allele == 1] = _major / (_minor + _major)
            centers = x.start_g.values + (x.end_g.values - x.start_g.values) / 2
            return centers, _f

        plt.figure(6, figsize=[19.2, 5.39])
        plt.clf()
        full_df = list(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        for i, c in enumerate(self.cluster_dict.keys()):
            for seg in self.cluster_dict[c]:
                x = full_df[seg][1].loc[:,
                    ["start_g", "end_g", 'allelic_cluster', 'cov_DP_mu', 'allele', 'maj_count', 'min_count']]
                adp = x['allelic_cluster'].values[0]
                if len(x) > 10:
                    major, minor = x['maj_count'].sum(), x['min_count'].sum()
                else:
                    major, minor = ADP_dict[adp]

                locs, f = _scatter_apply(x, minor, major)
                y = np.exp(x.cov_DP_mu)
                plt.scatter(
                    locs,
                    f * y,
                    color=np.array(colors)[i % len(colors)],
                    marker='.',
                    alpha=0.03,
                    s=4
                )
        for chrbdy in chr_ends[:-1]:
            plt.axvline(chrbdy, color='k')

        plt.xlabel("Genomic position")
        plt.ylabel("Coverage of major/minor alleles")

        plt.xlim((0.0, 2879000000.0))
        plt.ylim([0, 300])

        plt.savefig(os.path.join(save_path, 'acdp_genome_plot.png'))

        #plot individual tuples within clusters
        rs = []
        for c in self.cluster_dict:
            rs.append(
                (np.array([np.array(self.segment_r_list[i]).mean() for i in self.cluster_dict[c]]).mean(), c))

        f, ax = plt.subplots(1, figsize=[19.2, 10])
        #plt.clf()
        counter = 0
        cc = 0
        for c in [t[1] for t in sorted(rs)]:
            c0 = counter
            vals = [np.array(self.segment_r_list[i]) for i in self.cluster_dict[c]]

            for arr in vals:
                ax.scatter(np.repeat(counter, len(arr)), arr, marker='.', alpha=0.05, s=4)
                counter += 1
            ax.add_patch(mpl.patches.Rectangle((c0, 0), counter - c0, 300, fill=True, alpha=0.10,
                                               color=colors[cc % len(colors)]))
            if self.cluster_counts[c] > 2000:
                ax.text(c0 + (counter - c0) / 2, 0, '{}'.format(c), horizontalalignment='center')
            cc += 1

        plt.savefig(os.path.join(save_path, 'acdp_tuples_plot.png'))

        #simple clusters plot
        f, ax = plt.subplots(1, figsize=[19.2, 10])
        #plt.clf()

        counter = 0
        for c in [t[1] for t in sorted(rs)]:
            vals = [np.array(self.segment_r_list[i]).mean() for i in self.cluster_dict[c]]
            ax.scatter(np.r_[counter:counter + len(vals)], vals)
            counter += len(vals)

        plt.savefig(os.path.join(save_path, 'acdp_clusters_plot.png'))
