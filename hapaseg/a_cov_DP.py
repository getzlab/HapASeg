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
            endog = np.exp(np.log(r) - (C @ dp_pickle.beta).flatten())
            exog = np.ones(r.shape)
            sNB = statsNB(endog, exog)
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
    def __init__(self, cov_df, beta, allelic_segs_path):
        self.cov_df = cov_df
        self.beta = beta
        self.allelic_segs_path = allelic_segs_path

        self.num_segments = len(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        self.segment_r_arr = np.zeros(self.num_segments, dtype=float)
        self.segment_V_arr = np.zeros(self.num_segments, dtype=float)
        self.segment_counts = np.zeros(self.num_segments, dtype=int)
        self.segment_cov_bins = np.zeros(self.num_segments, dtype=int)
        self.segment_allele = np.zeros(self.num_segments, dtype=int)
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
        self._init_clusters()

        # containers for saving the MCMC trace_cov_dp
        self.clusters_to_segs = []
        self.bins_to_clusters = []

        self.alpha = 0.1

    # initialize each segment object with its data
    def _init_segments(self):
        fallback_counts = sc.SortedDict({})
        for ID, (name, grouped) in enumerate(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw'])):
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
                f = minor / (minor + major)
                a = minor
                b = major
            else:
                f = major / (minor + major)
                a = major
                b = minor
            r = np.exp(mu) * f

            # V2 mean uses an empirical estimate of the f * mu posterior variance
            V = (np.exp(s.norm.rvs(mu, sigma, size=10000)) * s.beta.rvs(a,b, size=10000)).var()

            # we scale and threshold this variance (somewhat aribitrarily) to translate the dynamic range into allelic
            # coverage space
            self.segment_V_arr[ID] = min(np.sqrt(V) * 8, 20)
            self.segment_r_arr[ID] = r 
            self.segment_cov_bins[ID] = group_len
            self.segment_counts[ID] = 1

    def _init_clusters(self):
        # we currently ommit greylisted segments
        [self.unassigned_segs.discard(seg) for seg in self.greylist_segments]
        first = (set(range(self.num_segments)) - self.greylist_segments)[0]
        self.cluster_counts[0] = self.segment_counts[0]
        self.unassigned_segs.discard(0)
        self.cluster_dict[0] = sc.SortedSet([first])
        self.cluster_MLs[0] = self._ML_cluster([first])
        #next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
        self.next_cluster_index = 1

    def _ML_cluster(self, cluster_set):
        r = self.segment_r_arr[cluster_set]
        V = self.segment_V_arr[cluster_set]
        # the V_scale parameter is the weighted (by coverage bin size) average of the tuple V2 variances
        V_scale = (V * self.segment_cov_bins[cluster_set] / self.segment_cov_bins[cluster_set].sum()).sum()
        # we scale the prior importance with the number of tuples with a floor of 10
        alpha = max(10,len(r) / 2)
        beta = alpha/2 * V_scale
        return self.ML_normalgamma(r, r.mean(), 1e-4, alpha, beta)

    #worker function for normal-gamma distribution log Marginal Liklihood
    def ML_normalgamma(self, x, mu0, kappa0, alpha0, beta0):
        x_mean = x.mean()
        n = len(x)

        #mu_n = (kappa0*mu0 + n * x_mean) / (kappa0 + n)
        kappa_n = kappa0 + n
        alpha_n = alpha0 + n/2
        beta_n = beta0 + 0.5 * ((x - x_mean)**2).sum() + kappa0 * n * (x_mean - mu0)**2 / 2*(kappa0 + n)

        return ss.loggamma(alpha_n) - ss.loggamma(alpha0) + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n) + np.log(kappa0 / kappa_n) / 2 - n * np.log(2*np.pi) / 2

    # may need to update this if we want to do multiple acdp draws
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

    def save_ML_total(self):
        ML_tot = np.r_[self.cluster_MLs.values()].sum()
        self.ML_total_history.append(ML_tot)

    # function for computing new DP merge prior
    def DP_merge_prior(self, cur_cluster):
        cur_index = self.cluster_counts.index(cur_cluster)
        cluster_vals = np.array(self.cluster_counts.values())
        N = cluster_vals.sum()
        M = cluster_vals[cur_index]
        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            if i != cur_index:
                prior_results[i] = ss.loggamma(M + nc) + ss.loggamma(N + self.alpha - M) - (ss.loggamma(nc) + ss.loggamma(N + self.alpha))
        prior_results[cur_index] = np.log(1-np.exp(LSE(prior_results[prior_results != 0])))
        return prior_results

    # TODO: will need to update this for acdp
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

    def run(self, n_iter):

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
                if not all_assigned and (self.cluster_assignments[white_segments] > -1).all():
                    all_assigned = True
                    n_it_last = n_it

                # burn in after n_seg / n_clust iteration
                if not burned_in and all_assigned and n_it - n_it_last > max(1000, self.num_segments):
                    if np.diff(np.r_[self.ML_total_history[-1000:]]).mean() <= 0:
                        print('burnin')
                        burned_in = True
                        n_it_last = n_it
                        
            # pick either a segment or a cluster at random
            # here we pick a segment
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
                    ML_A = self._ML_cluster(self.cluster_dict[clustID].difference([segID]))

                # compute ML of S on its own
                ML_S = self._ML_cluster([segID])

                # compute ML of every other cluster C = Ck, k != s (cached)
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
                    [count_prior[self.prior_clusters.index(x)] for x in
                     prior_diff], self.cluster_counts.values(), self.segment_counts[segID] * self.alpha]
                count_prior /= count_prior.sum()
                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()

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
                            # if seg was previously assigned remove it from previous cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_dict[clustID].discard(segID)

                            self.cluster_MLs[clustID] = ML_A
                    else:
                        self.unassigned_segs.discard(segID)

                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] = self.segment_counts[segID]
                    self.cluster_dict[choice] = sc.SortedSet([segID])
                    self.cluster_MLs[choice] = ML_S
                    self.next_cluster_index += 1
                else:
                    # if remaining in same cluster, skip
                    if clustID == choice:
                        n_it += 1
                        continue

                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] += self.segment_counts[segID]
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]

                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if len(self.cluster_dict[clustID]) == 1:
                            del self.cluster_counts[clustID]
                            del self.cluster_dict[clustID]
                            del self.cluster_MLs[clustID]

                        else:
                            # otherwise update former cluster
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
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

                    clust_prior_p = np.maximum(
                        np.exp(prior_MLs - prior_MLs.max()) / np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                    # expand MLs to account for multiple new merge clusters--which have liklihood = cluster staying as is = 0
                    ML_rat = np.r_[np.full(len(prior_diff), 0), ML_rat]
                    # DP prior based on clusters sizes now with no alpha

                #will need to change this when we incorporate muliple smaples
                count_prior = np.r_[
                    [count_prior[self.prior_clusters.index(x)] for x in prior_diff], self.DP_merge_prior(clust_pick)]

               # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + count_prior + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + count_prior + np.log(clust_prior_p)).sum()

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

    # by default uses last sample
    def visualize_ACDP(self, save_path, sample_num=-1):

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
        for c in self.cluster_dict.keys():
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
                    color=np.array(colors)[c % len(colors)],
                    marker='.',
                    alpha=0.03,
                    s=4
                )
        for chrbdy in chr_ends[:-1]:
            plt.axvline(chrbdy, color='k')

        plt.xlabel("Genomic position")
        plt.ylabel("Coverage of major/minor alleles")

        plt.xlim((0.0, 2879000000.0));
        plt.ylim([0, 300]);

        plt.savefig(save_path)
