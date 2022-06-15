import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.special as ss
import sortedcontainers as sc

from .model_optimizers import CovLNP_NR, covLNP_ll
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
            cov_df,
            bin_width=1): #set to one for exomes, binsize for genomes
        self.segmentation_draws = segmentation_draws
        self.beta = beta
        self.cov_df = cov_df
        self.bin_exposure = bin_width

        # number of seg samples to use and draws from each DP to take
        self.num_seg_samples = None
        self.num_dp_samples = None

        # Coverage DP run object for each seg sample
        self.DP_runs = None

        self.bins_to_clusters = None
        self.segs_to_clusters = None

    def run_dp(self, 
                num_seg_samples=-1, 
                num_dp_samples=10,
                sample_idx=None): # option to pass sample number to scatter DP jobs

        if num_seg_samples == -1:
            self.num_seg_samples = self.segmentation_draws.shape[1]
        else:
            self.num_seg_samples = num_seg_samples
        
        self.num_dp_samples = num_dp_samples
        
        if sample_idx is not None:
            if num_seg_samples > 1:
                raise ValueError("cannot pass sample number and num_samples")
            if sample_idx > self.segmentation_draws.shape[1]:
                raise ValueError("invalid input segmentation sample number", self.sample_idx)
        
        self.sample_idx = sample_idx
            
        self.DP_runs = [None] * self.num_seg_samples
        prior_run = None
        count_prior_sum = None

        # TODO: load segmentation samples randomly
        self.bins_to_clusters = []
        self.segs_to_clusters = []
        for samp in range(self.num_seg_samples):
            #if we're just doing a single sample change samp to that
            if self.sample_idx is not None:
                samp = self.sample_idx
                run_idx = 0
            else:
                run_idx = samp
            print('starting sample {}'.format(samp))
            self.cov_df['segment_ID'] = self.segmentation_draws[:, samp].astype(int)
            
            DP_runner = Run_Cov_DP(self.cov_df.copy(), self.beta, self.bin_exposure, prior_run, count_prior_sum)
            self.DP_runs[run_idx] = DP_runner
            draws, count_prior_sum = DP_runner.run(self.num_dp_samples, samp)
            self.bins_to_clusters.append(draws)
            prior_run = DP_runner

    def visualize_DP_run(self, run_idx, save_path):
        if run_idx > len(self.DP_runs):
            raise ValueError('DP run index out of range')
        
        cov_dp = self.DP_runs[run_idx]
        cur = 0
        
        # get residuals to compute ylim
        # could save these results to not compute residuals twice but it costs ~2ms
        max_resid = 0
        for c in cov_dp.cluster_dict:
            for seg in cov_dp.cluster_dict[c]:
                resids = np.exp(np.log(cov_dp.segment_r_list[seg]) - (cov_dp.segment_C_list[seg] @ cov_dp.beta + np.log(self.bin_exposure)).flatten())
                max_resid = max(max_resid, resids.max())
        
        f, axs = plt.subplots(1, figsize = (25,10))
        for i, c in enumerate(cov_dp.cluster_dict.keys()):
            clust_start = cur
            for seg in cov_dp.cluster_dict[c]:
                len_seg = len(cov_dp.segment_r_list[seg])
                residuals = np.exp(np.log(cov_dp.segment_r_list[seg]) - (cov_dp.segment_C_list[seg] @ cov_dp.beta + np.log(self.bin_exposure)).flatten())
                axs.scatter(np.r_[cur:len_seg+cur], residuals)
                cur += len_seg
            
            axs.add_patch(mpl.patches.Rectangle((clust_start,0), cur-clust_start, max_resid + 10, fill=True, alpha=0.15, color = colors[i % 10]))
        
        plt.savefig(save_path)

# This class implements the actual DP clustering
# for now input will be coverage_df with global segment id column
class Run_Cov_DP:
    def __init__(self, cov_df, beta, bin_exposure, prior_run=None, count_prior_sum=None):
        self.cov_df = cov_df
        self.seg_id_col = self.cov_df.columns.get_loc('segment_ID')
        self.beta = beta
        self.bin_exposure=bin_exposure
        self.covar_cols = sorted(self.cov_df.columns[self.cov_df.columns.str.contains("^C_.*_z$|^C_log_len$")])
        
        self.num_segments = self.cov_df.iloc[:, self.seg_id_col].max() + 1
        self.segment_r_list = [None] * self.num_segments
        self.segment_C_list = [None] * self.num_segments
        self.segment_sizes = np.zeros(self.num_segments, dtype=int)
        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1

        self.cluster_counts = sc.SortedDict({})
        self.unassigned_segs = sc.SortedList(np.r_[0:self.num_segments])
        self.cluster_dict = sc.SortedDict({})
        self.cluster_MLs = sc.SortedDict({})

        self.segment_idxs = sc.SortedSet({})
        self.blacklisted_segments = sc.SortedSet({})
        self.greylist_segments = sc.SortedSet({})

        self.cluster_LLs = sc.SortedDict({})
        self.cluster_ml_cache={}
        self.cluster_prior_ml_cache={}
        
        self.prior_clusters = None
        self.prior_r_list = None
        self.prior_C_list = None
        self.count_prior_sum = None

        # scale factor for dp prior to be set later
        self.dp_count_scale_factor = 1
        
        self._init_segments()
        self._init_clusters(prior_run, count_prior_sum)
        
        # containers for saving the MCMC trace
        self.clusters_to_segs = []
        self.bins_to_clusters = []
        self.ll_history = []
        self.alpha = 0.1
        
    # initialize each segment object with its data
    def _init_segments(self):
        num_segments = 0
        for ID, seg_df in self.cov_df.groupby('segment_ID'):
            seg = seg_df['covcorr'].values
            seg_len = len(seg)
            self.segment_r_list[ID] = seg
            
            if seg_len < 3:
                # this is too few coverage bins to confidently fit NB to
                # we will throw these segments out for now
                self.blacklisted_segments.add(ID)
                self.unassigned_segs.discard(ID)
                continue     
            
            self.segment_C_list[ID] = np.c_[seg_df[self.covar_cols]]
            self.segment_sizes[ID] = seg_len
            num_segments += 1
            
            # if we don't have many bins in a segment we'll greylist them to
            # establish clusters from larger segments
            if seg_len < 10:
                self.greylist_segments.add(ID)
                self.unassigned_segs.discard(ID)
        
        self.segment_idxs = set(range(self.num_segments)) - self.blacklisted_segments

        # set dp scale factor such that the average segment count is set to one
        self.dp_count_scale_factor =  self.segment_sizes[self.segment_idxs].mean()

    def _init_clusters(self, prior_run, count_prior_sum, warm_start=True):
        # if first iteration then add first segment to first cluster
        if prior_run is None:
            if not warm_start:
                # get first cluster ID since 0 may be blacklisted
                first_seg = self.segment_idxs[0] 
                self.cluster_counts[0] = self.segment_sizes[first_seg]
                self.cluster_assignments[first_seg] = 0
                self.unassigned_segs.discard(first_seg)
                self.cluster_dict[0] = sc.SortedSet([first_seg])
                self.cluster_MLs[0] = self._ML_cluster([first_seg])
                # next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
                self.next_cluster_index = 1
            else:
                #initialize each segment to its own cluster
                for ID in self.segment_idxs:
                    # dont seed clusters from greylisted segments
                    if ID in self.greylist_segments:
                        continue
                    self.cluster_counts[ID] = self.segment_sizes[ID]
                    self.unassigned_segs.discard(ID)
                    self.cluster_dict[ID] = sc.SortedSet([ID])
                    self.cluster_MLs[ID] = self._ML_cluster([ID])
                    self.cluster_assignments[ID] = ID
 
                self.next_cluster_index = ID + 1
        else:
            #otherwise we initialize the prior clusters
            self.prior_clusters = prior_run.cluster_dict.copy()
            self.prior_r_list = prior_run.segment_r_list.copy()
            self.prior_C_list = prior_run.segment_C_list.copy()
            self.count_prior_sum = count_prior_sum.copy()
            self.next_cluster_index = np.r_[self.prior_clusters.keys()].max() + 1
            self.clust_prior_ML = None
    
    @staticmethod
    def stats_ll_nbinom(r, mu, C, beta, lepsi, bin_exposure):
        r = r.flatten()
        epsi = np.exp(lepsi)
        exposure = np.log(bin_exposure)
        bc = (C @ beta).flatten() + exposure
        exp = np.exp(mu + bc).flatten()
        return (ss.gammaln(r + epsi) - ss.gammaln(r + 1) - ss.gammaln(epsi) +
                (r * (mu + bc - np.log(epsi + exp))) +
                (epsi * np.log(epsi / (epsi + exp)))).sum()
    
    @staticmethod
    def ll_nbinom(r, mu, C, beta, lepsi, bin_exposure=1):
        return covLNP_ll(r[:,None], mu, lepsi, C, beta, np.log(bin_exposure)).sum()
    
    # main worker function for computing marginal likelihoods of clusters
    #TODO: move to a bounded size LFU chache over dictionary?
    def _ML_cluster(self, cluster_set):
        fs = frozenset(cluster_set)
        if fs in self.cluster_ml_cache:
            return self.cluster_ml_cache[fs]
        else:
            # aggregate r and C arrays
            r = np.hstack([self.segment_r_list[i] for i in cluster_set])
            C = np.concatenate([self.segment_C_list[i] for i in cluster_set])
            mu_opt, lepsi_opt, H_opt = self.lnp_optimizer(r, C, ret_hess=True)
            ll_opt = self.ll_nbinom(r, mu_opt, C, self.beta, lepsi_opt, self.bin_exposure)
            
            res = ll_opt + self._get_laplacian_approx(H_opt)
            self.cluster_ml_cache[fs] = res
            return res

    # computes the log likelihood for a set of segments comprising a cluster
    def _LL_cluster(self, cluster_set):
        # aggregate r and C arrays
        r = np.hstack([self.segment_r_list[i] for i in cluster_set])
        C = np.concatenate([self.segment_C_list[i] for i in cluster_set])
        mu_opt, lepsi_opt = self.lnp_optimizer(r, C, ret_hess=False)
        ll_opt = self.ll_nbinom(r, mu_opt, C, self.beta, lepsi_opt, self.bin_exposure)

        return ll_opt

    # computes the ML of a cluster with some clusters optionally containing 
    # segments from previous iterations (i.e. prior clustering)
    def _ML_cluster_prior(self, cluster_set, new_segIDs=None):
        if new_segIDs is None:
            fs_new = None
        else:
            fs_new = frozenset(new_segIDs)
        query = (frozenset(cluster_set), fs_new)
        
        if query in self.cluster_prior_ml_cache:
            return self.cluster_prior_ml_cache[query]
        else:
            # aggregate r and C arrays
            if new_segIDs is not None:
                r_new = np.hstack([self.segment_r_list[s] for s in new_segIDs])
                C_new = np.concatenate([self.segment_C_list[s] for s in new_segIDs])
                r = np.r_[np.hstack([self.prior_r_list[i] for i in cluster_set]), r_new]
                C = np.r_[np.concatenate([self.prior_C_list[i] for i in cluster_set]), C_new]
            else:
                r = np.hstack([self.prior_r_list[i] for i in cluster_set])
                C = np.concatenate([self.prior_C_list[i] for i in cluster_set])

            mu_opt, lepsi_opt, H_opt = self.lnp_optimizer(r, C, ret_hess=True)
            ll_opt = self.ll_nbinom(r, mu_opt, C, self.beta, lepsi_opt, self.bin_exposure)
            res =  ll_opt + self._get_laplacian_approx(H_opt)
            self.cluster_prior_ml_cache[query] = res
            return res

    @staticmethod
    def _get_laplacian_approx(H):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-H))) / 2

    # returns optimal NB parameter values, along with optionally the hessian
    # from the MLE point
    def stats_optimizer(self, r, C, ret_hess=False):
        offset = (C @ self.beta).flatten()
        exog = np.ones(r.shape[0])
        exposure = np.ones(r.shape[0]) * self.bin_exposure
        sNB = statsNB(r, exog, exposure=exposure, offset=offset)
        res = sNB.fit(disp=0)
        if ret_hess:
            return res.params[0], -np.log(res.params[1]), sNB.hessian(res.params)
        else:
            return res.params[0], -np.log(res.params[1])
    
    def lnp_optimizer(self, r, C, ret_hess=False):
        lnp = CovLNP_NR(r[:,None], self.beta, C, exposure = np.log(self.bin_exposure))
        return lnp.fit(ret_hess = ret_hess)

    # for assigning greylisted segments at for each draw
    def assign_greylist(self):
        
        greylist_added_dict = sc.SortedDict({k: v.copy() for k, v in self.cluster_dict.items()})
        greylist_cluster_assignments = self.cluster_assignments.copy()
        greylist_cluster_counts = self.cluster_counts.copy()
        ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])

        for segID in self.greylist_segments:
            
            # compute ML of every cluster if S joins  
            ML_BC = np.array([self._ML_cluster(self.cluster_dict[k].union([segID])) for k in self.cluster_counts.keys()])
            
            # likelihood ratios of S joining each other cluster S -> Ck
            ML_rat = ML_BC - ML_C
            # construct transition probability distribution and draw from it
            log_count_prior = self.DP_move_segment_prior(segID, [])[:-1]
            MLs_max = (ML_rat + log_count_prior).max()
            choice_p = np.exp(ML_rat + log_count_prior - MLs_max) / np.exp(ML_rat + log_count_prior - MLs_max).sum()
            choice_idx = np.random.choice(np.r_[0:len(ML_rat)], p=choice_p)
            choice = np.r_[self.cluster_dict.keys()][choice_idx]
            choice = int(choice)
            
            greylist_added_dict[choice].add(segID)        
            greylist_cluster_assignments[segID] = choice          
            greylist_cluster_counts[choice] += self.segment_sizes[segID]
            
        # with all greylist segments assigned we can now update our clustering data fields
        self.cluster_dict = greylist_added_dict
        self.cluster_assignments = greylist_cluster_assignments
        for clust in self.cluster_dict.keys():
            self.cluster_MLs[clust] = self._ML_cluster(self.cluster_dict[clust])
            self.cluster_LLs[choice] = self._LL_cluster(self.cluster_dict[clust])
        #need to update the cluster counts as well
        self.cluster_counts = sc.SortedDict({c: self.segment_sizes[self.cluster_dict[c]].sum() for c in self.cluster_dict.keys()})
 
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
    
    # updated DP prior for moving a single segment into an existing or new cluster
    def DP_move_segment_prior(self, seg_id, prev_clusters):
        cur_cluster = self.cluster_assignments[seg_id]
        seg_size = self.segment_sizes[seg_id]
        cluster_vals = np.array(self.cluster_counts.values())
        
        # if segment was already assigned, remove its bins from the calculation
        if cur_cluster > -1:
            cur_index = self.cluster_counts.index(cur_cluster)
            cluster_vals[cur_index] -= seg_size

        # expand cluster vals to include counts from old clusters
        cluster_vals = np.r_[prev_clusters, cluster_vals]

        N = cluster_vals.sum()
        
        # apply scale factor
        N = N / self.dp_count_scale_factor
        seg_size = seg_size / self.dp_count_scale_factor

        loggamma_N_alpha = ss.loggamma(N + self.alpha)
        loggamma_N_alpha_seg = ss.loggamma(N + self.alpha - seg_size)
        prior_results = np.zeros(len(cluster_vals))
        
        for i, nc in enumerate(cluster_vals):
            prior_results[i] = ss.loggamma(seg_size + nc) + loggamma_N_alpha_seg - (ss.loggamma(nc) + loggamma_N_alpha)

        # the prior prob of starting a new cluster
        prior_new = ss.gammaln(seg_size) + np.log(self.alpha) + loggamma_N_alpha_seg - loggamma_N_alpha
        return np.r_[prior_results, prior_new]

    # updated DP prior for potentially merging a cluster
    def DP_move_cluster_prior(self, cur_cluster, prev_clusters):
        cur_index = self.cluster_counts.index(cur_cluster)
        cluster_vals = np.array(self.cluster_counts.values())
        
        # expland cluster vals to include counts from old clusters
        cluster_vals = np.r_[prev_clusters, cluster_vals] / self.dp_count_scale_factor
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
        
    def run(self, n_iter, sample_num=0):

        burned_in = False
        all_assigned = False
        greylist_added = False

        n_it = 0
        n_it_last = 0

        if self.prior_clusters:
            self.clust_prior_ML = sc.SortedDict(
                {k: self._ML_cluster_prior(self.prior_clusters[k]) for k in self.prior_clusters.keys()})
            count_prior = np.r_[[v for (k, v) in self.count_prior_sum.items() if
                                 k in self.prior_clusters.keys()]] / sample_num
            self.initial_prior_assignment(count_prior)
        
        white_segments = self.segment_idxs - self.greylist_segments
        while len(self.bins_to_clusters) < n_iter:
            
            # status update
            if not n_it % 250 and self.prior_clusters is None:
                print("n unassigned: {}".format(len(self.unassigned_segs)), flush=True)

            # start couting for burn in
            if not n_it % 100:
                print("n_it:", n_it, flush=True)
                if not all_assigned and (self.cluster_assignments[white_segments] > -1).all():
                    all_assigned = True
                    n_it_last = n_it
                    #start keeping track of the total likelihood after all have been assigned
                    for ID in self.cluster_counts.keys():
                        self.cluster_LLs[ID] = self._LL_cluster(self.cluster_dict[ID])
                
                # burn in based on ll changes after all are assigned
                if not burned_in and all_assigned and n_it - n_it_last > 500:
                    # wait until the chain has stabilized
                    if np.diff(np.r_[self.ll_history[-500:]]).mean() <= 0:
                        # if we havent added back the greylist segments, do so
                        # and then wait to stabilize again
                        if not greylist_added:
                            self.assign_greylist()
                            greylist_added=True
                            n_it_last = n_it
                        else:
                            print('burnin', flush=True)
                            burned_in = True
                            n_it_last = n_it

            # pick either a segment or a cluster at random

            # pick segment
            if np.random.rand() < 0.5:
                if len(self.unassigned_segs) > 0 and len(
                        self.unassigned_segs) / len(white_segments)< 0.1 and np.random.rand() < 0.5:
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
                log_dp_count_prior = self.DP_move_segment_prior(segID, [count_prior[self.prior_clusters.index(x)] for x in prior_diff])

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + log_dp_count_prior + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + log_dp_count_prior + np.log(clust_prior_p)).sum()

                if np.isnan(choice_p.sum()):
                    print('skipping iteration {} due to nan. picked segment {}'.format(n_it, segID), flush=True)
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
                            self.cluster_counts[clustID] -= self.segment_sizes[segID]
                            self.cluster_dict[clustID].discard(segID)

                            self.cluster_MLs[clustID] = ML_A
                            if all_assigned:
                                self.cluster_LLs[clustID] = self._LL_cluster(self.cluster_dict[clustID]) 
                    else:
                        self.unassigned_segs.discard(segID)
                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] = self.segment_sizes[segID]
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
                    self.cluster_counts[choice] += self.segment_sizes[segID]
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]
                    if all_assigned:
                        self.cluster_LLs[choice] = self._LL_cluster(self.cluster_dict[choice])

                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if len(self.cluster_dict[clustID]) == 1:
                            del self.cluster_counts[clustID]
                            del self.cluster_dict[clustID]
                            del self.cluster_MLs[clustID]
                            if all_assigned:
                                del self.cluster_LLs[clustID]
                        else:
                            # otherwise update former cluster
                            self.cluster_counts[clustID] -= self.segment_sizes[segID]
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
                log_dp_count_prior = self.DP_move_cluster_prior(clust_pick, [count_prior[self.prior_clusters.index(x)] for x in prior_diff])

                # construct transition probability distribution and draw from it
                MLs_max = ML_rat.max()
                choice_p = np.exp(ML_rat - MLs_max + log_dp_count_prior + np.log(clust_prior_p)) / np.exp(
                    ML_rat - MLs_max + log_dp_count_prior + np.log(clust_prior_p)).sum()
                if np.isnan(choice_p.sum()):
                    print("skipping iteration {} due to nan".format(n_it), flush=True)
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
                        self.cluster_counts[merged_ID] += self.segment_sizes[vacating_segs].sum()
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
            if burned_in and n_it - n_it_last > len(white_segments) / (len(self.cluster_counts) * 2):
                self.bins_to_clusters.append(self.cluster_assignments)
                self.clusters_to_segs.append(self.cluster_dict)
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
