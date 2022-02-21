import numpy as np
import scipy.special as ss
import sortedcontainers as sc
from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from .model_optimizers import PoissonRegression

# turn off warnings for statsmodels fitting
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
np.seterr(divide='ignore')

colors = mpl.cm.get_cmap("tab10").colors


class AllelicCluster:
    def __init__(self, r, C, mu_0, beta_0):
        # cluster wide params
        self.r = r.flatten()
        self.C = C
        self.mu = mu_0.flatten()
        self.beta = beta_0
        self.lepsi = 1
        self.epsi = np.exp(self.lepsi)

        # individual segment params
        self.mu_i = 0
        self.epsi_i = 0
        self.lepsi_i = 0
        self.exp = 0

        # all segment params
        self.lepsi_i_arr = np.ones(self.r.shape)
        self.mu_i_arr = np.zeros(self.r.shape)

        # all intervals start out in one segment, which also means no breakpoints
        self.segment_ID = np.zeros(r.shape)
        self.segments = sc.SortedSet([0])
        self.segment_lens = sc.SortedDict([(0, len(self.r))])
        
        # keep cache of previously computed breakpoints for fast splitting
        # these breakpoints keys are in the form (st, en, breakpoint)
        self.breakpoint_cache = {}

        self.phase_history = []
        self.F = sc.SortedList()
        self.F.update([0, len(self.r)])

        self.ll_traces = []

        self.opt_views = []
        self.mui_views = []
        self.lepsi_views = []

        self._init_params()

    # fits NB model to this cluster using initial params as starting point
    def _init_params(self):
        self.NR_init()
        self.lepsi_i_arr = np.ones(self.r.shape) * self.lepsi
        self.epsi_i_arr = np.exp(self.lepsi_i_arr)

    # fit initial mu_k and epsilon_k for the cluster
    def NR_init(self):
        self.mu, self.lepsi = self.stats_init()

    def get_seg_ind(self, seg):
        return seg, seg + self.segment_lens[seg]

    def get_join_seg_ind(self, seg_l, seg_r):
        return seg_l, seg_r + self.segment_lens[seg_r]

    def get_ll(self):
        return self.ll_cluster(self.mu_i_arr, self.lepsi_i_arr, True)
    
    # use stats optimizer to set initial cluster mu and lepsi
    def stats_init(self):
        endog = self.r.flatten()
        exog = np.ones(self.r.shape[0])
        sNB = statsNB(endog, exog, offset=(self.C @ self.beta).flatten())
        res = sNB.fit(disp=0)
        return res.params[0], -np.log(res.params[1])

    # statsmodels NB BFGS optimizer is more stable than NR so we will use it until migration to LNP
    def stats_optimizer(self, ind, ret_hess=False):
        endog = self.r[ind[0]:ind[1]].flatten()
        exog = np.ones(self.r[ind[0]:ind[1]].shape[0])
        exposure = np.ones(self.r[ind[0]:ind[1]].shape[0]) * np.exp(self.mu)
        sNB = statsNB(endog, exog, exposure=exposure, offset=(self.C[ind[0]:ind[1]] @ self.beta).flatten())
        res = sNB.fit(disp=0)

        if ret_hess:
            return res.params[0], -np.log(res.params[1]), sNB.hessian(res.params)
        else:
            return res.params[0], -np.log(res.params[1])

    # method for calculating the overall log likelihood of an allelic cluster given a hypothetical mu_i and lepsi arrays
    def ll_cluster(self, mu_i_arr, lepsi_i_arr, take_sum=True):
        mu_i_arr = mu_i_arr.flatten()
        epsi_i_arr = np.exp(lepsi_i_arr).flatten()
        bc = (self.C @ self.beta).flatten()
        exp = np.exp(self.mu + bc + mu_i_arr).flatten()

        lls = (ss.gammaln(self.r + epsi_i_arr) - ss.gammaln(self.r + 1) - ss.gammaln(epsi_i_arr) +
               (self.r * (self.mu + bc + mu_i_arr - np.log(epsi_i_arr + exp))) +
               (epsi_i_arr * np.log(epsi_i_arr / (epsi_i_arr + exp))))
        if not take_sum:
            return lls
        return lls.sum()

    # method for calculating the log likelihood of our NB model
    @staticmethod
    def ll_nbinom(r, mu, C, beta, mu_i, lepsi):
        r = r.flatten()
        epsi = np.exp(lepsi)
        bc = (C @ beta).flatten()
        exp = np.exp(mu + bc + mu_i).flatten()
        return (ss.gammaln(r + epsi) - ss.gammaln(r + 1) - ss.gammaln(epsi) +
                (r * (mu + bc + mu_i - np.log(epsi + exp))) +
                (epsi * np.log(epsi / (epsi + exp)))).sum()

    """
    method for convolving a change kernel with a given window across an array of residuals. This change kernel returns 
    the absolute difference between the means of the residuals within windows on either side of a rolling change point.
    
    returns a list of peaks above the 98th quantile that are seperated by at least max(25, window/2)
    """
    def _change_kernel(self, ind, residuals, window):
        
        difs = []
        for i in np.r_[window:len(residuals) - window]:
            difs.append(np.abs(residuals[i - window:i].mean() - residuals[i:i + window].mean()))
        difs = np.r_[difs]
        if window > 10:
            x = np.r_[:len(difs)]
            prior = np.exp(2 * np.abs(x - len(difs) / 2) / len(difs))
            difs = prior * difs
        
        #cutoff = difs.mean() + 2 * np.sqrt(difs.var())
        cutoff = np.quantile(difs, 0.98)

        # use helper function to find peaks above cutoff that are seperated
        peaks, _ = find_peaks(difs, height=cutoff, distance=max(25, window/2))
        peaks = peaks + ind[0] + window
        return list(peaks)

    """
    method for narrowing search space of possible split positions in a segment. Finds the indices of the top ~2% of 
    the difference kernel values for multiple window sizes in addition to the first and last window bins. Falls back on 
    an exhaustive search, which means returning all possible split points.
    
    returns list of possible split points to try
    """
    #
    def _find_difs(self, ind):
        residuals = np.exp(np.log(self.r[ind[0]:ind[1]].flatten()) - (self.mu.flatten()) - (
                    self.C[ind[0]:ind[1]] @ self.beta).flatten())
       
        len_ind = ind[1]-ind[0]
        windows = np.array([10, 50, 100, 250])
        windows = windows[2*windows < len_ind]
        difs = []
        for window in windows:
            difs.append(self._change_kernel(ind, residuals, window))
        difs_idxs = set().union(*difs)
        difs_idxs = np.r_[list(difs_idxs)]
        # if there are no that pass the threshold return them all
        if len(difs_idxs) ==0:
            print('no significant bins')
            return list(np.r_[ind[0] + 2:ind[1] - 1])

        # add on first and last windows
        difs_idxs = np.r_[np.r_[ind[0] + 2: ind[0] + 10], difs_idxs, np.r_[ind[1] - 10:ind[1] - 1]]
        return list(difs_idxs)

    """
    Given the indices of the segment and the proposed split indices to try, this method computes the likelihood of each
    split proposal and saves the associated mu, lepsi and Hessian results associated to the split segments. 
    
    Returns lists of likelihoods and optimal split parameters
    """
    def _calculate_splits(self, ind, split_indices):
        lls = []
        mus = []
        lepsis = []
        Hs = []
        for ix in split_indices:
            if ix < 0:
                # no split proposal
                ll_join = self.ll_cluster(self.mu_i_arr, self.lepsi_i_arr)
                lls.append(ll_join)
                mus.append(None)
                lepsis.append(None)
                Hs.append(None)
            else:
                mu_l, lepsi_l, H_l = self.stats_optimizer((ind[0], ix), True)
                mu_r, lepsi_r, H_r = self.stats_optimizer((ix, ind[1]), True)

                mus.append((mu_l, mu_r))
                lepsis.append((lepsi_l, lepsi_r))
                Hs.append((H_l, H_r))

                tmp_mui = self.mu_i_arr.copy()
                tmp_mui[ind[0]:ix] = mu_l
                tmp_mui[ix: ind[1]] = mu_r
                tmp_lepsi = self.lepsi_i_arr.copy()
                tmp_lepsi[ind[0]:ix] = lepsi_l
                tmp_lepsi[ix: ind[1]] = lepsi_r

                ll = self.ll_cluster(tmp_mui, tmp_lepsi)
                lls.append(ll)

        return lls, mus, lepsis, Hs

    """
    This method is simpler of the two options for refining the search space of possible splits, which is important for 
    both finding the best split and for properly normalizing the split likelihoods. This neighborhood sampling approach
    simply samples the 5 positions on either side of a split proposal.
    
    returns an amended list of likelihoods and optimal parameters for the original split indicies in addition to the 
    ones sampled from the neighbors of the originals.
    """
    def _neighborhood_sampling(self, ind, lls, split_indices, mus, lepsis, Hs):

        # find the most likely index and those indices within a significant range
        # (rn 7 -> at least 1/1000th as likely)
        top_lik = max(lls)
        within_range = np.r_[lls] > (top_lik - 7)
        significant = np.r_[split_indices][within_range]

            # increase sampling of significantly likely regions
        extra_samples = sc.SortedSet({})
        for s in significant:
            #dont sample around no split index
            if s < 0:
                continue
            for i in np.r_[max(ind[0] + 2, s - 5):min(ind[1] - 1, s + 5)]:
                extra_samples.add(i)
        
        # no need to resample things we already calculated
        extra_samples = list(extra_samples - set(significant))
        lls_add, mus_add, lepsis_add, Hs_add = self._calculate_splits(ind, extra_samples)
        lls.extend(lls_add)
        mus.extend(mus_add)
        lepsis.extend(lepsis_add)
        Hs.extend(Hs_add)
        split_indices.extend(extra_samples)
        
        return lls, split_indices, mus, lepsis, Hs

    """
        This method is used for refining the search space of possible splits, which is important for 
        both finding the best split and for properly normalizing the split likelihoods. It works by iteratively sampling
        points in either direction of a proposed split until it reaches a potential split point which is less than 1/1k 
        as likely as the most likely point seen so far.

        returns an amended list of likelihoods and optimal parameters for the original split indicies in addition to the 
        ones sampled.
        """
    def _detailed_sampling(self, ind, lls, split_indices, mus, lepsis, Hs):
        max_samples = 25
        # find the most likely index and those indices within a significant range
        # (rn 7 -> at least 1/1000th as likely)
        top_lik = max(lls)
        sig_threshold = top_lik - 5
        within_range = np.r_[lls] > sig_threshold
        sig_idx = np.r_[split_indices][within_range]
        sig_set = set(sig_idx)
        num_samps = 0
        for ix in sig_idx:
            #dont sample areound the split index
            if ix < 0:
                continue
            #sample to the left of the position until were out of the significant range
            ix_l = ix - 1
            r_samples = 0
            while r_samples < max_samples:
                if ix_l < ind[0] + 2:
                    break
                ll_add, mu_add, lepsi_add, Hs_add = self._calculate_splits(ind, [ix_l])
                if ll_add[0] > sig_threshold and ix_l not in sig_set:
                    split_indices.append(ix_l)
                    lls.extend(ll_add)
                    mus.extend(mu_add)
                    lepsis.extend(lepsi_add)
                    Hs.extend(Hs_add)
                    
                    sig_set.add(ix_l)
                    ix_l -= 1
                    num_samps+=1
                    r_samples += 1
                else:
                    break
            
            #now do the same for the right side
            ix_r = ix + 1
            l_samples = 0
            while l_samples < max_samples:
                if ix_r > ind[1] -1:
                    break
                ll_add, mu_add, lepsi_add, Hs_add = self._calculate_splits(ind, [ix_r])
                if ll_add[0] > sig_threshold and ix_r not in sig_set:
                    split_indices.append(ix_r)
                    lls.extend(ll_add)
                    mus.extend(mu_add)
                    lepsis.extend(lepsi_add)
                    Hs.extend(Hs_add)

                    sig_set.add(ix_r)
                    ix_r += 1
                    num_samps+=1
                    l_samples +=1
                else:
                    break
        return lls, split_indices, mus, lepsis, Hs

    # function for calculating the log marginal likelihood of a split given the log likelihood and the hessians for
    # the NB fit of each split segment
    def _lls_to_MLs(self, lls, Hs):
        MLs = np.zeros(len(lls))
        for i, (ll, Hs) in enumerate(zip(lls, Hs)):
            laplacian = self._get_log_ML_split(Hs[0], Hs[1])
            # the split results in a nan make it impossible to split there
            if np.isnan(laplacian):
                laplacian = -1e50
            MLs[i] = ll + laplacian
        return MLs

    """
    Function for computing log MLs for all possible split points of interest. If a segment is small enough we will 
    consider every possible split point, otherwise will use change kernel and detailed sampling
    """
    def _get_split_liks(self, ind, debug=False):
        ind_len = ind[1] - ind[0]
        # do not allow segments to be split into segments of size less than 2
        if ind_len < 4 and not debug:
            raise Exception('segment was too small to split: length: ', ind[1] - ind[0])

        # if the cluster is sufficiently small we interrogate the whole space of splits
        if ind_len < 150:
            split_indices = list(np.r_[ind[0] + 2: ind[1] - 1])
            lls, mus, lepsis, Hs = self._calculate_splits(ind, split_indices)
        # otherwise we deploy our change kernel to limit our search space
        else:
            split_indices = self._find_difs(ind)

            # get lls from the difference kernel guesses
            lls, mus, lepsis, Hs = self._calculate_splits(ind, split_indices)
            lls, split_indices, mus, lepsis, Hs = self._detailed_sampling(ind, lls, split_indices, mus, lepsis, Hs)
        MLs = self._lls_to_MLs(lls, Hs)

        return split_indices, MLs, mus, lepsis

    # computes ML component from hessian approximation for a single segment
    def _get_log_ML_approx_join(self, Hess):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-Hess))) / 2

    # computes ML component from hessian approximation for two split segments
    def _get_log_ML_split(self, H1, H2):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-H1) * np.linalg.det(-H2))) / 2

    # computes the log ML of joining two segments
    def _log_ML_join(self, ind, ret_opt_params=False):
        mu_share, lepsi_share, H_share = self.stats_optimizer(ind, True)
        tmp_mui = self.mu_i_arr.copy()
        tmp_mui[ind[0]:ind[1]] = mu_share
        tmp_lepsi = self.lepsi_i_arr.copy()
        tmp_lepsi[ind[0]:ind[1]] = lepsi_share
        ll_join = self.ll_cluster(tmp_mui, tmp_lepsi)
        if ret_opt_params:
            return mu_share, lepsi_share, self._get_log_ML_join(H_share) + ll_join
        return mu_share, lepsi_share, self._get_log_ML_approx_join(H_share) + ll_join

    """
    Split segment method. This method chooses a segment at random
    from the allelic cluster, and calculates the MLs of splitting the segment either every possible position or positions
    informed by the change kernel heuristic. Compares the MLs of splitting the segment at each position with the ML of 
    the joint cluster and probabilistically takes an action proportional to likelihood. 
    
    Returns -1 if splitting was skipped due to segment being too small to split (<4 bins), 0 if the choice was not to
    leave the cluster as is, and <breakpoint> if the segment was split at <breakpoint>
    """
    def split(self, debug):

        # pick a random segment
        seg = np.random.choice(self.segments)

        # if segment is a singleton then skip
        ## if segment is less than 4 bins then skip
        if self.segment_lens[seg] < 4:
            return -1

        ind = self.get_seg_ind(seg)

        split_indices, log_split_MLs, all_mus, all_lepsis = self._get_split_liks(ind, debug=debug)
        _, _, log_join_ML = self._log_ML_join(ind)

        log_MLs = np.r_[log_split_MLs, log_join_ML]
        split_indices.append(-1)

        max_ML = max(log_MLs)
        k_probs = np.exp(log_MLs - max_ML) / np.exp(log_MLs - max_ML).sum()
        choice_idx = np.random.choice(len(split_indices), p=k_probs)
        break_idx = split_indices[choice_idx]
        if break_idx > 0:
            # split the segments
            mus = all_mus[choice_idx]
            lepsis = all_lepsis[choice_idx]
            # update segment assignments for new segment
            self.segment_ID[break_idx:ind[1]] = break_idx
            self.segments.add(break_idx)
            self.segment_lens[break_idx] = ind[1] - break_idx

            # update old seglen
            self.segment_lens[ind[0]] = break_idx - ind[0]

            # update mu_i and lepsi_i values
            self.mu_i_arr[ind[0]:break_idx] = mus[0]
            self.mu_i_arr[break_idx:ind[1]] = mus[1]
            self.lepsi_i_arr[ind[0]:break_idx] = lepsis[0]
            self.lepsi_i_arr[break_idx:ind[1]] = lepsis[1]

            self.F.update([break_idx, break_idx])
            self.phase_history.append(self.F.copy())
            return break_idx
        # otherwise we have chosen to join and we do nothing
        return 0

    """
    Join segments method. This method chooses a segment at random from the allelic cluster, and probabilistically 
    chooses to join that segment with its neighbor to the right. This action is taken with probability equal to the 
    ratio of the joint and split MLs
    
    returns -1 if there is only one segment, 0 if the proposed join is rejected, and <breakpoint> if a join occurred,
    where <breakpoint is the left most index of the right segment joined.
    """

    def join(self, debug):
        num_segs = len(self.segments)
        # if theres only one segment, skip
        if num_segs == 1:
            return -1

        # otherwise pick a left segment to join
        seg_l_ind = np.random.choice(num_segs - 1)
        seg_l = self.segments[seg_l_ind]
        seg_r = self.segments[seg_l_ind + 1]
        ind = self.get_join_seg_ind(seg_l, seg_r)

        lls_split, _, _, Hs = self._calculate_splits(ind, [seg_r])
        log_split_ML = lls_split[0] + self._get_log_ML_split(Hs[0][0], Hs[0][1])
        mu_share, lepsi_share, log_join_ML = self._log_ML_join(ind)

        log_MLs = np.r_[log_split_ML, log_join_ML]
        max_ML = max(log_MLs)
        k_probs = np.exp(log_MLs - max_ML)/np.exp(log_MLs - max_ML).sum()

        join_choice = np.random.choice([0, 1], p=k_probs)

        if join_choice:
            # join the segments
            # update segment assignments for new segment
            self.segment_ID[ind[0]:ind[1]] = seg_l
            self.segments.discard(seg_r)
            del self.segment_lens[seg_r]
            self.segment_lens[seg_l] = ind[1] - ind[0]
            # update mu_i and lepsi_i values
            self.mu_i_arr[ind[0]:ind[1]] = mu_share
            self.lepsi_i_arr[ind[0]:ind[1]] = lepsi_share

            # we need to discard twice since the breakpoint is saved
            # both as an end to an interval and a start to the next interval
            self.F.discard(seg_r)
            self.F.discard(seg_r)
            self.phase_history.append(self.F.copy())
            return seg_r

        return 0


class NB_MCMC_AllClusters:

    """
    This class is for running segmentation on all allelic clusters on the same node. Each iteration first randomly
    chooses a cluster, and then proposes and split/join operation.
    """
    def __init__(self, n_iter, r, C, Pi):
        self.n_iter = n_iter
        self.r = r
        self.C = C
        self.Pi = Pi
        self.beta = None

        # for now assume that the Pi vector assigns each bin to exactly one cluster
        self.c_assignments = np.argmax(self.Pi, axis=1)
        self.n_clusters = Pi.shape[1]
        self.clusters = [None] * self.n_clusters
        self.cluster_sizes = np.array([np.sum(self.c_assignments == i) for i in range(self.n_clusters)])
        self.cluster_probs = np.ones(self.n_clusters) / self.n_clusters

        self.burnt_in = False

        self.num_segments = np.ones(self.n_clusters)

        self.mu_i_samples = []
        self.lepsi_i_samples = []
        self.F_samples = []

        self.ll_clusters = np.zeros(self.n_clusters)
        self.ll_iter = []
        self._init_clusters()

    def _init_clusters(self):
        # first we find good starting values for mu and beta for each cluster by fitting a poisson model
        pois_regr = PoissonRegression(self.r, self.C, self.Pi)
        mu_0, self.beta = pois_regr.fit()

        for k in range(self.n_clusters):
            cluster_mask = (self.c_assignments == k)
            new_acluster = AllelicCluster(self.r[cluster_mask], self.C[cluster_mask, :], mu_0[k], self.beta)
            self.clusters[k] = new_acluster

            # set initial ll
            self.ll_clusters[k] = new_acluster.get_ll()

    def save_sample(self):
        mu_i_save = []
        lepsi_i_save = []
        F_save = []

        for clust in self.clusters:
            mu_i_save.append(clust.mu_i_arr.copy())
            lepsi_i_save.append(clust.lepsi_i_arr.copy())
            F_save.append(clust.F.copy())
        self.mu_i_samples.append(mu_i_save)
        self.lepsi_i_samples.append(lepsi_i_save)
        self.F_samples.append(F_save)

    def pick_cluster(self, n_it):
        # randomly select clusters with equal probabilites for the first 1k iterations then select based on size
        # TODO: tweak this to dynamically decide when to switch selection probabilites
        if n_it < 1000:
            return np.random.choice(range(self.n_clusters))
        return np.random.choice(range(self.n_clusters), p=self.cluster_probs)

    def run(self,
            debug=False,
            stop_after_burnin=False):
        print("starting MCMC coverage segmentation...")

        past_it = 0

        # for n_it in tqdm.tqdm(range(n_iter)):
        n_it = 0
        while self.n_iter > len(self.F_samples):

            # check if we have burnt in
            if n_it > 2000 and not self.burnt_in and not n_it % 100:
                if np.diff(np.array(self.ll_iter[-500:])).mean() < 0:
                    print('burnt in!')
                    self.burnt_in = True
                    past_it = n_it
                    if stop_after_burnin:
                        print('Burn-in complete: ll: {} n_it: {}'.format(self.ll_clusters.sum(), n_it))
                        return

            # update cluster probs to reflect number of segments in each cluster
            if n_it > 1000 and not n_it % 100:
                self.cluster_probs = self.num_segments / self.num_segments.sum()

            # save dynamicaly thinned chain samples
            if not n_it % 50 and self.burnt_in and \
                    n_it - past_it > self.num_segments.sum():
                self.save_sample()
                past_it = n_it

            # decide whether to split or join on this iteration
            cluster_pick = self.pick_cluster(n_it)
            # cluster_pick = 1
            if np.random.rand() > 0.5:
                # split
                res = self.clusters[cluster_pick].split(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_clusters[cluster_pick] = self.clusters[cluster_pick].get_ll()
                    self.num_segments[cluster_pick] += 1
            else:
                # join
                res = self.clusters[cluster_pick].join(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_clusters[cluster_pick] = self.clusters[cluster_pick].get_ll()
                    self.num_segments[cluster_pick] -= 1
            n_it += 1
            self.ll_iter.append(self.ll_clusters.sum())

    def update_beta(self, total_exposure):
        endog = np.exp(np.log(self.r.flatten()) - total_exposure)
        exog = self.C
        start_params = np.r_[self.beta.flatten(), 1]
        sNB = statsNB(endog, exog, start_params=start_params)
        res = sNB.fit(start_params=start_params, disp=0)
        return res.params[:-1]

    def prepare_results(self):
        # convert saved Cov_MCMC cluster results into global result arrays
        seg_samples = np.zeros((self.Pi.shape[0], len(self.F_samples)))
        mu_i_samples = np.zeros((self.Pi.shape[0], len(self.F_samples)))
        mu_global = np.zeros(self.Pi.shape[0])

        pi_argmax = self.Pi.argmax(1)
        for it in range(len(self.F_samples)):
            global_seg_counter = 0
            for c in range(len(self.F_samples[it])):
                og_positions = np.where(pi_argmax == c)[0]
                if it == 0:
                    mu_global[og_positions] = self.clusters[c].mu
                seg_intervals = np.array(self.F_samples[it][c]).reshape(-1, 2)
                mu_i_samples[og_positions, it] = self.mu_i_samples[it][c]
                for st, en in seg_intervals:
                    seg_ind = og_positions[st:en]
                    seg_samples[seg_ind, it] = global_seg_counter
                    global_seg_counter += 1

        overall_exposure = mu_global + mu_i_samples[:, -1]
        global_beta = self.update_beta(overall_exposure)
        return seg_samples, global_beta, mu_i_samples

        # TODO: add visualization script

    """
    This class is for running segmentation on a single allelic clusters, which allows for scattering cluster segmentations
    across nodes. Under the current structuring of the package the global beta values still need to be calculated in each
    node, hence the need for passing the full r, C, and Pi matrices to each call. In the future this should be pushed to
    the initialization sequence in the run_coverage_mcmc class.
    """
class NB_MCMC_SingleCluster:

    def __init__(self, n_iter, r, C, mu, beta, cluster_num):
        self.n_iter = n_iter
        self.r = r
        self.C = C
        self.beta = beta
        self.mu = mu
        self.cluster_num = cluster_num
        # for now assume that the Pi vector assigns each bin to exactly one cluster

        self.burnt_in = False

        self.cluster = None
        self.num_segments = 1

        self.mu_i_samples = []
        self.lepsi_i_samples = []
        self.F_samples = []

        self.ll_cluster = 0
        self.ll_iter = []
        self._init_cluster()

    def _init_cluster(self):
        self.cluster = AllelicCluster(self.r, self.C, self.mu, self.beta)

        # set initial ll
        self.ll_cluster = self.cluster.get_ll()

    def save_sample(self):
        self.mu_i_samples.append(self.cluster.mu_i_arr.copy())
        self.lepsi_i_samples.append(self.cluster.lepsi_i_arr.copy())
        self.F_samples.append(self.cluster.F.copy())

    def run(self,
            debug=False,
            stop_after_burnin=False):
        print("starting MCMC coverage segmentation for cluster {}...".format(self.cluster_num))

        past_it = 0
        n_it = 0
        min_it = min(200, max(50, self.r.shape[0]))
        while self.n_iter > len(self.F_samples):

            # check if we have burnt in
            if n_it >= min_it and not self.burnt_in and not n_it % 50:
                if np.diff(np.array(self.ll_iter[-min_it:])).mean() <= 0:
                    print('burnt in!')
                    self.burnt_in = True
                    past_it = n_it
                    if stop_after_burnin:
                        print('Burn-in complete: ll: {} n_it: {}'.format(self.ll_cluster, n_it))
                        return
            #status update
            if not n_it % 50:
                print('n_it: {}'.format(n_it))
            # save dynamicaly thinned chain samples
            if not n_it % 25 and self.burnt_in and n_it - past_it > self.num_segments:
                self.save_sample()
                past_it = n_it

            if np.random.rand() > 0.5:
                # split
                res = self.cluster.split(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_cluster = self.cluster.get_ll()
                    self.num_segments += 1
            else:
                # join
                res = self.cluster.join(debug)
                # if we made a change, update ll of cluster
                if res > 0:
                    self.ll_cluster = self.cluster.get_ll()
                    self.num_segments -= 1
            n_it += 1
            self.ll_iter.append(self.ll_cluster)

    # return just the local beta in this case since we cant do global calculation until we see all of the clusters
    def prepare_results(self):
        num_draws = len(self.F_samples)
        num_bins = len(self.cluster.r)

        segmentation_samples = np.zeros((num_bins, num_draws))
        mu_i_full = np.zeros((num_bins, num_draws))

        for d in range(num_draws):
            seg_counter = 0
            seg_intervals = np.array(self.F_samples[d]).reshape(-1, 2)
            mu_i_full[:, d] = self.mu_i_samples[d]

            for st, en in seg_intervals:
                segmentation_samples[st:en, d] = seg_counter
                seg_counter += 1

        return segmentation_samples, self.beta, mu_i_full

    def visualize_cluster_samples(self, savepath):
        residuals = np.exp(np.log(self.cluster.r.flatten()) - (self.cluster.mu.flatten()) - (self.cluster.C@self.cluster.beta).flatten())
        num_draws = len(self.F_samples)
        num_rows = int(np.ceil(num_draws / 4))
        fig, axs = plt.subplots(num_rows, 4, figsize = (25,num_rows*3), sharey=True)
        ax_lst = axs.flatten()
        for d in range(num_draws):
            ax_lst[d].scatter(np.r_[:len(residuals)], residuals)
            hist = np.array(self.F_samples[d]).reshape(-1,2)
            for j, r in enumerate(hist):
                ax_lst[d].add_patch(mpl.patches.Rectangle((r[0],0), r[1]-r[0], 2.3, fill=True, alpha=0.3, color = colors[j % 10]))
        plt.savefig(savepath)
