import numpy as np
import scipy.special as ss
import sortedcontainers as sc
import os
from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
np.seterr(divide='ignore')

colors = mpl.cm.get_cmap("tab10").colors

def LSE(x):
    lmax = np.max(x)
    return lmax + np.log(np.exp(x - lmax).sum())


class PoissonRegression:
    def __init__(self, r, C, Pi):
        self.r = r
        self.C = C
        self.Pi = Pi

        self.mu = np.log(r.mean() * np.ones([Pi.shape[1], 1]))
        self.beta = np.ones([C.shape[1], 1])
        self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu)

    # mu gradient
    def gradmu(self):
        return self.Pi.T @ (self.r - self.e_s)

    # mu Hessian
    def hessmu(self):
        return -self.Pi.T @ np.diag(self.e_s.ravel()) @ self.Pi

    # beta gradient
    def gradbeta(self):
        return self.C.T @ (self.r - self.e_s)

    # beta Hessian
    def hessbeta(self):
        return -self.C.T @ np.diag(self.e_s.ravel()) @ self.C

    # mu,beta Hessian
    def hessmubeta(self):
        return -self.C.T @ np.diag(self.e_s.ravel()) @ self.Pi

    def NR_poisson(self):
        for i in range(100):
            self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu)
            gmu = self.gradmu()
            gbeta = self.gradbeta()
            grad = np.r_[gmu, gbeta]

            hmu = self.hessmu()
            hbeta = self.hessbeta()
            hmubeta = self.hessmubeta()
            H = np.r_[np.c_[hmu, hmubeta.T], np.c_[hmubeta, hbeta]]

            delta = np.linalg.inv(H) @ grad
            self.mu -= delta[0:len(self.mu)]
            self.beta -= delta[len(self.mu):]

            if np.linalg.norm(grad) < 1e-5:
                break

    def fit(self):
        self.NR_poisson()
        return self.mu, self.beta


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
        self.lepsi = np.log((((np.log(self.r) - self.mu) ** 2 / self.mu - 1)).sum() / (len(self.r)))

        for _ in range(100):
            self.exp = np.exp(self.mu + self.C @ self.beta).flatten()
            self.epsi = np.exp(self.lepsi)
            gmu = self.gradmu()
            gepsi = self.gradepsi()
            hmu = self.hessmu()
            hepsi = self.hessepsi() * self.epsi ** 2 + self.epsi * gepsi
            hmuepsi = self.hessmuepsi()

            grad = np.r_[gmu, gepsi * self.epsi + 1]
            H = np.r_[np.c_[hmu, hmuepsi], np.c_[hmuepsi, hepsi]]
            delta = np.linalg.inv(H) @ grad

            self.mu -= delta[0]
            self.lepsi -= delta[1]
            if np.linalg.norm(grad) < 1e-5:
                break

    # helper init derivative functions

    def gradmu(self):
        return ((self.epsi * (self.r - self.exp)) / (self.epsi + self.exp)).sum(0)

    def gradepsi(self):
        return (ss.digamma(self.r + self.epsi) - ss.digamma(self.epsi) + (
                self.exp - self.r + self.exp *
                np.log(self.epsi /
                       (self.exp + self.epsi)) + self.epsi * np.log(self.epsi / (self.exp + self.epsi))) /
                (self.exp + self.epsi)).sum(0)

    def hessmu(self):
        return (-(self.exp * self.epsi * (self.r + self.epsi)) / ((self.exp + self.epsi) ** 2)).sum(0)

    def hessepsi(self):
        return (ss.polygamma(1, self.r + self.epsi) -
                ss.polygamma(1, self.epsi) + (self.exp ** 2 + self.r * self.epsi) /
                (self.epsi * (self.exp + self.epsi) ** 2)).sum(0)

    def hessmuepsi(self):
        return ((self.exp * (self.r - self.exp)) / (self.exp + self.epsi) ** 2).sum(0)

    # TODO something wrong here
    # optimizes mu_i and epsi_i values for either a split segment or a join segment
    def NR_segment(self, ind, ret_hess=False):
        # start by setting mu_i to the average of the residuals
        mui_init = np.log(np.exp(np.log(self.r[ind[0]:ind[1]]) - (self.mu) -
                                 (self.C[ind[0]:ind[1], :] @ self.beta).flatten()).mean())
        # lepsi_init = np.log((((np.log(self.r[ind[0]:ind[1]]) - self.mu)**2 / self.mu - 1)).sum() / (ind[1]-ind[0]))
        self.mu_i = mui_init
        self.lepsi_i = self.lepsi_i_arr[ind[0]]
        cur_iter = 0
        max_iter = 75
        self.mu_i, self.lepsi_i = self.stats_optimizer(ind)
        stats_ll = self.ll_nbinom(self.r[ind[0]:ind[1]], self.mu, self.C[ind[0]:ind[1]], self.beta,
                                  np.ones(self.r[ind[0]:ind[1]].shape) * self.mu_i, self.lepsi_i)
        stats_mui, stats_lepsi = self.mu_i, self.lepsi_i
        while (cur_iter < max_iter):
            self.epsi_i = np.exp(self.lepsi_i)
            self.exp = np.exp(self.mu + (self.C[ind[0]:ind[1]] @ self.beta).flatten() + self.mu_i)

            gmu_i = self.gradmu_i(ind)
            hmu_i = self.hessmu_i(ind)
            gepsi_i = self.gradepsi_i(ind)
            hepsi_i = self.hessepsi_i(ind) * self.epsi_i ** 2 + gmu_i * self.epsi_i
            hmuepsi_i = self.hessepsi_i(ind)

            grad = np.r_[gmu_i, gepsi_i * self.epsi_i]
            H = np.r_[np.c_[hmu_i, hmuepsi_i], np.c_[hmuepsi_i, hepsi_i]]

            try:
                inv_H = np.linalg.inv(H)
            except:
                print('reached singular matrix. reseeting with grid search')
                self.lepsi_i = self.ll_gridsearch(ind)
                continue
            delta = inv_H @ grad

            self.mu_i -= delta[0]
            self.lepsi_i -= delta[1]

            if np.isnan(self.mu_i):
                # if we hit a nan try a new initialization
                print('hit a nan in optimizer, reseting')
                self.mu_i = mui_init + np.random.rand() - 0.5
                self.lepsi_i = self.ll_gridsearch(ind)
                continue

            if np.linalg.norm(grad) < 1e-5:
                # print('opt reached')
                # print(np.linalg.det)
                break

            if cur_iter == 25:
                # if its taking this long we should reset with a grid search
                print('failing to converge. Trying grid search')
                self.mu_i = mui_init
                self.lepsi_i = self.ll_gridsearch(ind)

            if cur_iter == 50:
                print('trying stats opt')
                self.mu_i, self.lepsi_i = self.stats_optimizer(ind)

            if cur_iter == 75:
                print('failed to optimize: ', ind)

            cur_iter += 1
        # print(ind, self.mu_i, self.lepsi_i)
        # need to theshold due to overflow
        self.lepsi_i = min(self.lepsi_i, 40)

        NR_ll = self.ll_nbinom(self.r[ind[0]:ind[1]], self.mu, self.C[ind[0]:ind[1]], self.beta,
                               np.ones(self.r[ind[0]:ind[1]].shape) * self.mu_i, self.lepsi_i)
        self.opt_views.append((ind, stats_ll, stats_mui, stats_lepsi, NR_ll, self.mu_i, self.lepsi_i))
        if ret_hess:
            return self.mu_i, self.lepsi_i, H
        else:
            return self.mu_i, self.lepsi_i

    # helper segment derivative function ** make sure to set epsi_i, mu_i and exp before use
    def gradmu_i(self, ind):
        return ((self.epsi_i * (self.r[ind[0]:ind[1]] - self.exp)) / (self.epsi_i + self.exp)).sum(0)

    def gradepsi_i(self, ind):
        return (ss.digamma(self.r[ind[0]: ind[1]] + self.epsi_i) - ss.digamma(self.epsi_i) + (
                self.exp - self.r[ind[0]: ind[1]] + self.exp * np.log(
            self.epsi_i / (self.exp + self.epsi_i)) + self.epsi_i * np.log(
            self.epsi_i / (self.exp + self.epsi_i))) / (self.exp + self.epsi_i)).sum(0)

    def hessmu_i(self, ind):
        return (-(self.exp * self.epsi_i * (self.r[ind[0]: ind[1]] + self.epsi_i)) /
                ((self.exp + self.epsi_i) ** 2)).sum(0)

    def hessepsi_i(self, ind):
        return (ss.polygamma(1, self.r[ind[0]: ind[1]] + self.epsi_i) -
                ss.polygamma(1, self.epsi_i) + (self.exp ** 2 + self.r[ind[0]: ind[1]] * self.epsi_i) /
                (self.epsi_i * (self.exp + self.epsi_i) ** 2)).sum(0)

    def hessmuepsi_i(self, ind):
        return ((self.exp * (self.r[ind[0]: ind[1]] - self.exp)) / (self.exp + self.epsi_i) ** 2).sum(0)

    def get_seg_ind(self, seg):
        return seg, seg + self.segment_lens[seg]

    def get_join_seg_ind(self, seg_l, seg_r):
        return seg_l, seg_r + self.segment_lens[seg_r]

    def get_ll(self):
        return self.ll_cluster(self.mu_i_arr, self.lepsi_i_arr, True)

    # off the shelf optimizer for testing
    def stats_optimizer(self, ind, ret_hess=False):
        endog = np.exp(np.log(self.r[ind[0]:ind[1]].flatten()) - (self.C[ind[0]:ind[1]] @ self.beta).flatten())
        exog = np.ones(self.r[ind[0]:ind[1]].shape[0])
        exposure = np.ones(self.r[ind[0]:ind[1]].shape[0]) * np.exp(self.mu)
        sNB = statsNB(endog, exog, exposure=exposure)
        res = sNB.fit(disp=0)

        if ret_hess:
            return res.params[0], -np.log(res.params[1]), sNB.hessian(res.params)
        else:
            return res.params[0], -np.log(res.params[1])

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

    @staticmethod
    def ll_nbinom(r, mu, C, beta, mu_i, lepsi):
        r = r.flatten()
        epsi = np.exp(lepsi)
        bc = (C @ beta).flatten()
        exp = np.exp(mu + bc + mu_i).flatten()
        return (ss.gammaln(r + epsi) - ss.gammaln(r + 1) - ss.gammaln(epsi) +
                (r * (mu + bc + mu_i - np.log(epsi + exp))) +
                (epsi * np.log(epsi / (epsi + exp)))).sum()

    # TODO make this into a binary search
    def ll_gridsearch(self, ind):
        eps = np.r_[-5:15:0.1]
        res = np.zeros(eps.shape)
        r_ind = self.r[ind[0]:ind[1]]
        C_ind = self.C[ind[0]:ind[1]]
        for i, ep in enumerate(eps):
            res[i] = self.ll_nbinom(r_ind, self.mu, C_ind, self.beta, self.mu_i, ep)
        return eps[np.argmax(res)]

    # finds the indices of the top ~3% (2 stds above mean) of the difference kernel values in addition to the first and last window bins
    def _find_difs(self, ind, window=10):
        residuals = np.exp(np.log(self.r[ind[0]:ind[1]].flatten()) - (self.mu.flatten()) - (
                    self.C[ind[0]:ind[1]] @ self.beta).flatten())
        difs = []
        for i in np.r_[window:len(residuals) - window]:
            difs.append(np.abs(residuals[i - window:i].mean() - residuals[i:i + window].mean()))
        difs = np.r_[difs]
        cutoff = difs.mean() + 2 * np.sqrt(difs.var())

        full_idxs = np.r_[ind[0] + 2:ind[1] - 1]
        difs_full = np.r_[np.zeros(window - 2), difs, np.zeros(window - 1)]
        difs_idxs = full_idxs[difs_full > cutoff]

        # add on first and last windows
        difs_idxs = np.r_[np.r_[ind[0] + 2: ind[0] + window], difs_idxs, np.r_[ind[1] - window:ind[1] - 1]]
        return difs_idxs

    def _calculate_splits(self, ind, split_indices):
        lls = []
        mus = []
        lepsis = []
        Hs = []

        for ix in split_indices:
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

    def calc_pk(self, ind, break_pick=None, debug=False):

        ind_len = ind[1] - ind[0]
        # swtiching to disallow singletons from split
        if ind_len < 4 and not debug:
            raise Exception('segment was too small to split: length: ', ind[1] - ind[0])

        if ind_len:
            split_indices = np.r_[ind[0] + 2: ind[1] - 1]
            lls, mus, lepsis, Hs = self._calculate_splits(ind, split_indices)
        else:
            split_indices = self._find_difs(ind)
            if break_pick:
                split_indices.append(break_pick)
            # get lls from the difference kernel guesses
            lls, mus, lepsis, Hs = self._calculate_splits(ind, split_indices)

            # find the most likely index and those indices within a significant range
            # (rn 7 -> at least 1/1000th as likely)
            top_lik = max(lls)
            within_range = np.r_[lls] > (top_lik - 7)
            significant = np.r_[split_indices][within_range]

            # increase sampling of significantly likely regions
            extra_samples = sc.SortedSet({})
            for s in significant:
                for i in np.r_[max(2, s - 5):min(len(ind_len) - 1, s + 5)]:
                    extra_samples.add(i)
            extra_samples = list(extra_samples - set(significant))
            lls_add, mus_add, lepsis_add, Hs_add = self._calculate_splits(ind, extra_samples)
            lls.extend(lls_add)
            mus.extend(mus_add)
            lepsis.extend(lepsis_add)
            Hs.extend(Hs_add)
            split_indices.extend(extra_samples)

        lls = np.array(lls)
        log_k_probs = (lls - LSE(lls)).flatten()
        if break_pick:
            #i = break_pick - ind[0] - 2
            i = split_indices.index(break_pick)
            # return the probability of proposing a split here along with the optimal parameter values
            return lls[i], np.exp(log_k_probs[i]), mus[i], lepsis[i], Hs[i]

        else:
            # pick a breakpoint based on relative likelihood
            b_idx = np.random.choice(np.r_[:len(split_indices)], p=np.exp(log_k_probs))
            break_pick = split_indices[b_idx]
            if debug:
                return lls, break_pick, lls[b_idx], np.exp(log_k_probs[b_idx]), mus[b_idx], lepsis[b_idx], Hs[b_idx]

            return break_pick, lls[b_idx], np.exp(log_k_probs[b_idx]), mus[b_idx], lepsis[b_idx], Hs[b_idx]

    def _get_log_ML_approx_join(self, Hess):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-Hess))) / 2

    def _get_log_ML_split(self, H1, H2):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-H1) * np.linalg.det(-H2))) / 2

    def _log_ML_join(self, ind, ret_opt_params=False):
        # mu_share, lepsi_share, H_share = self.NR_segment(ind, True)
        mu_share, lepsi_share, H_share = self.stats_optimizer(ind, True)
        tmp_mui = self.mu_i_arr
        tmp_mui[ind[0]:ind[1]] = mu_share
        tmp_lepsi = self.lepsi_i_arr
        tmp_lepsi[ind[0]:ind[1]] = lepsi_share
        ll_join = self.ll_cluster(tmp_mui, tmp_lepsi)
        if ret_opt_params:
            return mu_share, lepsi_share, self._get_log_ML_join(H_share) + ll_join
        return mu_share, lepsi_share, self._get_log_ML_approx_join(H_share) + ll_join

    def _log_q_split(self, pk):
        return np.log(pk / len(self.segments))

    def _log_q_join(self):
        return - np.log(len(self.segments))

    # TODO: check if marginal liklihoods are correct
    def split(self, debug):

        if debug:
            ll_trace = np.zeros(self.segment_ID.shape[0])
        # pick a random segment
        seg = np.random.choice(self.segments)

        # print('attempting split on segment: ', seg)
        # if segment is a singleton then skip
        ## if segment is less than 4 bins then skip
        if self.segment_lens[seg] < 4:
            return -1

        ind = self.get_seg_ind(seg)
        # print('ind: ', ind)
        if debug:
            lls, break_pick, ll_split, pk, mus, lepsis, Hs = self.calc_pk(ind, debug=debug)
            ll_trace[ind[0] + 2: ind[1] - 1] = lls - LSE(lls)

            for s in self.segments.difference(set([seg])):
                s_ind = self.get_seg_ind(s)
                if (s_ind[1] - s_ind[0] < 4):
                    continue
                print('s_ind:', seg, s_ind)
                lls, _, _, _, _, _, _ = self.calc_pk(s_ind, debug=debug)
                ll_trace[s_ind[0] + 2: s_ind[1] - 1] = lls - LSE(lls)

        else:
            break_pick, ll_split, pk, mus, lepsis, Hs = self.calc_pk(ind, debug=debug)
        split_log_ML = ll_split + self._get_log_ML_split(Hs[0], Hs[1])
        _, _, join_log_ML = self._log_ML_join(ind)

        # print('split log ml ratio: ', split_log_ML - join_log_ML)
        # print('split log mh ratio: ', split_log_ML - join_log_ML + self._log_q_join() - self._log_q_split(pk))
        if np.log(np.random.rand()) < np.minimum(0, split_log_ML - join_log_ML +
                                                    self._log_q_join() - self._log_q_split(pk)):
            # print('split!')
            if debug:
                self.ll_traces.append(ll_trace)
            # split the segments
            # update segment assignments for new segment
            self.segment_ID[break_pick:ind[1]] = break_pick
            self.segments.add(break_pick)
            self.segment_lens[break_pick] = ind[1] - break_pick

            # update old seglen
            self.segment_lens[ind[0]] = break_pick - ind[0]

            # update mu_i and lepsi_i values
            self.mu_i_arr[ind[0]:break_pick] = mus[0]
            self.mu_i_arr[break_pick:ind[1]] = mus[1]
            self.lepsi_i_arr[ind[0]:break_pick] = lepsis[0]
            self.lepsi_i_arr[break_pick:ind[1]] = lepsis[1]

            self.F.update([break_pick, break_pick])
            # print(self.F)
            self.phase_history.append(self.F.copy())
            return break_pick

        return 0

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
        # print('attempting join on segs:', seg_l, seg_r)
        # print('join ind: ', ind)
        ll_split, pk, mus, lepsis, Hs = self.calc_pk(ind, break_pick=seg_r)
        split_log_ML = ll_split + self._get_log_ML_split(Hs[0], Hs[1])
        mu_share, lepsi_share, join_log_ML = self._log_ML_join(ind)

        if np.log(np.random.rand()) < np.minimum(0, join_log_ML - split_log_ML +
                                                    self._log_q_split(pk) - self._log_q_join()):
            # join the segments
            # update segment assignments for new segment
            self.segment_ID[ind[0]:ind[1]] = seg_l
            self.segments.discard(seg_r)
            del self.segment_lens[seg_r]
            self.segment_lens[seg_l] = ind[1] - ind[0]
            # print('joining!')
            # update mu_i and lepsi_i values
            self.mu_i_arr[ind[0]:ind[1]] = mu_share
            self.lepsi_i_arr[ind[0]:ind[1]] = lepsi_share

            self.F.discard(seg_r)
            self.F.discard(seg_r)
            # self.phase_history.append(self.F.copy())
            #   print(self.F)
            return seg_r

        return 0


class NB_MCMC_AllClusters:

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

class NB_MCMC_SingleCluster:

    def __init__(self, n_iter, r, C, Pi, cluster_num):
        self.n_iter = n_iter
        self.r = r
        self.C = C
        self.Pi = Pi
        self.beta = None
        self.mu = None
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
        # first we find good starting values for mu and beta for each cluster by fitting a poisson model
        pois_regr = PoissonRegression(self.r, self.C, self.Pi)
        all_mu, self.beta = pois_regr.fit()
        self.mu = all_mu[self.cluster_num]

        c_assignments = np.argmax(self.Pi, axis=1)
        cluster_mask = (c_assignments == self.cluster_num)
        self.cluster = AllelicCluster(self.r[cluster_mask], self.C[cluster_mask], self.mu, self.beta)

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
        while self.n_iter > len(self.F_samples):

            # check if we have burnt in
            if n_it > self.cluster.r.shape[0] / 2  and not self.burnt_in and not n_it % 50:
                if np.diff(np.array(self.ll_iter[-50:])).mean() <= 0:
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

    def update_beta(self, total_exposure):
        endog = np.exp(np.log(self.r.flatten()) - total_exposure)
        exog = self.C
        start_params = np.r_[self.beta.flatten(), 1]
        sNB = statsNB(endog, exog, start_params=start_params)
        res = sNB.fit(start_params=start_params, disp=0)
        return res.params[:-1]

    # return just the preliminary beta in this case since we cant do global calculation until we see all of the clusters
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

        # TODO: add visualization script

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
