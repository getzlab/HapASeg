import numpy as np
import scipy.special as ss
import sortedcontainers as sc
import tqdm


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
        self.r = r
        self.C = C
        self.mu = mu_0
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

        self._init_params()

    # fits NB model to this cluster using initial params as starting point
    def _init_params(self):
        self.NR_init()
        self.lepsi_i_arr = np.ones(self.r.shape) * self.lepsi
        self.epsi_i_arr = np.exp(self.lepsi_i_arr)

    # fit initial mu_k and epsilon_k for the cluster
    def NR_init(self):
        for _ in range(100):
            self.exp = np.exp(self.mu + self.C @ self.beta)
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
    #TODO something wrong here
    # optimizes mu_i and epsi_i values for either a split segment or a join segment
    def NR_segment(self, ind, ret_hess=False):
        self.mu_i = self.mu_i_arr[ind[0]]
        self.lepsi_i = self.lepsi_i_arr[ind[0]]
        for _ in range(100):
            self.epsi_i = np.exp(self.lepsi_i)
            self.exp = np.exp(self.mu + self.C[ind[0]:ind[1]] @ self.beta + self.mu_i)

            gmu_i = self.gradmu_i(ind)
            hmu_i = self.hessmu_i(ind)
            gepsi_i = self.gradepsi_i(ind)
            hepsi_i = self.hessepsi_i(ind) * self.epsi_i ** 2 + gmu_i * self.epsi_i
            hmuepsi_i = self.hessepsi_i(ind)

            grad = np.r_[gmu_i, gepsi_i * self.epsi_i]
            H = np.r_[np.c_[hmu_i, hmuepsi_i], np.c_[hmuepsi_i, hepsi_i]]
            delta = np.linalg.inv(H) @ grad

            self.mu_i -= delta[0]
            self.lepsi_i -= delta[1]
            if np.linalg.norm(grad) < 1e-5:
                break
        
        print(ind, self.mu_i, self.lepsi_i)
        if ret_hess:
            return self.mu_i, self.lepsi_i, H
        else:
            return self.mu_i, self.lepsi_i

    # helper segment derivative function ** make sure to set epsi_i, mu_i and exp before use
    def gradmu_i(self, ind):
        return ((self.epsi_i * (self.r[ind[0]: ind[1]] - self.exp)) / (self.epsi_i + self.exp)).sum(0)

    def gradepsi_i(self, ind):
        return (ss.digamma(self.r[ind[0]: ind[1]] + self.epsi) - ss.digamma(self.epsi_i) + (
                self.exp - self.r[ind[0]: ind[1]] + self.exp *
                np.log(self.epsi_i /
                       (self.exp + self.epsi_i)) + self.epsi_i * np.log(self.epsi_i / (self.exp + self.epsi_i))) /
                (self.exp + self.epsi_i)).sum(0)

    def hessmu_i(self, ind):
        return (-(self.exp * self.epsi_i * (self.r[ind[0]: ind[1]] + self.epsi_i)) / ((self.exp + self.epsi_i) ** 2)).sum(0)

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
        return self.ll_cluster(self.mu_i_arr, self.lepsi_i_arr)

    def ll_cluster(self, mu_i_arr, lepsi_arr):
        exp = np.exp(self.mu + self.C @ self.beta + mu_i_arr)
        return (ss.gammaln(self.r + np.exp(lepsi_arr)) - ss.gammaln(self.r + 1) - ss.gammaln(np.exp(lepsi_arr)) +
        self.r * (self.mu + self.C @ self.beta + mu_i_arr - np.log(np.exp(lepsi_arr) + exp)) +
        np.exp(lepsi_arr) * (lepsi_arr - np.log(np.exp(lepsi_arr) + exp))).sum(0)

    def calc_pk(self, ind, break_pick=None):
        lls = []
        mus = []
        lepsis = []
        Hs = []
        ks = np.r_[ind[0] + 1: ind[1]]
        for k in ks:
            mu_l, lepsi_l, H_l= self.NR_segment((ind[0], k), True)
            mu_r, lepsi_r, H_r = self.NR_segment((k, ind[1]), True)
            mus.append((mu_l, mu_r))
            lepsis.append((lepsi_l, lepsi_r))
            Hs.append((H_l, H_r))

            tmp_mui = self.mu_i_arr
            tmp_mui[ind[0]:k] = mu_l
            tmp_mui[k: ind[1]] = mu_r
            tmp_lepsi = self.lepsi_i_arr
            tmp_lepsi[ind[0]:k] = lepsi_l
            tmp_lepsi[k: ind[1]] = lepsi_r

            lls.append(self.ll_cluster(tmp_mui, tmp_lepsi))
        lls = np.array(lls)
        k_probs = (lls / lls.sum()).flatten()
        print('all_mus: ', mus)
        if break_pick:
            i = break_pick - ind[0] - 1
            # return the probability of proposing a split here along with the optimal parameter values
            return lls[i], k_probs[i], mus[i], lepsis[i], Hs[i]

        else:
            # pick a breakpoint based on relative likelihood
            break_pick = np.random.choice(np.r_[ind[0] + 1: ind[1]], p=k_probs)
            i = break_pick - ind[0] - 1
            return break_pick, lls[i], k_probs[i], mus[i], lepsis[i], Hs[i]

    def _get_log_ML_approx_join(self, Hess):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-Hess))) / 2

    def _get_log_ML_split(self, H1, H2):
        return np.log(2 * np.pi) - (np.log(np.linalg.det(-H1) * np.linalg.det(-H2))) / 2

    def _log_ML_join(self, ind, ret_opt_params=False):
        mu_share, lepsi_share, H_share = self.NR_segment(ind, True)
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

#TODO: check if marginal liklihoods are correct
    def split(self):

        # pick a random segment
        seg = np.random.choice(self.segments)

        #if segment is a singleton then skip
        if self.segment_lens[seg] == 1:
            return -1

        ind = self.get_seg_ind(seg)
        break_pick, ll_split, pk, mus, lepsis, Hs = self.calc_pk(ind)

        split_log_ML = ll_split + self._get_log_ML_split(Hs[0], Hs[1])
        _, _, join_log_ML = self._log_ML_join(ind)
        
        print('split log ml ratio: ', split_log_ML - join_log_ML)
        if np.log(np.random.rand()) < np.minimum(0, split_log_ML - join_log_ML +
                                                    self._log_q_join() - self._log_q_split(pk)):
            print('split!')
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
            
            print('new muis: ', mus)
            print('new epsis: ', lepsis)
            self.F.update([break_pick, break_pick])
            print(self.F)
            self.phase_history.append(self.F)
            return break_pick

        return 0

    def join(self):
        num_segs = len(self.segments)
        # if theres only one segment, skip
        if num_segs == 1:
            return -1

        # otherwise pick a left segment to join
        seg_l_ind = np.random.choice(num_segs - 1)
        seg_l = self.segments[seg_l_ind]
        seg_r = self.segments[seg_l_ind + 1]
        ind = self.get_join_seg_ind(seg_l, seg_r)
        
        ll_split, pk, mus, lepsis, Hs = self.calc_pk(ind, break_pick=seg_r)
        split_log_ML = ll_split + self._get_log_ML_split(Hs[0], Hs[1])
        mu_share, lepsi_share, join_log_ML = self._log_ML_join(ind)
        
        if np.log(np.random.rand()) < np.minimum(0, join_log_ML  - split_log_ML +
                                                    self._log_q_split(pk) - self._log_q_join()):
            # join the segments
            # update segment assignments for new segment
            self.segment_ID[ind[0]:ind[1]] = seg_l
            self.segments.discard(seg_r)
            self.segment_lens[seg_l] = ind[1] - ind[0]
            #print('joining!')
            # update mu_i and lepsi_i values
            self.mu_i_arr[ind[0]:ind[1]] = mu_share
            self.lepsi_i_arr[ind[0]:ind[1]] = lepsi_share

            self.F.discard(seg_r)
            self.F.discard(seg_r)
            self.phase_history.append(self.F)
            print(self.F)
            return seg_r

        return 0

class NB_MCMC:

    def __init__(self, r, C, Pi):
        self.r = r
        self.C = C
        self.Pi = Pi
        self.beta = None

        # for now assume that the Pi vector assigns each bin to exactly one cluster
        self.c_assignments = np.argmax(self.Pi, axis=1)
        self.n_clusters = Pi.shape[1]
        self.clusters = [None] * self.n_clusters
        self.cluster_sizes = np.array([np.sum(self.c_assignments == i) for i in range(self.n_clusters)])
        self.cluster_probs = self.cluster_sizes / self.cluster_sizes.sum()

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

    def pick_cluster(self):
        return np.random.choice(range(self.n_clusters), p=self.cluster_probs)

    def run(self, n_iter=1000):
        print("starting MCMC coverage segmentation...")
        for iter in tqdm.tqdm(range(n_iter)):
            # decide whether to split or join on this iteration
            #cluster_pick = self.pick_cluster()
            #print("using cluster 18 only for testing")
            cluster_pick = 18

            self.ll_iter.append(self.clusters[cluster_pick].get_ll())
            if np.random.rand() > 0.5:
                # split
                #print("split cluster {}".format(cluster_pick))
                self.clusters[cluster_pick].split()
            else:
                # join
                #print("join cluster {}".format(cluster_pick))
                self.clusters[cluster_pick].join()
