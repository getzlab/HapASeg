import numpy as np
import scipy.special as ss
import sortedcontainers as sc
import tqdm

class MySortedList(sc.SortedList):
    # since the sorted list represents intervals, for debugging purposes it's
    # a lot easier to output them in columnar format:
    def __repr__(self):
        assert not len(self) % 2
        return str(np.array(self).reshape(-1, 2))

    def intervals(self):
        return np.array(self).reshape(-1, 2)

class Cov_MCMC:

    def __init__(self, r, C, Pi, cov_df):

        self.r = r
        self.C = C
        self.Pi = Pi

        self.mu = np.log(r.mean() * np.ones([Pi.shape[1], 1]))
        self.beta = np.ones([C.shape[1], 1])
        self.mu_i = np.zeros(C.shape[0])

        self.mu_shared = 0
        self.mu_l = 0
        self.mu_r = 0

        # keep simple exponential value saved
        self.e_s = np.exp(self.C@self.beta_0 + self.Pi@self.mu_0 + self.mu_i[:, None])
        self.e_l = 0
        self.e_r = 0

        # init segment list with all intervals as their own segment
        self.F = MySortedList()
        self.F.update([i for i in range(len(self.cov_df))])
        self.F.update([i + 1 for i in range(len(self.cov_df))])
        self.segment_history = []

        # init breakpoint set and remove chromosome boundaries
        self.cov_breaks = sc.SortedSet(np.r_[:len(self.cov_df)])
        _ = [self.cov_breaks.discard(elt) for elt in self.cov_df.groupby('chr').size().cumsum().values - 1]

        self.cov_df['SegID'] = range(len(self.cov_df))
        self.segLenDict = sc.SortedDict(zip(range(len(self.cov_df)), np.ones(len(self.cov_df))))


    @staticmethod
    def LSE(x):
        lmax = np.max(x)
        return lmax + np.log(np.exp(x - lmax).sum())

    # mu gradient
    def gradmu(self):
        return self.Pi.T@(self.r - self.e_s)

    # mu Hessian
    def hessmu(self):
        return -self.Pi.T@np.diag(self.e_s.ravel())@self.Pi

    # beta gradient
    def gradbeta(self):
        return self.C.T@(self.r - self.e_s)

    # beta Hessian
    def hessbeta(self):
        return -self.C.T@np.diag(self.e_s.ravel())@self.C

    # mu,beta Hessian
    def hessmubeta(self):
        return -self.C.T@np.diag(self.e_s.ravel())@self.Pi

    # join mu_shared gradient
    def joingradmu_shared(self, ind_l, ind_r):

        grad_l = self.r[ind_l[0]:ind_l[1]] - self.e_l
        grad_r = self.r[ind_r[0]:ind_r[1]] - self.e_r
        return (grad_l.sum() + grad_r.sum())[None, None]


    def joinhessmu_shared(self):
        return (-self.e_l.sum() - self.e_r.sum())[None, None]

    def splitgradmu_l(self, ind_l):
        grad_l = self.r[ind_l[0]:ind_l[1]] - self.e_l
        return grad_l.sum()[None, None]

    def splitgradmu_r(self, ind_r):
        grad_r = self.r[ind_r[0]:ind_r[1]] - self.e_r
        return grad_r.sum()[None, None]

    def splithessmu_l(self):
        return (-self.e_l).sum()[None, None]

    def splithessmu_r(self):
        return (-self.e_r).sum()[None, None]

    def splithessmu_lr(self):
        return np.array(0)[None, None]

    # split newton-raphson
    def NR_split(self, ind_l, ind_r, ret_hess=False):
        mu_l = 0
        mu_r = 0

        for i in range(100):
            self.e_l = np.exp(self.C[ind_l[0]:ind_l[1]] @ self.beta + self.Pi[ind_l[0]:ind_l[1]] @ self.mu + mu_l)
            self.e_r = np.exp(self.C[ind_r[0]:ind_r[1]] @ self.beta + self.Pi[ind_r[0]:ind_r[1]] @ self.mu + mu_r)
            # print(mu, beta, mu_l, mu_r)
            gmu_l = self.splitgradmu_l(ind_l)
            gmu_r = self.splitgradmu_r(ind_r)
            grad = np.r_[gmu_l, gmu_r]

            hmu_l = self.splithessmu_l()
            hmu_r = self.splithessmu_r()

            H = np.r_[np.c_[hmu_l, 0], np.c_[0, hmu_r]]

            delta = np.linalg.inv(H) @ grad
            mu_l -= delta[0]
            mu_r -= delta[1]

            if np.linalg.norm(grad) < 1e-5:
                break
        if ret_hess:
            return mu_l, mu_r, H
        else:
            return mu_l, mu_r

    # join newton-raphson
    def NR_join(self, ind_l, ind_r, ret_hess=False):
        mu_share = 0

        for i in range(100):
            self.e_l = np.exp(self.C[ind_l[0]:ind_l[1]] @ self.beta + self.Pi[ind_l[0]:ind_l[1]] @ self.mu + mu_share)
            self.e_r = np.exp(self.C[ind_r[0]:ind_r[1]] @ self.beta + self.Pi[ind_r[0]:ind_r[1]] @ self.mu + mu_share)

            gmu_shared = self.joingradmu_shared(ind_l, ind_r)
            hmu_shared = self.joinhessmu_shared(ind_l, ind_r)

            delta = gmu_shared / hmu_shared
            mu_share -= delta

            if delta < 1e-5:
                break
        if ret_hess:
            return mu_share, hmu_shared
        else:
            return mu_share

    def NR_simp(self):

        for i in range(100):
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

    def split_log_liklihood(self, mu_l, mu_r, ind_l, ind_r):
        l_tmp = self.C[ind_l[0]:ind_l[1]] @ self.beta + self.Pi[ind_l[0]:ind_l[1]] @ self.mu + mu_l
        r_tmp = self.C[ind_r[0]:ind_r[1]] @ self.beta + self.Pi[ind_r[0]:ind_r[1]] @ self.mu + mu_r
        outer_l = self.C[:ind_l[0]] @ self.beta + self.Pi[:ind_l[0]] @ self.mu + self.mu_i[:ind_l[0], None]
        outer_r = self.C[ind_r[1]:] @ self.beta + self.Pi[ind_r[1]:] @ self.mu + self.mu_i[ind_r[1]:, None]
        e_l = np.exp(l_tmp)
        e_r = np.exp(r_tmp)
        e_outer_l = np.exp(outer_l)
        e_outer_r = np.exp(outer_r)

        r_sum = self.r[ind_l[0]:ind_l[1]].T @ l_tmp - e_l.sum() - ss.gammaln(self.r[ind_l[0]:ind_l[1]]).sum()
        l_sum = self.r[ind_r[0]:ind_r[1]].T @ r_tmp - e_r.sum() - ss.gammaln(self.r[ind_r[0]:ind_r[1]]).sum()
        outer_sum_l = self.r[:ind_l[0]].T @ outer_l - e_outer_l.sum() - ss.gammaln(self.r[:ind_l[0]]).sum()
        outer_sum_r = self.r[ind_r[1]:].T @ outer_r - e_outer_r.sum() - ss.gammaln(self.r[ind_r[1]:]).sum()
        return l_sum + r_sum + outer_sum_l + outer_sum_r


    def join_log_liklihood(self, mu_shared, ind_l, ind_r):
        joined = self.C[ind_l[0]:ind_r[1]] @ self.beta + self.Pi[ind_l[0]:ind_r[1]] @ self.mu + mu_shared
        outer_l = self.C[:ind_l[0]] @ self.beta + self.Pi[:ind_l[0]] @ self.mu + self.mu_i[:ind_l[0], None]
        outer_r = self.C[ind_r[1]:] @ self.beta + self.Pi[ind_r[1]:] @ self.mu + self.mu_i[ind_r[1]:, None]

        e_joined = np.exp(joined)
        e_outer_l = np.exp(outer_l)
        e_outer_r = np.exp(outer_r)

        joined_sum = self.r[ind_l[0]:ind_r[1]].T @ joined - e_joined.sum() - ss.gammaln(self.r[ind_l[0]:ind_r[1]]).sum()
        outer_sum_l = self.r[:ind_l[0]].T @ outer_l - e_outer_l.sum() - ss.gammaln(self.r[:ind_l[0]]).sum()
        outer_sum_r = self.r[ind_r[1]:].T @ outer_r - e_outer_r.sum() - ss.gammaln(self.r[ind_r[1]:]).sum()
        return joined_sum + outer_sum_l + outer_sum_r

    def get_ML_split(self, l_ind, r_ind):
        mu_l, mu_r, h_split = self.NR_split(l_ind, r_ind, True)
        logprod_s = np.log((2 * np.pi) ** (h_split.shape[0] / 2) * (1 / np.sqrt(np.linalg.det(-h_split))))
        ll_s = self.split_log_liklihood(mu_l, mu_r, l_ind, r_ind)

        return ll_s + logprod_s

    def get_ML_join(self, l_ind, r_ind):
        mu_share, h_join = self.NR_join(l_ind, r_ind, True)
        logprod_j = np.log((2 * np.pi) ** (h_join.shape[0] / 2) * (1 / np.sqrt(np.linalg.det(-h_join))))
        ll_j = self.join_log_liklihood(mu_share, l_ind, r_ind)
        return logprod_j + ll_j

    def log_p_k(self, l_ind, r_ind, use_mid = False):
        # iterate through the possible breakpoints and return normalized value of breakpoint k

        st = l_ind[0]
        end = r_ind[1]

        M_lst = []
        for mid in np.r_[st+1,end-1]:
            M_lst.append(self.get_ML_split((st,mid),(mid,end)))
        M_arr = np.array(M_lst)
        if use_mid:
            return M_lst[l_ind[1] - l_ind[0]] - self.LSE(M_arr)

        else:
            #find the best place to split ourselves
            argmax = np.argmax(M_arr)
            arg_ind = st + argmax
            return M_arr[argmax] - self.LSE(M_arr), arg_ind

    def get_logq_join(self, l_ind, r_ind):
        return (self.log_p_k(l_ind, r_ind, use_mid=True) - np.log(len(self.cov_breaks))) - (-np.log(len(self.cov_breaks) - 1))

    def get_logq_split(self, l_pk):
        return -np.log(len(self.cov_breaks)) - (l_pk- np.log(len(self.cov_breaks) - 1))

    def run(self):
        for i in tqdm.tqdm(range(100)):

            # pick a breakpoint at random
            break_pick = self.cov_breaks[np.random.choice(len(self.cov_breaks))]

            #run the optimization with current m_i values
            self.NR_simp()

            # decide whether to join or split
            if np.random.rand() < 0.5:
                # join
                # find intervals from either side
                l_st, r_st = self.cov_df.iloc[break_pick:break_pick + 2]['SegID'].values

                l_ind = (l_st, break_pick)
                r_len = self.segLenDict[r_st]
                r_ind = (break_pick + 1, r_st + r_len)

                ML_join = self.get_ML_join(l_ind, r_ind)
                ML_split = self.get_ML_split(l_ind, r_ind)

                log_q_join = self.get_logq_join(l_ind, r_ind)

                if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split + log_q_join):
                    # join the segments

                    self.cov_df.iloc[r_ind[0]:r_ind[1]]['SegID'] = l_st
                    self.segLenDict.discard(r_st)
                    cur_len = self.segLenDict[l_st]
                    self.segLenDict[l_st] = cur_len + r_len

                    # update segment list
                    self.F.discard(r_st)
                    self.F.discard(r_st - 1)
                    self.segment_history.append(self.F)

                    #update mu_i values
                    self.mu_i[l_st:r_ind[1]] = self.mu_shared
            else:
                # split
                # use segment from left
                l_st = self.cov_df.iloc[break_pick]['SegID'].values
                seg_len = self.segLenDict[l_st]

                # if singleton segment we cant split, so skip
                if seg_len == 1:
                    continue

                seg_ind = (l_st, l_st + seg_len)

                # find where to split
                l_pk, mid = self.log_p_k((l_st, None), (None, seg_ind[1]))

                l_ind = (l_st, mid)
                r_ind = (mid, seg_ind[1])

                ML_join = self.get_ML_join(l_ind, r_ind)
                ML_split = self.get_ML_split(l_ind, r_ind)

                log_q_split = self.get_logq_join(l_pk)

                if np.log(np.random.rand()) < np.minimum(0, ML_split - ML_join + log_q_split):
                    # split the segments

                    self.cov_df.iloc[r_ind[0]:r_ind[1]]['SegID'] = r_ind[0]
                    r_len = r_ind[1] - r_ind[0]
                    self.segLenDict[r_ind[0]] = r_len
                    self.segLenDict[l_st] = self.segLenDict[l_st] - r_len

                    # update segment list
                    self.F.add(r_ind[0])
                    self.F.add(r_ind[0])
                    self.segment_history.append(self.F)

                    # update mu_i values
                    self.mu_i[l_ind[0]:l_ind[1]] = self.mu_l
                    self.mu_i[r_ind[0]:r_ind[1]] = self.mu_r
