import colorama
import copy
import distinctipy
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import more_itertools
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

from capy import seq

class A_DP:
    def __init__(self, allelic_segs_pickle, wgs, ref_fasta = None, min_seg_len = 0, min_seg_snps = 0):
        if min_seg_len != 0:
            try:
                min_seg_len=int(float(min_seg_len))
            except ValueError as ex:
                print('"%s" cannot be converted to an integer: %s' % (min_seg_len, ex))
        if min_seg_snps != 0:
            try:
                min_seg_snps=int(float(min_seg_snps))
            except ValueError as ex:
                print('"%s" cannot be converted to an integer: %s' % (min_seg_snps, ex))

        # dataframe of allelic imbalance segmentation samples for each chromosome arm
        self.allelic_segs = pd.read_pickle(allelic_segs_pickle).dropna(axis = 0)
        # if some chromsome arms couldn't find the MLE, just use current state of chain
        none_idx = self.allelic_segs["results"].apply(lambda x : x.breakpoints_MLE is None)
        for i in none_idx[none_idx].index:
            self.allelic_segs.iloc[i]["results"].breakpoints_MLE = self.allelic_segs.iloc[i]["results"].breakpoints

        # load SNPs
        self.SNPs = []
        clust_offset = 0
        for _, H in self.allelic_segs.iterrows():
            S = copy.deepcopy(H["results"].P)
            S["A_alt"] = 0
            S.loc[S["aidx"], "A_alt"] = S.loc[S["aidx"], "ALT_COUNT"]
            S["A_ref"] = 0
            S.loc[S["aidx"], "A_ref"] = S.loc[S["aidx"], "REF_COUNT"]
            S["B_alt"] = 0
            S.loc[~S["aidx"], "B_alt"] = S.loc[~S["aidx"], "ALT_COUNT"]
            S["B_ref"] = 0
            S.loc[~S["aidx"], "B_ref"] = S.loc[~S["aidx"], "REF_COUNT"]

            S = S.rename(columns = { "MIN_COUNT" : "min", "MAJ_COUNT" : "maj" })
            S = S.loc[:, ["chr", "pos", "min", "maj", "A_alt", "A_ref", "B_alt", "B_ref"]]

            # set initial cluster assignments based on segmentation
            S["clust"] = -1
            bpl = np.array(H["results"].breakpoints_MLE); bpl = np.c_[bpl[0:-1], bpl[1:]]
            for i, (st, en) in enumerate(bpl):
                # remove segments that are too short/have too few SNPs
                if (min_seg_len > 0 and S.iloc[en - 1]["pos"] - S.iloc[st]["pos"] < min_seg_len) \
                or (min_seg_snps > 0 and en - st < min_seg_snps):
                    clust_offset -= 1
                    continue
                S.iloc[st:en, S.columns.get_loc("clust")] = i + clust_offset
            clust_offset += i + 1

            # bug in segmentation omits final SNP?
            #S = S.iloc[:-1]
            #assert (S["clust"] != -1).all()

            self.SNPs.append(S)

        self.SNPs = pd.concat(self.SNPs, ignore_index = True)
        self.SNPs = self.SNPs.loc[self.SNPs["clust"] != -1].reset_index(drop = True)

        # convert chr-relative positions to absolute genomic coordinates
        self.ref_fasta = ref_fasta
        self.SNPs["pos_gp"] = seq.chrpos2gpos(self.SNPs["chr"], self.SNPs["pos"], ref = self.ref_fasta)

        # initial phasing orientation
        self.SNPs["flipped"] = False

        self.N_clust_samps = 100

        self.wgs = wgs

        # assignment of SNPs to DP clusters for each MCMC sample
        self.snps_to_clusters = None
        # phase correction of SNPs for each MCMC sample
        self.snps_to_phases = None
        # likelihoods of each clustering
        self.likelihoods = None

    def run(self):
        self.DP_run = DPinstance(
          self.SNPs,
          dp_count_scale_factor = self.SNPs["clust"].value_counts().mean(),
          betahyp = None if self.wgs else 1
        )
        self.snps_to_clusters, self.snps_to_phases, self.likelihoods = self.DP_run.run(n_samps = self.N_clust_samps)

        return self.snps_to_clusters, self.snps_to_phases, self.likelihoods

class DPinstance:
    def __init__(self, S, clust_prior = sc.SortedDict(), clust_count_prior = sc.SortedDict(), alpha = 1, dp_count_scale_factor = 1, betahyp = None):
        self.S = S
        self.clust_prior = clust_prior.copy()
        self.clust_count_prior = clust_count_prior.copy()
        self.alpha = alpha
        self.dp_count_scale_factor = dp_count_scale_factor

        self.mm_mat = self.S.loc[:, ["min", "maj"]].values.reshape(-1, order = "F") # numpy for speed
        self.ref_mat = self.S.loc[:, ["A_ref", "B_ref"]].values.reshape(-1, order = "F")
        self.alt_mat = self.S.loc[:, ["A_alt", "B_alt"]].values.reshape(-1, order = "F")

        self.betahyp = self.S.loc[:, ["min", "maj"]].sum(1).mean()/2 if betahyp is None else betahyp

        #
        # define column indices
        self.clust_col = self.S.columns.get_loc("clust")
        self.min_col = self.S.columns.get_loc("min")
        self.maj_col = self.S.columns.get_loc("maj")
        self.aalt_col = self.S.columns.get_loc("A_alt")
        self.aref_col = self.S.columns.get_loc("A_ref")
        self.balt_col = self.S.columns.get_loc("B_alt")
        self.bref_col = self.S.columns.get_loc("B_ref")
        self.flip_col = self.S.columns.get_loc("flipped")

        #
        # initialize priors

        # store likelihoods for each cluster in the prior (from previous iterations)
        self.clust_prior[-1] = np.r_[0, 0]
        self.clust_prior_liks = sc.SortedDict({ k : ss.betaln(v[0] + 1 + self.betahyp, v[1] + 1 + self.betahyp) for k, v in self.clust_prior.items()})
        self.clust_prior_mat = np.r_[self.clust_prior.values()]

        self.clust_count_prior[-1] = self.alpha # DP alpha factor, i.e. relative probability of opening new cluster

    def _Siat_ph(self, ridx, min = True):
        # min, flip => maj
        # ~min, ~flip => maj
        # min, ~flip => min
        # ~min, flip => min
        col = self.min_col if self.S.iat[ridx, self.flip_col] ^ min else self.maj_col
        return self.S.iat[ridx, col]

    def _Ssum_ph(self, seg_idx, min = True):
        #flip = self.flip_mat[seg_idx]
        flip = self.S.iloc[seg_idx, self.flip_col]
        flip_n = ~flip
        if min:
            return self.mm_mat[np.r_[seg_idx[flip_n], seg_idx[flip] + len(self.S)]].sum()
        else:
            return self.mm_mat[np.r_[seg_idx[flip], seg_idx[flip_n] + len(self.S)]].sum()

    def _Scumsum_ph(self, seg_idx, min = True):
        flip = self.S.iloc[seg_idx, self.flip_col]
        flip_n = ~flip
        if min:
            si = np.argsort(np.r_[seg_idx[flip_n], seg_idx[flip]])
            return self.mm_mat[np.r_[seg_idx[flip_n], seg_idx[flip] + len(self.S)]][si].cumsum()
        else:
            si = np.argsort(np.r_[seg_idx[flip], seg_idx[flip_n]])
            return self.mm_mat[np.r_[seg_idx[flip], seg_idx[flip_n] + len(self.S)]][si].cumsum()

    def compute_rephase_prob(self, seg_idx):
        # TODO: compute logcdf/logsf directly
        flip = self.S.iloc[seg_idx, self.flip_col]
        flip_n = ~flip

        A_a = self.alt_mat[np.r_[seg_idx[flip_n], seg_idx[flip] + len(self.S)]].sum() + 1 + self.betahyp
        A_b = self.ref_mat[np.r_[seg_idx[flip_n], seg_idx[flip] + len(self.S)]].sum() + 1 + self.betahyp
        B_a = self.alt_mat[np.r_[seg_idx[flip], seg_idx[flip_n] + len(self.S)]].sum() + 1 + self.betahyp
        B_b = self.ref_mat[np.r_[seg_idx[flip], seg_idx[flip_n] + len(self.S)]].sum() + 1 + self.betahyp

        # use normal approximation to beta if conditions are right
        if A_a > 20 and A_b > 20 and B_a > 20 and B_b > 20:
            m_x = A_a/(A_a + A_b)
            s_x = A_a*A_b/((A_a + A_b)**2*(A_a + A_b + 1))
            m_y = B_a/(B_a + B_b)
            s_y = B_a*B_b/((B_a + B_b)**2*(B_a + B_b + 1))

            return s.norm.cdf(0, m_y - m_x, np.sqrt(s_x + s_y))

        # Monte Carlo simulate difference of betas
        else:
            x = s.beta.rvs(A_a, A_b, size = 1000)
            y = s.beta.rvs(B_a, B_b, size = 1000)

            return (x > y).mean()

    def SJliks(self, targ_clust, upstream_clust, downstream_clust, J_a, J_b, U_a, U_b, D_a, D_b):
#            if st == en:
#                J_a = S.iat[st, min_col].sum()
#                J_b = S.iat[st, maj_col].sum()
#            else:
#                J_a = S.iloc[st:(en + 1), min_col].sum()
#                J_b = S.iloc[st:(en + 1), maj_col].sum()
        SU_a = SU_b = SD_a = SD_b = 0

        if targ_clust != -1 and targ_clust == upstream_clust:
            J_a += U_a
            J_b += U_b
        else:
            SU_a += U_a
            SU_b += U_b
        if targ_clust != -1 and targ_clust == downstream_clust:
            J_a += D_a
            J_b += D_b
        else:
            SD_a += D_a
            SD_b += D_b

        return (ss.betaln(SU_a + 1 + self.betahyp, SU_b + 1 + self.betahyp) if SU_a > 0 or SU_b > 0 else 0) + \
          ss.betaln(J_a + 1 + self.betahyp, J_b + 1 + self.betahyp) + \
          (ss.betaln(SD_a + 1 + self.betahyp, SD_b + 1 + self.betahyp) if SD_a > 0 or SD_b > 0 else 0)

    def compute_adj_prob(self, break_idx):
        if break_idx > 1:
            U_A, U_B = self.seg_sums[self.breakpoints[break_idx - 1]]
            U_cl = self.clusts[self.breakpoints[break_idx - 1]]
        else:
            U_A = U_B = 0
            U_cl = -1
        if break_idx + 2 < len(self.breakpoints):
            D_A, D_B = self.seg_sums[self.breakpoints[break_idx + 1]]
            D_cl = self.clusts[self.breakpoints[break_idx + 1]]
        else:
            D_A = D_B = 0
            D_cl = -1

        S_A, S_B = self.seg_sums[self.breakpoints[break_idx]]

        ## compute all four possible segmentations relative to neighbor, in
        ## both phasing orientations
        MLs = np.c_[
          # UTD             T  U  D
          # -^_ or -_- (U != T & T != D) (00)
          np.r_[self.SJliks(1, 0, 0, S_A, S_B, U_A, U_B, D_A, D_B),
                self.SJliks(1, 0, 0, S_B, S_A, U_A, U_B, D_A, D_B)],
          # -__ (U != T & T == D) (01)
          np.r_[self.SJliks(0, 1, 0, S_A, S_B, U_A, U_B, D_A, D_B),
                self.SJliks(0, 1, 0, S_B, S_A, U_A, U_B, D_A, D_B)],
          # --_ (U == T & T != D) (10)
          np.r_[self.SJliks(1, 1, 0, S_A, S_B, U_A, U_B, D_A, D_B),
                self.SJliks(1, 1, 0, S_B, S_A, U_A, U_B, D_A, D_B)],
          # --- (U == T & T == D) (11)
          np.r_[self.SJliks(0, 0, 0, S_A, S_B, U_A, U_B, D_A, D_B),
                self.SJliks(0, 0, 0, S_B, S_A, U_A, U_B, D_A, D_B)],
        ]

        ## match probs to cluster choices (will match MLs matrix in main calculation)
        probs = np.full([len(self.clust_sums), 2], MLs[0, 0])
        if U_cl == D_cl and U_cl != -1 and D_cl != -1:
            probs[self.clust_sums.index(U_cl), :] = MLs[:, 3]
            probs[self.clust_sums.index(D_cl), :] = MLs[:, 3]
        else:
            if U_cl != -1:
                probs[self.clust_sums.index(U_cl), :] = MLs[:, 2]
            if D_cl != -1:
                probs[self.clust_sums.index(D_cl), :] = MLs[:, 1]

        return probs

    def compute_adj_liks(self, seg_idx, cur_clust):
        # idea to simplify this code:
        # - strip out logic for working with noncontiguous seg_idx's
        # - compute all four possibile segmentations:
        #   ABC, AAB, ABB, AAA
        # - associate those segmentations with each cluster choice, in order
        #   to return `adj_BC` with same size as `MLs`

        adj_AB = 0
        adj_BC = np.zeros([len(self.clust_sums), 2])

        # start/end coordinates of consecutive runs of segments being moved
        # NOTE: ordpairs represents closed intervals!
        ordpairs = np.c_[
          [np.r_[list(x)][[0, -1]] for x in more_itertools.consecutive_groups(
            np.sort(seg_idx))
          ]
        ]

        UD_counts = np.zeros([len(ordpairs), 4])
        adj_clusters = np.full([len(ordpairs), 2], -1)

        for o, (st, en) in enumerate(ordpairs):
            # maj/min counts of contiguous upstream segments belonging to the same cluster
            if st - 1 > 0:
                j = 1

                U_cl = self.clusts[st - j]
                adj_clusters[o, 0] = U_cl

                while st - j > 0 and self.clusts[st - j] != -1 and \
                  self.clusts[st - j] == U_cl:
                    UD_counts[o, 0] += self._Siat_ph(st - j, min = True)
                    UD_counts[o, 1] += self._Siat_ph(st - j, min = False)

                    j += 1

            # maj/min counts of contiguous downstream segments belonging to the same cluster
            if en + 1 < len(self.S):
                j = 1

                D_cl = self.clusts[en + j]
                adj_clusters[o, 1] = D_cl

                while en + j < len(self.S) - 1 and self.clusts[en + j] != -1 and \
                  self.clusts[en + j] == D_cl:
                    UD_counts[o, 2] += self._Siat_ph(en + j, min = True)
                    UD_counts[o, 3] += self._Siat_ph(en + j, min = False)

                    j += 1

        # if there are any segments being moved adjacent to already existing clusters, get local split/join likelihoods
        adj_idx = ~(adj_clusters == -1).all(1)

        if adj_idx.any():
            # for each segment/segment block within this cluster,
            for j in np.flatnonzero(adj_idx):
                # index of cluster upstream of the segment(s) being moved
                cl_u = adj_clusters[j, 0]
                # index of cluster downstream of the segment(s) being moved
                cl_d = adj_clusters[j, 1]

                # min/maj counts of upstream contiguous segments belonging to the same cluster
                U_a = UD_counts[j, 0]
                U_b = UD_counts[j, 1]
                # min/maj counts of downstream contiguous segments belonging to the same cluster
                D_a = UD_counts[j, 2]
                D_b = UD_counts[j, 3]

                # min/maj counts of the segment(s) being moved
                st = ordpairs[j, 0]
                en = ordpairs[j, 1]
                S_a = self._Ssum_ph(np.r_[st:(en + 1)], min = True) # en + 1 because ordpairs is a closed interval
                S_b = self._Ssum_ph(np.r_[st:(en + 1)], min = False) 

                # adjacency likelihood of this segment remaining where it is
#                adj_AB += self.SJliks(
#                  targ_clust = cur_clust, 
#                  upstream_clust = cl_u, 
#                  downstream_clust = cl_d, 
#                  J_a = S_a, 
#                  J_b = S_b,
#                  U_a = U_a,
#                  U_b = U_b,
#                  D_a = D_a,
#                  D_b = D_b
#                )

                # adjacency likelihood of this segment joining each possible cluster:
                # 1. those it is actually adjacent to (+ new cluster)
                for cl in {-1, cl_u, cl_d}:
                    idx = self.clust_sums.index(cl)
                    adj_BC[idx, 0] += self.SJliks(
                      targ_clust = cl, 
                      upstream_clust = cl_u, 
                      downstream_clust = cl_d, 
                      J_a = S_a, 
                      J_b = S_b,
                      U_a = U_a,
                      U_b = U_b,
                      D_a = D_a,
                      D_b = D_b
                    )
                    adj_BC[idx, 1] += self.SJliks(
                      targ_clust = cl, 
                      upstream_clust = cl_u, 
                      downstream_clust = cl_d, 
                      J_a = S_b, 
                      J_b = S_a,
                      U_a = U_a,
                      U_b = U_b,
                      D_a = D_a,
                      D_b = D_b
                    )

                # 2. clusters it is not adjacent to (use default split value)
                for cl in self.clust_sums.keys() - ({-1} | set(adj_clusters[adj_idx].ravel())):
                    idx = self.clust_sums.index(cl)
                    adj_BC[idx, 0] += self.SJliks(
                      targ_clust = -1, 
                      upstream_clust = -1, 
                      downstream_clust = -1, 
                      J_a = S_a, 
                      J_b = S_b,
                      U_a = U_a,
                      U_b = U_b,
                      D_a = D_a,
                      D_b = D_b
                    )
                    adj_BC[idx, 1] += self.SJliks(
                      targ_clust = -1, 
                      upstream_clust = -1, 
                      downstream_clust = -1, 
                      J_a = S_b, 
                      J_b = S_a,
                      U_a = U_a,
                      U_b = U_b,
                      D_a = D_a,
                      D_b = D_b
                    )

        return adj_AB, adj_BC

    def compute_cluster_splitpoints(self, seg_idx):
        spl = []

        # left bias
        end = len(seg_idx)
        i = 0
        while True:
            seg_idx_sp = seg_idx[0:end]
            if len(seg_idx_sp) < 2:
                break

            min_cs = self._Scumsum_ph(seg_idx_sp, min = True)
            min_csr = self._Ssum_ph(seg_idx_sp, min = True) - min_cs
            maj_cs = self._Scumsum_ph(seg_idx_sp, min = False)
            maj_csr = self._Ssum_ph(seg_idx_sp, min = False) - maj_cs

            split_lik = ss.betaln(min_cs + 1 + self.betahyp, maj_cs + 1 + self.betahyp) + ss.betaln(min_csr + 1 + self.betahyp, maj_csr + 1 + self.betahyp)
            # split_lprob = split_lik - split_lik.max() - np.log(np.exp(split_lik - split_lik.max()).sum())
            # NOTE: instead of argmax, probabilistically choose? will this make a difference?

            end = split_lik.argmax()
            spl.append(end)

            if end <= 1 or end == len(split_lik) - 1:
                break

            i += 1

        # right bias
        start = 0
        i = 0
        while True:
            seg_idx_sp = seg_idx[start:]
            if len(seg_idx_sp) < 2:
                break

            min_cs = self._Scumsum_ph(seg_idx_sp, min = True)
            min_csr = self._Ssum_ph(seg_idx_sp, min = True) - min_cs
            maj_cs = self._Scumsum_ph(seg_idx_sp, min = False)
            maj_csr = self._Ssum_ph(seg_idx_sp, min = False) - maj_cs

            split_lik = ss.betaln(min_cs[:-1] + 1 + self.betahyp, maj_cs[:-1] + 1 + self.betahyp) + ss.betaln(min_csr[1:] + 1 + self.betahyp, maj_csr[1:] + 1 + self.betahyp)
            # split_lprob = split_lik - split_lik.max() - np.log(np.exp(split_lik - split_lik.max()).sum())

            start += split_lik.argmax() + 1
            spl.append(start - 1)

            if start > len(seg_idx) - 1 or split_lik.argmax() == 0:
                break

            i += 1

        bdy = np.unique(np.r_[0, spl, len(seg_idx)])
        bdy = np.c_[bdy[:-1], bdy[1:]]

        return bdy

    def add_breakpoint(self, start, mid, end, clust_idx):
        """
        Add breakpoint at mid belonging to clust_idx, between start and end
        """
        self.breakpoints.add(mid)
        self.clust_members_bps[clust_idx].add(mid)
        
        A = self._Ssum_ph(np.r_[mid:end], min = True)
        B = self._Ssum_ph(np.r_[mid:end], min = False)

        self.seg_sums[mid] = np.r_[A, B]
        self.seg_sums[start] -= self.seg_sums[mid]

        self.seg_liks[mid] = ss.betaln(A + 1 + self.betahyp, B + 1 + self.betahyp)
        A = self._Ssum_ph(np.r_[start:mid], min = True)
        B = self._Ssum_ph(np.r_[start:mid], min = False)
        self.seg_liks[start] = ss.betaln(A + 1 + self.betahyp, B + 1 + self.betahyp)

        self.seg_phase_probs[start] = self.compute_rephase_prob(np.r_[start:mid])
        self.seg_phase_probs[mid] = self.compute_rephase_prob(np.r_[mid:end])

    def compute_overall_lik_simple(self):
        ## overall clustering likelihood
        # p({a_i, b_i} | {c_k}, {phase_i})
        clust_lik = np.r_[[ss.betaln(v[0] + 1 + self.betahyp, v[1] + 1 + self.betahyp) for k, v in self.clust_sums.items() if k >= 0]].sum()

        ## overall phasing likelihood
        # p({phase_i} | {a_i, b_i})
        phase_probs = np.r_[self.seg_phase_probs.values()]
        phase_lik = np.log1p(phase_probs).sum() if not np.isnan(phase_probs).any() else np.nan

        ## Dirichlet count prior (Dirichlet-categorical marginal likelihood)
        # p({c_k})
        dirvec = np.r_[self.clust_counts.values()].astype(float)/self.dp_count_scale_factor
        k = len(dirvec)
        count_prior = k*np.log(self.alpha) + ss.gammaln(dirvec).sum() + ss.gammaln(self.alpha) - ss.gammaln(dirvec.sum() + self.alpha)

        ## segmentation likelihood
        # p({a_i, b_i} | {s}, {phase_i})
        # TODO: memoize
        seg_lik = np.r_[self.seg_liks.values()].sum()

        # p({c_k}, {s}, {phase_i} | {a_i, b_i})
        return np.r_[clust_lik, phase_lik, count_prior, seg_lik]

    def run(self, n_iter = 0, n_samps = 0):
        #
        # initialize cluster tracking hash tables
        self.clust_counts = sc.SortedDict(self.S["clust"].value_counts().drop(-1, errors = "ignore"))
        # for the first round of clustering, this is { 0 : 1, 1 : 1, ..., N - 1 : 1 }

        Sgc = self.S.groupby(["clust", "flipped"])[["min", "maj"]].sum()
        if (Sgc.droplevel(0).index == True).any():
            Sgc.loc[(slice(None), True), ["min", "maj"]] = Sgc.loc[(slice(None), True), ["maj", "min"]].values
        self.clust_sums = sc.SortedDict({
          **{ k : np.r_[v["min"], v["maj"]] for k, v in Sgc.groupby(level = "clust").sum().to_dict(orient = "index").items() },
          **{-1 : np.r_[0, 0]}
        })
        # for the first round, this is { -1 : np.r_[0, 0], 0 : np.r_[S[0, "min"], S[0, "maj"]], 1 : S[1, "min"], S[1, "maj"], ..., N : S[N - 1, "min"], S[N - 1, "maj"] }

        self.clust_members = sc.SortedDict({ k : set(v) for k, v in self.S.groupby("clust").groups.items() if k != -1 })
        # for the first round, this is { 0 : {0}, 1 : {1}, ..., N - 1 : {N - 1} }

        # store this as numpy for speed
        self.clusts = self.S["clust"].values

        max_clust_idx = np.max(self.clust_members.keys() | self.clust_prior.keys() if self.clust_prior is not None else {})

        #
        # breakpoint tracking

        # segmentation breakpoints
        self.breakpoints = sc.SortedSet(np.flatnonzero(np.diff(self.S["clust"]) != 0) + 1) | {0, len(self.S)}

        # min/maj counts in each segment
        self.seg_sums = sc.SortedDict()
        bpl = np.r_[self.breakpoints]
        for st, en in np.c_[bpl[:-1], bpl[1:]]:
            mn = self._Ssum_ph(np.r_[st:en], min = True)
            mj = self._Ssum_ph(np.r_[st:en], min = False)
            self.seg_sums[st] = np.r_[mn, mj]

        # likelihoods for each segment
        self.seg_liks = sc.SortedDict()
        for k, (a, b) in self.seg_sums.items():
            self.seg_liks[k] = ss.betaln(a + 1 + self.betahyp, b + 1 + self.betahyp)

        # breakpoints for each cluster
        self.clust_members_bps = sc.SortedDict({
          k : sc.SortedSet(v) for k, v in \
            self.S.loc[self.breakpoints[:-1], ["clust"]].groupby("clust").groups.items()
        })

        # misphase probabilities for each segment
        self.seg_phase_probs = sc.SortedDict({ k : np.nan for k in self.breakpoints[:-1] })

        # containers for saving the MCMC trace
        self.snps_to_clusters = []
        self.phase_orientations = []
        self.segment_trace = []
        self.likelihood_trace = []

        # likelihood trace for checking burnin status
        self.lik_trace = []
        burned_in = False
        self.burnin_iteration = -1
        touch90 = False
        likelihood_ready = False

        n_it = 0
        n_it_last = 0

        brk = 0

        while True:
            if not n_it % 1000:
                if len(self.clust_counts) > 20:
                    print(pd.Series(self.clust_counts.values()).value_counts().sort_index())
                else:
                    print("\n".join([str(self.clust_counts[k]) + ": " + str(x/(x + y)) for k, (x, y) in self.clust_sums.items() if k != -1]))
                if likelihood_ready:
                    print("[{}] Likelihood: {}".format("*" if burned_in else " ", self.lik_trace[-1].sum()))
                if burned_in:
                    print("{}/{} MCMC samples collected".format(len(self.snps_to_clusters), n_samps))

            # stop after a raw number of iterations
            if n_iter > 0 and n_it > n_iter:
                return

            # stop after a number of samples have been taken
            if n_samps > 0 and len(self.snps_to_clusters) > n_samps:
                break

            # poll every 100 iterations for various statuses
            if not n_it % min(len(self.breakpoints), 100):
                # have >95% of segments been touched?
                if (1 - (1 - 1/len(self.breakpoints))**n_it) > 0.95:
                    touch90 = True

                # start computing likelihoods
                if touch90:
                    lik = self.compute_overall_lik_simple()
                    # phasing likelihood will be NaN until we've touched every singlesegment
                    if not np.isnan(lik).any():
                        self.lik_trace.append(lik)
                        likelihood_ready = True

                # check if likelihood has stabilized enough to consider us "burned in"
                # also include contingency if we've unambiguously converged on an optimum and chain has not moved at all
                if likelihood_ready and not burned_in and len(self.lik_trace) > 500:
                    lt = np.vstack(self.lik_trace).sum(1)
                    if (np.convolve(np.diff(lt), np.ones(500)/500, mode = "same") < 0).sum() > 2 or\
                        (np.diff(lt[-500:]) == 0).all():
                        print("BURNED IN")
                        burned_in = True
                        self.burnin_iteration = len(self.lik_trace)
                        n_it_last = n_it

            #
            # pick  a segment to move

# diagnostic code to compute overall likelihood before move
#            compute_lik = False
#            lik_before = np.nan
#            if touch90 and np.random.rand() < 0.1:
#                compute_lik = True
#                lik_before = self.compute_overall_lik_simple()

            # >90% of segments have been moved; we are iterating over segments sequentially
            if touch90:
                break_idx = sc.SortedSet({brk % (len(self.breakpoints) - 1)})
                brk += 1
            # we are picking segments at random
            else:
                break_idx = sc.SortedSet({np.random.choice(len(self.breakpoints) - 1)})

            # get all SNPs within this segment
            seg_st = self.breakpoints[break_idx[0]]
            seg_en = self.breakpoints[break_idx[0] + 1]
            seg_idx = np.r_[seg_st:seg_en]

            cur_clust = int(self.clusts[seg_idx[0]])

            # propose breaking this segment
            if np.random.rand() < 0.1:
                # can't split segments of length 1
                if len(seg_idx) == 1:
                    n_it += 1
                    continue

                # TODO: memoize cumsums?
                min_cs = self._Scumsum_ph(seg_idx, min = True)
                min_csr = self.seg_sums[seg_idx[0]][0] - min_cs
                maj_cs = self._Scumsum_ph(seg_idx, min = False)
                maj_csr = self.seg_sums[seg_idx[0]][1] - maj_cs

                split_lik = ss.betaln(min_cs + 1 + self.betahyp, maj_cs + 1 + self.betahyp) + ss.betaln(min_csr + 1 + self.betahyp, maj_csr + 1 + self.betahyp)
                split_lik[-1] = ss.betaln(min_cs[-1] + 1 + self.betahyp, maj_cs[-1] + 1 + self.betahyp)
                split_lik -= split_lik.max()
                split_point = np.random.choice(np.r_[0:len(seg_idx)], p = np.exp(split_lik)/np.exp(split_lik).sum())
                seg_idx = seg_idx[:(split_point + 1)]

                # add breakpoint (can be erased subsequently if segment rejoins original cluster)
                new_bp = seg_idx[-1] + 1
                if len(seg_idx) < seg_en - seg_st: # don't add breakpoint if we're not splitting segment
                    self.add_breakpoint(start = seg_idx[0], mid = new_bp, end = seg_en, clust_idx = cur_clust)

            # propose splitting out a contiguous interval of segments within the current cluster {{{
            split_clust = False
            if False and touch90 and np.random.rand() < 0.1:
                # TODO: if we use cur_clust, this will be biased towards larger clusters. is this desireable?
                clust_snps = np.sort(np.r_[list(self.clust_members[cur_clust])])

                # can't split clusters of length 1
                if len(clust_snps) == 1:
                    n_it += 1
                    continue

                split_bdy = self.compute_cluster_splitpoints(clust_snps)

                A_tot, B_tot = self.clust_sums[cur_clust]

                lik0 = ss.betaln(A_tot + 1 + self.betahyp, B_tot + 1 + self.betahyp)

                liks = np.zeros(len(split_bdy) + 1)
                liks[-1] = lik0 # don't split at all

                # likelihood ratios for splitting each region into a new cluster
                for i, (st, en) in enumerate(split_bdy):
                    A = self._Ssum_ph(clust_snps[st:en], min = True)
                    B = self._Ssum_ph(clust_snps[st:en], min = False)

                    liks[i] = ss.betaln(A_tot - A + 1 + self.betahyp, B_tot - B + 1 + self.betahyp) + ss.betaln(A + 1 + self.betahyp, B + 1 + self.betahyp)

                # pick a region to split
                split_idx = np.random.choice(
                  len(split_bdy) + 1,
                  p = np.exp(liks - liks.max())/np.exp(liks - liks.max()).sum()
                )

                # don't split at all
                if split_idx == len(split_bdy):
                    n_it += 1
                    continue

                # seg_idx == SNPs to propose to split off
                seg_idx = clust_snps[slice(*split_bdy[split_idx])]

                split_clust = True

                # add breakpoints
                for si in [seg_idx[0], seg_idx[-1]]:
                    if si not in self.breakpoints:
                        seg_st_idx = self.breakpoints.bisect_left(si) - 1
                        seg_st = self.breakpoints[seg_st_idx]
                        seg_en_idx = self.breakpoints.bisect_left(si)
                        seg_en = self.breakpoints[seg_en_idx]

                        self.add_breakpoint(start = seg_st, mid = si, end = seg_en, clust_idx = cur_clust)

                # get all breakpoints within this cluster/interval
                left_idx = self.clust_members_bps[cur_clust].bisect_left(seg_idx[0])
                right_idx = self.clust_members_bps[cur_clust].bisect_right(seg_idx[-1])
                break_idx = sc.SortedSet([self.breakpoints.index(x) for x in self.clust_members_bps[cur_clust][left_idx:right_idx]])

            # }}}

            n_move = len(seg_idx)

            # if segment was already assigned to a cluster, unassign it
            if cur_clust >= 0:
                self.clust_counts[cur_clust] -= n_move
                if self.clust_counts[cur_clust] == 0:
                    del self.clust_counts[cur_clust]
                    del self.clust_sums[cur_clust]
                    del self.clust_members[cur_clust]
                    del self.clust_members_bps[cur_clust]
                else:
                    self.clust_sums[cur_clust] -= np.r_[self._Ssum_ph(seg_idx, min = True), self._Ssum_ph(seg_idx, min = False)]
                    self.clust_members[cur_clust] -= set(seg_idx)
                    for b in break_idx:
                        self.clust_members_bps[cur_clust].remove(self.breakpoints[b])

                self.clusts[seg_idx] = -1

            #
            # perform phase correction on segment/cluster
            # flip min/maj with probability that alleles are oriented the "wrong" way
#            if not np.isnan(self.seg_phase_probs[seg_idx[0]]):
#                rfp = self.compute_rephase_prob(seg_idx)
#                rfp_mem = self.seg_phase_probs[seg_idx[0]]
#                if np.abs(rfp - rfp_mem) > 0.05:
#                    print(rfp_mem, rfp)
#                    breakpoint()
            if np.isnan(self.seg_phase_probs[seg_idx[0]]):
                self.seg_phase_probs[seg_idx[0]] = self.compute_rephase_prob(seg_idx)
            rephase_prob = self.seg_phase_probs[seg_idx[0]]

            #
            # choose to join a cluster or make a new one
            # probabilities determined by similarity of segment/cluster to existing ones

            # B is segment/cluster to move
            # A is cluster B is currently part of
            # C is all possible clusters to move to
            A_a = self.clust_sums[cur_clust][0] if cur_clust in self.clust_sums else 0
            A_b = self.clust_sums[cur_clust][1] if cur_clust in self.clust_sums else 0
            B_a = self._Ssum_ph(seg_idx, min = True)
            B_b = self._Ssum_ph(seg_idx, min = False)
            C_ab = np.r_[self.clust_sums.values()] # first terms: -1 = make new cluster
            #C_ab = np.r_[[v for k, v in clust_sums.items() if k != cur_clust or cur_clust == -1]] # if we don't want to explicitly propose letting B rejoin cur_clust

            # A+B,C -> A,B+C

            # A+B is likelihood of current cluster B is part of
            #AB = ss.betaln(A_a + B_a + 1, A_b + B_b + 1)
            # C is likelihood of target cluster pre-join
            C = ss.betaln(C_ab[:, 0] + 1 + self.betahyp, C_ab[:, 1] + 1 + self.betahyp)
            C[0] = 0 # don't count prior twice when opening a new cluster
            # A is likelihood cluster B is part of, minus B
            #A = ss.betaln(A_a + 1, A_b + 1)
            # B+C is likelihood of target cluster post-join, with both phase orientations
            BC = ss.betaln(C_ab[:, [0]] + np.c_[B_a, B_b] + 1 + self.betahyp, C_ab[:, [1]] + np.c_[B_b, B_a] + 1 + self.betahyp)

            MLs = BC - C[:, None]

            #
            # priors

            ## prior on previous cluster fractions {{{

            prior_diff = []
            prior_com = []
            clust_prior_p = 1
            if self.clust_prior is not None: 
                #
                # divide prior into three sections:
                # * clusters in prior not currently active (if picked, will open a new cluster with that ID)
                # * clusters in prior currently active (if picked, will weight that cluster's posterior probability)
                # * currently active clusters not in the prior (if picked, would weight cluster's posterior probability with prior probability of making brand new cluster)

                # not currently active
                prior_diff = self.clust_prior.keys() - self.clust_counts.keys()

                # currently active clusters in prior
                prior_com = self.clust_counts.keys() & self.clust_prior.keys()

                # currently active clusters not in prior
                prior_null = self.clust_counts.keys() - self.clust_prior.keys()

                # order of prior vector:
                # [-1 (totally new cluster), <prior_diff>, <prior_com + prior_null>]
                prior_idx = np.r_[
                  np.r_[[self.clust_prior.index(x) for x in prior_diff]],
                  np.r_[[self.clust_prior.index(x) if x in self.clust_prior else 0 for x in (prior_com | prior_null)]]
                ].astype(int)

                # prior marginal likelihoods for both phase orientations
                prior_MLs = ss.betaln( # prior clusters + segment
                  np.c_[self.clust_prior_mat[prior_idx, 0]] + np.c_[B_a, B_b] + 1,
                  np.c_[self.clust_prior_mat[prior_idx, 1]] + np.c_[B_b, B_a] + 1
                ) \
                - np.c_[ss.betaln(B_a + 1, B_b + 1) + np.r_[np.r_[self.clust_prior_liks.values()][prior_idx]]] # prior clusters, segment

                clust_prior_p = np.maximum(np.exp(prior_MLs - prior_MLs.max())/np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                # expand MLs to account for multiple new clusters
                MLs = np.r_[np.full([len(prior_diff), 2], MLs[0]), MLs[1:, :]]

            # }}}
                
            ## DP prior based on clusters sizes
            n_c = np.c_[self.clust_counts.values()]/self.dp_count_scale_factor
            M = n_move/self.dp_count_scale_factor
            N = n_c.sum() + M
            log_count_prior = np.full([len(self.clust_sums), 1], np.nan)
            log_count_prior[1:] = ss.gammaln(M + n_c) + ss.gammaln(N + self.alpha - M) \
              - (ss.gammaln(n_c) + ss.gammaln(N + self.alpha))
            # probability of opening a new cluster
            log_count_prior[0] = ss.gammaln(M) + np.log(self.alpha) + ss.gammaln(N + self.alpha - M) - ss.gammaln(N + self.alpha)

            # p(phase|X)
            log_phase_prob = np.log(np.maximum(1e-300, np.r_[1 - rephase_prob, rephase_prob]))

            #
            # adjacent segment likelihood

            log_adj_lik = self.compute_adj_prob(break_idx[0])
 
            # p(X|clust,phase)p(X|seg,phase)p(clust)p(phase)
            num = (MLs               # p({a_i, b_i}_{i\in B} | {a_i, b_i}_{i\in clust}, phase_{i\in B})
                  + log_adj_lik      # p({a_i, b_i}_{i\in B} | U, D, phase_{i\in B})
                  + log_count_prior  # p(clust) (DP prior on clust counts)
                  + log_phase_prob)  # p(phase)

            num -= num.max() # avoid underflow in sum-exp

            # p(clust,phase|X)
            choice_p = np.exp(num - np.log(np.exp(num).sum()))

            # row major indexing: choice_idx//2 = cluster index, choice_idx & 1 = rephase true
            choice_idx = np.random.choice(
              np.r_[0:np.prod(choice_p.shape)],
              p = choice_p.ravel()
            )
            # -1 = brand new, -2, -3, ... = -(prior clust index) - 2
            choice = np.r_[-np.r_[prior_diff] - 2, self.clust_counts.keys()][choice_idx//2]

            # save rephasing status
            if choice_idx & 1:
                self.S.iloc[seg_idx, self.flip_col] = ~self.S.iloc[seg_idx, self.flip_col]
                for b in break_idx:
                    st = self.breakpoints[b]
                    en = self.breakpoints[b + 1]
                    self.seg_sums[st] = self.seg_sums[st][::-1]

            # create new cluster
            if choice < 0:
                # if we are moving an entire cluster, give it the same index it used to have
                # otherwise, cluster indices will be inconsistent
                if cur_clust not in self.clust_counts:
                    new_clust_idx = cur_clust
                else: # totally new cluster
                    max_clust_idx += 1
                    new_clust_idx = max_clust_idx

                self.clust_counts[new_clust_idx] = n_move
                self.S.iloc[seg_idx, self.clust_col] = new_clust_idx
                self.clusts[seg_idx] = new_clust_idx

                self.clust_sums[new_clust_idx] = np.r_[B_a, B_b] if not choice_idx & 1 else np.r_[B_b, B_a]
                self.clust_members[new_clust_idx] = set(seg_idx)

            # join existing cluster
            else:
                self.clust_counts[choice] += n_move 
                self.clust_sums[choice] += np.r_[B_a, B_b] if not choice_idx & 1 else np.r_[B_b, B_a]
                self.S.iloc[seg_idx, self.clust_col] = choice
                self.clusts[seg_idx] = choice

                self.clust_members[choice].update(set(seg_idx))

            # if segment was rephased, update saved phasing probabilities
            if choice_idx & 1:
                for bp_idx in break_idx:
                    st = self.breakpoints[bp_idx]
                    en = self.breakpoints[bp_idx + 1]
                    self.seg_phase_probs[st] = self.compute_rephase_prob(np.r_[st:en])

            # update breakpoints

            # B->A
            #    .   .     .   break_idx + 1
            # A B A B A C B A
            #  +   +     +     break_idx
            #*           *     update_idx

            break_idx_bi = break_idx | { x + 1 for x in break_idx }
            snp_idx_bi = sc.SortedSet([self.breakpoints[b] for b in break_idx_bi])
            snp_idx = sc.SortedSet([self.breakpoints[b] for b in break_idx])
            update_idx = sc.SortedSet()
            for snp in snp_idx_bi:
                if snp < len(self.S) and snp != 0 and self.clusts[snp - 1] == self.clusts[snp]:
                    snp_idx.discard(snp) # discard rather than remvoe because this could be in snp_idx + 1
                    self.breakpoints.remove(snp)
                    self.seg_sums.pop(snp)
                    self.seg_liks.pop(snp)
                    self.seg_phase_probs.pop(snp)
                    self.clust_members_bps[self.clusts[snp]].discard(snp) # discard rather than remove since this breakpoint could be in break_idx + 1, which would belong to another cluster
                    update_idx.add(self.breakpoints.bisect_left(snp) - 1)
                    snp_idx.add(self.breakpoints[self.breakpoints.bisect_left(snp) - 1])
#            if len(update_idx):
#                usnp = self.breakpoints[self.breakpoints.bisect_left(seg_idx[0]) - 1]
#                print(f"{usnp}: {self.clusts[usnp]}")
#                print(f"{snp_idx[0]}: {self.clusts[snp_idx[0]]} <")
#                print(f"{snp_idx[1]}: {self.clusts[snp_idx[1]]} <")
#                dsnp = self.breakpoints[self.breakpoints.bisect_right(seg_idx[0])]
#                print(f"{dsnp}: {self.clusts[dsnp]}")
#                print(f"Update: {self.breakpoints[update_idx[0]]}")
            for bp_idx in update_idx:
                st = self.breakpoints[bp_idx]
                en = self.breakpoints[bp_idx + 1]
                A = self._Ssum_ph(np.r_[st:en], min = True)
                B = self._Ssum_ph(np.r_[st:en], min = False)
                self.seg_sums[st] = np.r_[A, B]
                self.seg_liks[st] = ss.betaln(A + 1 + self.betahyp, B + 1 + self.betahyp)
                self.seg_phase_probs[st] = self.compute_rephase_prob(np.r_[st:en])

            if choice < 0:
                self.clust_members_bps[new_clust_idx] = snp_idx
            else:
                self.clust_members_bps[choice] |= snp_idx

# diagnostic code to check if breakpoint list is properly updated
#            if touch90:
#                x = sc.SortedSet()
#                for y in self.clust_members_bps.values():
#                    x |= y
#                if len(x) != len(self.breakpoints) - 1:
#                    breakpoint()

# diagnostic code to compute overall likelihood delta for iteration
#            if compute_lik:
#                lik_after = self.compute_overall_lik_simple()
#                lik_delta = lik_after.sum() - lik_before.sum()
#                ML_choice = num.ravel()[choice_idx]
#                if not np.isnan(lik_delta) and (lik_delta != 0 or ML_choice != 0):
#                    print("lik: {}; MLs: {}".format(lik_delta, ML_choice))
##                if lik_delta < 0 and ML_choice == 0:
##                    breakpoint()

            # save a sample from the MCMC when >95% of segments have been touched since the last iteration
            if burned_in and (1 - (1 - 1/len(self.breakpoints))**(n_it - n_it_last)) > 0.95:
                self.snps_to_clusters.append(self.S["clust"].copy())
                self.phase_orientations.append(self.S["flipped"].copy())
                self.segment_trace.append({ snp : self.S.iloc[snp, self.clust_col] for snp in self.breakpoints[:-1]})
                self.likelihood_trace.append(self.compute_overall_lik_simple().sum())
                n_it_last = n_it

            n_it += 1

        return np.r_[self.snps_to_clusters], np.r_[self.phase_orientations], np.r_[self.likelihood_trace]

    def get_unique_clust_idxs(self, snps_to_clusters = None):
        if snps_to_clusters is None:
            snps_to_clusters = np.r_[self.snps_to_clusters]
        s2cu, s2cu_j = np.unique(snps_to_clusters, return_inverse = True)
        return s2cu, s2cu_j.reshape(snps_to_clusters.shape)

    def get_colors(self):
        s2cu, s2cu_j = self.get_unique_clust_idxs()
        if len(self.breakpoints) == 2:
            return np.r_[np.c_[0.368417, 0.506779, 0.709798]]
        T = pd.DataFrame(np.c_[np.r_[self.breakpoints[:-2]], np.r_[self.breakpoints[1:-1]]], columns = ["snp_st", "snp_end"])
        T["gp_st"] = self.S.loc[T["snp_st"], "pos_gp"].values
        T["gp_end"] = self.S.loc[T["snp_end"], "pos_gp"].values
        T["terr"] = T["gp_end"] - T["gp_st"]
        T["clust"] = self.S.loc[T["snp_st"], "clust"].values

        clust_terr = T.groupby("clust")["terr"].sum()
        si = clust_terr.sort_values(ascending = False).index.argsort()
        
        base_colors = np.array([
          [0.368417, 0.506779, 0.709798],
          [0.880722, 0.611041, 0.142051],
          [0.560181, 0.691569, 0.194885],
          [0.922526, 0.385626, 0.209179],
          [0.528488, 0.470624, 0.701351],
          [0.772079, 0.431554, 0.102387],
          [0.363898, 0.618501, 0.782349],
          [1, 0.75, 0],
          [0.647624, 0.37816, 0.614037],
          [0.571589, 0.586483, 0.],
          [0.915, 0.3325, 0.2125],
          [0.400822, 0.522007, 0.85],
          [0.972829, 0.621644, 0.073362],
          [0.736783, 0.358, 0.503027],
          [0.280264, 0.715, 0.429209]
        ])
 
        extra_colors = np.array(
          distinctipy.distinctipy.get_colors(
            (clust_terr/clust_terr.sum() >= 0.003).sum() - base_colors.shape[0],
            exclude_colors = [list(x) for x in np.r_[np.c_[0, 0, 0], np.c_[1, 1, 1], np.c_[0.5, 0.5, 0.5], np.c_[1, 0, 1], base_colors]],
            rng = 1234
          )
        )


        return np.r_[base_colors, extra_colors if extra_colors.size > 0 else np.empty([0, 3])][si % (len(base_colors) + len(extra_colors))]
    
    def visualize_segs(self, f = None, ax = None, use_clust = False, show_snps = False, chroms = None):
        if ax is None:
            f = plt.figure(figsize = [16, 4]) if f is None else f
            ax = plt.gca()

        if chroms is None:
            ax.set_xlim([0, self.S["pos_gp"].max()])
        else:
            if not all(type(item) is int and item > 0 and item < 25 for item in chroms):
                raise ValueError("chromosomes must be an integer chromosome")
            ax.set_xlim([*self.S.loc[self.S["chr"].isin(chroms), "pos_gp"].iloc[[0, -1]]])
        ax.set_ylim([0, 1])

        colors = self.get_colors()
        s2cu, s2cu_j = self.get_unique_clust_idxs()

        n_samp = len(self.snps_to_clusters)

        selff = copy.deepcopy(self)

        if show_snps:
            # set SNP alpha based on number of SNPs
            logistic = lambda A, K, B, M, x : A + (K - A)/(1 + np.exp(-B*(x - M)))
            default_alpha = logistic(A = 0.4, K = 0.025, B = 0.00001, M = 120000, x = len(self.S) if chroms is None else (self.S["chr"].isin(chroms)).sum())

            ph_prob = np.r_[self.phase_orientations].mean(0)

            cu = np.searchsorted(s2cu, self.S["clust"])

            # only plot unambiguous SNPs once
            uidx = ph_prob == 0
            ax.scatter(
              self.S.loc[uidx, "pos_gp"],
              self.S.loc[uidx, "min"]/self.S.loc[uidx, ["min", "maj"]].sum(1),
              color = colors[cu[uidx] % len(colors)], marker = '.', alpha = default_alpha, s = 1
            )
            uidx = ph_prob == 1
            ax.scatter(
              self.S.loc[uidx, "pos_gp"],
              self.S.loc[uidx, "maj"]/self.S.loc[uidx, ["min", "maj"]].sum(1),
              color = colors[cu[uidx] % len(colors)], marker = '.', alpha = default_alpha, s = 1
            )

            # plot ambiguous SNPs with opacity weighted by phase probability
            nuidx = (ph_prob != 0) & (ph_prob != 1)
            if nuidx.sum() > 0: # protect against zero ambiguous segs
                ax.scatter(
                  selff.S.loc[nuidx, "pos_gp"],
                  self.S.loc[nuidx, "min"]/self.S.loc[nuidx, ["min", "maj"]].sum(1),
                  color = colors[cu[nuidx] % len(colors)], marker = '.', alpha = default_alpha*(1 - ph_prob[nuidx]), s = 1
                )
                ax.scatter(
                  selff.S.loc[nuidx, "pos_gp"],
                  self.S.loc[nuidx, "maj"]/self.S.loc[nuidx, ["min", "maj"]].sum(1),
                  color = colors[cu[nuidx] % len(colors)], marker = '.', alpha = default_alpha*ph_prob[nuidx], s = 1
                )

        for seg2c, s2ph in zip(self.segment_trace, self.phase_orientations):
            # only show maximum likelihood if we're overlaying SNPs
            if show_snps:
                mlidx = np.r_[self.likelihood_trace].argmax()
                seg2c, s2ph = self.segment_trace[mlidx], self.phase_orientations[mlidx]

            # get uniqued clust indices for each segment start
            seg_cu = np.searchsorted(s2cu, np.r_[list(seg2c.values())])

            # rephase segments according to phase orientation sample
            selff.S["flipped"] = s2ph

            seg_bdy = np.r_[list(seg2c.keys()), len(selff.S)]
            seg_bdy = np.c_[seg_bdy[:-1], seg_bdy[1:]]

            for i, (st, en) in enumerate(seg_bdy):
                if use_clust:
                    ci_lo, med, ci_hi = s.beta.ppf(
                      [0.05, 0.5, 0.95],
                      selff.clust_sums[seg2c[st]][0] + 1 + self.betahyp,
                      selff.clust_sums[seg2c[st]][1] + 1 + self.betahyp,
                    )
                else:
                    ci_lo, med, ci_hi = s.beta.ppf(
                      [0.05, 0.5, 0.95],
                      selff._Ssum_ph(np.r_[st:en], min = True) + 1 + self.betahyp,
                      selff._Ssum_ph(np.r_[st:en], min = False) + 1 + self.betahyp,
                    )
                ax.add_patch(mpl.patches.Rectangle(
                  (selff.S.iloc[st]["pos_gp"], ci_lo),
                  selff.S.iloc[en - 1]["pos_gp"] - selff.S.iloc[st]["pos_gp"],
                  np.maximum(0, ci_hi - ci_lo),
                  facecolor = colors[seg_cu[i] % len(colors)],
                  edgecolor = 'k' if show_snps else None, linewidth = 0.5 if show_snps else None,
                  fill = True, alpha = 1 if show_snps else 1/n_samp, zorder = 1000
                ))
                ax.scatter(
                  (selff.S.iloc[en - 1]["pos_gp"] + selff.S.iloc[st]["pos_gp"])/2,
                  med,
                  color = 'k',
                  marker = '.', s = 1, alpha = 1 if show_snps else 1/n_samp
                )

            if show_snps:
                break

    def visualize_clusts(self, **kwargs):
        self.visualize_segs(use_clust = True, **kwargs)

    def plot_likelihood_trace(self):
        lt = np.vstack(self.lik_trace)
        lt = lt[np.isnan(lt).sum(1) == 0, :]

        lt = lt[self.burnin_iteration:, :]

        plt.figure(); plt.clf()
        plt.scatter(np.r_[0:len(lt)], lt[:, 0] - lt[:, 0].max())
        #plt.scatter(np.r_[0:len(lt)], lt[:, 1] - lt[:, 1].max())
        plt.scatter(np.r_[0:len(lt)], lt[:, 2] - lt[:, 2].max())
        plt.scatter(np.r_[0:len(lt)], lt[:, 3] - lt[:, 3].max())
        plt.scatter(np.r_[0:len(lt)], lt.sum(1) - lt.sum(1).max(), marker = '+', color = 'k')
        plt.legend(["Clust", "DP", "Seg", "Total"])
        plt.xlabel(r"Post-burnin iteration ($\times 100$)")
        plt.ylabel(r"$\Delta$ likelihood")
        
# helper method to allow user to reload a allelic DP object for plotting using
# the output files. note that fields that are not essential for plotting may not
# reflect the true end state of the ADP
def load_DP_object_from_outputs(snps_path, dp_data_path, segmentation_path):
    snps_df = pd.read_pickle(snps_path)
    dp_data = np.load(dp_data_path)
    segmentations = pd.read_pickle(segmentation_path)
    
    self = DPinstance(snps_df, dp_count_scale_factor = snps_df['clust'].value_counts().mean())
    
    # we need to repopulate fields that are only filled during runtime
    self.breakpoints = sc.SortedSet(np.flatnonzero(np.diff(self.S["clust"]) != 0) + 1) | {0, len(self.S)}

    self.clust_counts = sc.SortedDict(self.S["clust"].value_counts().drop(-1, errors = "ignore"))
    Sgc = self.S.groupby(["clust", "flipped"])[["min", "maj"]].sum()
    if (Sgc.droplevel(0).index == True).any():
        Sgc.loc[(slice(None), True), ["min", "maj"]] = Sgc.loc[(slice(None), True), ["maj", "min"]].values
    self.clust_sums = sc.SortedDict({**{ k : np.r_[v["min"], v["maj"]] for k, v in Sgc.groupby(level = "clust").sum().to_dict(orient = "index").items() },**{-1 : np.r_[0, 0]}})    
    self.clust_members = sc.SortedDict({ k : set(v) for k, v in self.S.groupby("clust").groups.items() if k != -1 })
    
    self.segment_trace = segmentations
    self.phase_orientations = dp_data['snps_to_phases']
    self.likelihood_trace = dp_data['likelihoods']
    self.snps_to_clusters = dp_data['snps_to_clusters']
    self.clusts = self.S["clust"].values

    return self
