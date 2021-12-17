import colorama
import copy
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
    def __init__(self, allelic_segs_pickle, ref_fasta = None):
        # dataframe of allelic imbalance segmentation samples for each chromosome arm
        self.allelic_segs = pd.read_pickle(allelic_segs_pickle).dropna(0)
        self.allelic_segs = self.allelic_segs.loc[self.allelic_segs["results"].apply(lambda x : len(x.breakpoint_list)) > 0]

        # number of total segmentation samples
        self.n_samp = self.allelic_segs["results"].apply(lambda x : len(x.breakpoint_list)).min()
        self.ref_fasta = ref_fasta

        # DP run objects for each segmentation sample
        self.DP_runs = None

        # dataframe of SNPs
        self.SNPs = None

        # number of segmentation samples used for DP run
        self.N_seg_samps = None
        # number of DP samples per segmentation sample
        self.N_clust_samps = None

        # assignment of SNPs to DP clusters for each MCMC sample
        self.snps_to_clusters = None
        # phase correction of SNPs for each MCMC sample
        self.snps_to_phases = None

    def load_seg_samp(self, samp_idx):
        if samp_idx > self.n_samp:
            raise ValueError(f"Only {self.n_samp} MCMC samples were taken!")

        all_segs = []
        all_SNPs = []

        maj_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("MAJ_COUNT")
        min_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("MIN_COUNT")

        alt_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("ALT_COUNT")
        ref_idx = self.allelic_segs["results"].iloc[0].P.columns.get_loc("REF_COUNT")

        chunk_offset = 0
        for _, H in self.allelic_segs.dropna(subset = ["results"]).iterrows():
            r = copy.deepcopy(H["results"])

            # set phasing orientation back to original
            for st, en in r.F.intervals():
                # code excised from flip_hap
                x = r.P.iloc[st:en, maj_idx].copy()
                r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
                r.P.iloc[st:en, min_idx] = x

            # save SNPs for this chunk
            if self.SNPs is None:
                all_SNPs.append(pd.DataFrame({
                  "maj" : r.P["MAJ_COUNT"],
                  "min" : r.P["MIN_COUNT"],
                  # TODO: gpos should be computed earlier, so that that we don't need to pass ref_fasta here
                  "gpos" : seq.chrpos2gpos(r.P.loc[0, "chr"], r.P["pos"], ref = self.ref_fasta),
                  "allele" : r.P["allele_A"]
                }))

            # draw breakpoint, phasing, and SNP inclusion sample from segmentation MCMC trace
            bp_samp, pi_samp, inc_samp = (r.breakpoint_list[samp_idx], r.phase_interval_list[samp_idx] if r.phase_correct else None, r.include[samp_idx])
            # flip everything according to sample
            if r.phase_correct:
                for st, en in pi_samp.intervals():
                    x = r.P.iloc[st:en, maj_idx].copy()
                    r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
                    r.P.iloc[st:en, min_idx] = x

            bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]

            # get major/minor sums for each segment
            # also get {alt, ref} x {aidx, bidx}
            for st, en in bpl:
                all_segs.append([
                  st + chunk_offset, en + chunk_offset,                        # SNP index for seg
                  r.P.loc[st, "chr"], r.P.loc[st, "pos"], r.P.loc[en, "pos"],  # chromosomal position of seg
                  r._Piloc(st, en, min_idx, inc_samp).sum(),                   # min/maj counts
                  r._Piloc(st, en, maj_idx, inc_samp).sum(),

                  r._Piloc(st, en, alt_idx, inc_samp & r.P["aidx"]).sum(),     # allele A alt/ref
                  r._Piloc(st, en, ref_idx, inc_samp & r.P["aidx"]).sum(),
                  r._Piloc(st, en, alt_idx, inc_samp & ~r.P["aidx"]).sum(),    # allele B alt/ref
                  r._Piloc(st, en, ref_idx, inc_samp & ~r.P["aidx"]).sum()
                ])

            chunk_offset += len(r.P)

        # convert samples into dataframe
        S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj", "A_alt", "A_ref", "B_alt", "B_ref"])

        # convert chr-relative positions to absolute genomic coordinates
        S["start_gp"] = seq.chrpos2gpos(S["chr"], S["start"], ref = self.ref_fasta)
        S["end_gp"] = seq.chrpos2gpos(S["chr"], S["end"], ref = self.ref_fasta)

        # initial cluster assignments
        S["clust"] = -1 # initially, all segments are unassigned
        S.iloc[0, S.columns.get_loc("clust")] = 0 # first segment is assigned to cluster 0

        # initial phasing orientation
        S["flipped"] = False

        if self.SNPs is None:
            self.SNPs = pd.concat(all_SNPs, ignore_index = True)
            CI = s.beta.ppf([0.05, 0.5, 0.95], self.SNPs["min"].values[:, None] + 1, self.SNPs["maj"].values[:, None] + 1)
            self.SNPs[["f_CI_lo", "f", "f_CI_hi"]] = CI

        return S, self.SNPs

    # map trace of segment cluster assignments to the SNPs within
    @staticmethod
    def map_seg_clust_assignments_to_SNPs(segs_to_clusters, S):
        st_col = S.columns.get_loc("SNP_st")
        en_col = S.columns.get_loc("SNP_en")
        snps_to_clusters = np.zeros((segs_to_clusters.shape[0], S.iloc[-1, en_col] + 1), dtype = int)
        for i, seg_assign in enumerate(segs_to_clusters):
            for j, seg in enumerate(seg_assign):
                snps_to_clusters[i, S.iloc[j, st_col]:S.iloc[j, en_col]] = seg

        return snps_to_clusters

    @staticmethod
    def map_seg_phases_to_SNPs(phase, S):
        st_col = S.columns.get_loc("SNP_st")
        en_col = S.columns.get_loc("SNP_en")
        snps_to_phase = np.zeros((phase.shape[0], S.iloc[-1, en_col] + 1), dtype = int)
        for i, phase_orient in enumerate(phase):
            for j, ph in enumerate(phase_orient):
                snps_to_phase[i, S.iloc[j, st_col]:S.iloc[j, en_col]] = ph

        return snps_to_phase

    def run(self, N_seg_samps = 50, N_clust_samps = 5, seg_sample_idx = None):
        self.N_seg_samps = N_seg_samps if seg_sample_idx is None else 1
        self.N_clust_samps = N_clust_samps

        seg_sample_idx = np.random.choice(self.n_samp - 1, self.N_seg_samps, replace = False) if seg_sample_idx is None else [seg_sample_idx]
        S, SNPs = self.load_seg_samp(seg_sample_idx[0])
        N_SNPs = len(SNPs)
        
        self.snps_to_clusters = -1*np.ones((self.N_clust_samps*self.N_seg_samps, N_SNPs), dtype = np.int16)
        self.snps_to_phases = np.zeros((self.N_clust_samps*self.N_seg_samps, N_SNPs), dtype = bool)
        self.DP_likelihoods = np.zeros((self.N_clust_samps*self.N_seg_samps, 2))

        self.DP_runs = [None]*self.N_seg_samps

        clust_prior = sc.SortedDict()
        clust_count_prior = sc.SortedDict()
        n_iter_clust_exist = sc.SortedDict()
        cur_samp_iter = 0

        for n_it in range(self.N_seg_samps):
            if n_it > 0:
                S, SNPs = self.load_seg_samp(seg_sample_idx[n_it])

            # run clustering
            self.DP_runs[n_it] = DPinstance(S, clust_prior = clust_prior, clust_count_prior = clust_count_prior)
            segs_to_clusters, segs_to_phases = self.DP_runs[n_it].run(n_iter = self.N_clust_samps)

            # compute likelihoods for each clustering
            self.DP_likelihoods[self.N_clust_samps*n_it:self.N_clust_samps*(n_it + 1), :] = self.DP_runs[n_it].compute_overall_lik()

            # assign clusters to individual SNPs, to use as segment assignment prior for next DP iteration
            self.snps_to_clusters[self.N_clust_samps*n_it:self.N_clust_samps*(n_it + 1), :] = self.map_seg_clust_assignments_to_SNPs(segs_to_clusters, S)

            # assign phase orientations to individual SNPs
            self.snps_to_phases[self.N_clust_samps*n_it:self.N_clust_samps*(n_it + 1), :] = self.map_seg_phases_to_SNPs(segs_to_phases, S)

            # compute prior on cluster locations/counts
            max_clust_idx = segs_to_clusters.max()
            for seg_assignments, seg_phases in zip(segs_to_clusters, segs_to_phases):
                # reset phases
                S2 = S.copy()
                S2.loc[S2["flipped"], ["min", "maj"]] = S2.loc[S2["flipped"], ["min", "maj"]].values[:, ::-1]

                # match phases to current sample
                S2.loc[seg_phases, ["min", "maj"]] = S2.loc[seg_phases, ["min", "maj"]].values[:, ::-1]

                # minor/major counts for each cluster in this iteration
                S_a = npg.aggregate(seg_assignments, S2["min"], size = max_clust_idx + 1)
                S_b = npg.aggregate(seg_assignments, S2["maj"], size = max_clust_idx + 1)
                c = np.c_[S_a, S_b]

                # total numer of SNPs for each cluster in this iteration
                #N_c = npg.aggregate(seg_assignments, S2["SNP_en"] - S2["SNP_st"], size = max_clust_idx + 1)
                N_c = npg.aggregate(seg_assignments, 1, size = max_clust_idx + 1)

                # iteratively update priors
                next_clust_prior = sc.SortedDict(zip(np.flatnonzero(c.sum(1) > 0), c[c.sum(1) > 0]))
                next_clust_count_prior = sc.SortedDict(zip(np.flatnonzero(c.sum(1) > 0), N_c[N_c > 0]))

                for cl in np.unique(seg_assignments):
                    if cl in n_iter_clust_exist:
                        n_iter_clust_exist[cl] += 1
                    else:
                        n_iter_clust_exist[cl] = 1
                cur_samp_iter += 1

                for k, v in next_clust_prior.items():
                    nccp = next_clust_count_prior[k]
                    if k in clust_prior:
                        clust_prior[k] += (v - clust_prior[k])/n_iter_clust_exist[k]
                        clust_count_prior[k] += (nccp - clust_count_prior[k])/cur_samp_iter
                    else:
                        clust_prior[k] = v
                        clust_count_prior[k] = nccp/cur_samp_iter
                # for clusters that don't exist in this iteration, average counts with zero
                for k, v in clust_prior.items():
                    if k != -1 and k not in next_clust_prior:
                        clust_count_prior[k] -= clust_count_prior[k]/cur_samp_iter


            # remove improbable clusters from prior
            for kk in [k for k, v in clust_count_prior.items() if v < 1]:
                del clust_prior[kk]
                del clust_count_prior[kk]

        return self.snps_to_clusters, self.snps_to_phases, self.DP_likelihoods

    def visualize_segs(self, snps_to_clusters = None, f = None, n_vis_samp = None):
        f = plt.figure(figsize = [17.56, 5.67]) if f is None else f

        snps_to_clusters = snps_to_clusters if snps_to_clusters is not None else self.snps_to_clusters

        # plot all samples from DP
        if n_vis_samp is None:
            run_idx = np.r_[0:self.N_seg_samps]
            N_seg_samps = self.N_seg_samps

        # only plot up to n_vis_samp _segmentation samples_ from DP
        # (all DP samples for a given segmentation sample will be plotted)
        else:
            run_idx = np.random.choice(self.N_seg_samps, n_vis_samp, replace = False)
            N_seg_samps = n_vis_samp

        for d in [self.DP_runs[x] for x in run_idx]:
            d.visualize_adjacent_segs(f = f.number, n_samp = N_seg_samps*self.N_clust_samps)

    def visualize_clusts(self, snps_to_clusters = None, f = None, thick = False, nocolor = False, n_vis_samp = None):
        f = plt.figure(figsize = [17.56, 5.67]) if f is None else f

        snps_to_clusters = snps_to_clusters if snps_to_clusters is not None else self.snps_to_clusters

        # plot all samples from DP
        if n_vis_samp is None:
            run_idx = np.r_[0:self.N_seg_samps]
            N_seg_samps = self.N_seg_samps

        # only plot up to n_vis_samp _segmentation samples_ from DP
        # (all DP samples for a given segmentation sample will be plotted)
        else:
            run_idx = np.random.choice(self.N_seg_samps, n_vis_samp, replace = False)
            N_seg_samps = n_vis_samp

        for d in [self.DP_runs[x] for x in run_idx]:
            d.visualize_clusts(f = f.number, n_samp = N_seg_samps*self.N_clust_samps, thick = thick, nocolor = nocolor)

    def visualize_SNPs(self, snps_to_phases = None, color = True, f = None):
        snps_to_phases = snps_to_phases if snps_to_phases is not None else self.snps_to_phases
        ph_prob = snps_to_phases.mean(0)

        if color:
            rb = np.r_[np.c_[1, 0, 0], np.c_[0, 0, 1]]
        else:
            rb = np.full([2, 3], 0)

        logistic = lambda A, K, B, M, x : A + (K - A)/(1 + np.exp(-B*(x - M)))

        def scerrorbar(idx, rev = False, alpha = 1, show_CI = True):
            if rev:
                f = 1 - self.SNPs.loc[idx, "f"]
                eb_bot = self.SNPs.loc[idx, "f"] - self.SNPs.loc[idx, "f_CI_hi"]
                eb_top = self.SNPs.loc[idx, "f_CI_lo"] - self.SNPs.loc[idx, "f"]
            else:
                f = self.SNPs.loc[idx, "f"]
                eb_bot = self.SNPs.loc[idx, "f"] - self.SNPs.loc[idx, "f_CI_lo"]
                eb_top = self.SNPs.loc[idx, "f_CI_hi"] - self.SNPs.loc[idx, "f"]

            if show_CI:
                plt.errorbar(
                  x = self.SNPs.loc[idx, "gpos"],
                  y = f,
                  yerr = np.c_[
                    eb_bot,
                    eb_top
                  ].T,
                  fmt = 'none', ecolor = np.c_[rb[self.SNPs.loc[idx, "allele"]], (alpha if isinstance(alpha, np.ndarray) else alpha*np.ones(idx.sum()))**2]
                )

            plt.scatter(
              self.SNPs.loc[idx, "gpos"],
              f,
              color = rb[self.SNPs.loc[idx, "allele"]],
              marker = '.',
              s = 1,
              alpha = alpha if show_CI else alpha
            )

        default_alpha = logistic(A = 0.4, K = 0.01, B = 0.00001, M = 120000, x = len(self.SNPs))

        f = plt.figure(figsize = [17.56, 5.67]) if f is None else f
        scerrorbar(ph_prob == 0, alpha = default_alpha, show_CI = color)
        scerrorbar(ph_prob == 1, rev = True, alpha = default_alpha, show_CI = color)
        idx = (ph_prob > 0) & (ph_prob < 1)
        scerrorbar(idx, alpha = (1 - ph_prob[idx])*default_alpha, show_CI = color)
        scerrorbar(idx, rev = True, alpha = ph_prob[idx]*default_alpha, show_CI = color)

class DPinstance:
    def __init__(self, S, clust_prior = sc.SortedDict(), clust_count_prior = sc.SortedDict(), n_iter = 50, alpha = 1):
        self.S = S
        self.clust_prior = clust_prior.copy()
        self.clust_count_prior = clust_count_prior.copy()
        self.alpha = alpha

        self.mm_mat = self.S.loc[:, ["min", "maj"]].values.reshape(-1, order = "F") # numpy for speed
        self.ref_mat = self.S.loc[:, ["A_ref", "B_ref"]].values.reshape(-1, order = "F")
        self.alt_mat = self.S.loc[:, ["A_alt", "B_alt"]].values.reshape(-1, order = "F")

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
        self.clust_prior_liks = sc.SortedDict({ k : ss.betaln(v[0] + 1, v[1] + 1) for k, v in self.clust_prior.items()})
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
        flip = self.S.iloc[seg_idx, self.flip_col]
        flip_n = ~flip

        A_a = self.alt_mat[np.r_[seg_idx[flip_n], seg_idx[flip] + len(self.S)]].sum() + 1
        A_b = self.ref_mat[np.r_[seg_idx[flip_n], seg_idx[flip] + len(self.S)]].sum() + 1
        B_a = self.alt_mat[np.r_[seg_idx[flip], seg_idx[flip_n] + len(self.S)]].sum() + 1
        B_b = self.ref_mat[np.r_[seg_idx[flip], seg_idx[flip_n] + len(self.S)]].sum() + 1

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

        if targ_clust != - 1 and targ_clust == upstream_clust:
            J_a += U_a
            J_b += U_b
        else:
            SU_a += U_a
            SU_b += U_b
        if targ_clust != - 1 and targ_clust == downstream_clust:
            J_a += D_a
            J_b += D_b
        else:
            SD_a += D_a
            SD_b += D_b

        return ss.betaln(SU_a + 1, SU_b + 1) + ss.betaln(J_a + 1, J_b + 1) + ss.betaln(SD_a + 1, SD_b + 1)

    def compute_adj_prob(self, seg_idx):
        ## compute boundaries of adjacent segments

        # maj/min counts of contiguous upstream segments belonging to the same cluster
        st = seg_idx[0]
        U_A = 0
        U_B = 0
        U_cl = -1
        if st - 1 > 0:
            U_cl = self.clusts[st - 1]
            j = 1
            while st - j > 0 and self.clusts[st - j] != -1 and \
              self.clusts[st - j] == U_cl:
                U_A += self._Siat_ph(st - j, min = True)
                U_B += self._Siat_ph(st - j, min = False)

                j += 1

        # maj/min counts of contiguous downstream segments belonging to the same cluster
        en = seg_idx[-1]
        D_A = 0
        D_B = 0
        D_cl = -1
        if en + 1 < len(self.S):
            D_cl = self.clusts[en + 1]
            j = 1
            while en + j < len(self.S) - 1 and self.clusts[en + j] != -1 and \
              self.clusts[en + j] == D_cl:
                D_A += self._Siat_ph(en + j, min = True)
                D_B += self._Siat_ph(en + j, min = False)

                j += 1 

        # maj/min counts of segment(s) being moved
        S_A = self._Ssum_ph(seg_idx, min = True)
        S_B = self._Ssum_ph(seg_idx, min = False)

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
        probs = np.zeros([len(self.clust_sums), 2])
        probs_idx = np.zeros([len(self.clust_sums), 2]).astype(np.uint8)
        for k in self.clust_sums.keys():
            MLs_idx = np.r_[k == U_cl, k == D_cl]@np.r_[2, 1]
            probs[self.clust_sums.index(k), :] = MLs[:, MLs_idx]
            probs_idx[self.clust_sums.index(k), :] = np.r_[0, 4] + MLs_idx

        ## convert to conditional likelihoods, by scaling each likelihood by number of 
        ## cluster candidates with that segmentation configuration
        return probs - np.log(np.bincount(probs_idx.ravel())[probs_idx])

    def compute_adj_liks(self, seg_idx, cur_clust):
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
                S_a = self._Ssum_ph(np.r_[st:(en + 1)], min = True) # en + 1 because ordpairs is closed
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

            split_lik = ss.betaln(min_cs[:-1] + 1, maj_cs[:-1] + 1) + ss.betaln(min_csr[1:] + 1, maj_csr[1:] + 1)
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

            split_lik = ss.betaln(min_cs[:-1] + 1, maj_cs[:-1] + 1) + ss.betaln(min_csr[1:] + 1, maj_csr[1:] + 1)
            # split_lprob = split_lik - split_lik.max() - np.log(np.exp(split_lik - split_lik.max()).sum())

            start += split_lik.argmax() + 1
            spl.append(start - 1)

            if start > len(seg_idx) - 1 or split_lik.argmax() == 0:
                break

            i += 1

        bdy = np.unique(np.r_[0, spl, len(seg_idx)])
        bdy = np.c_[bdy[:-1], bdy[1:]]

        return bdy

    def compute_overall_lik(self, segs_to_clusters = None, phase_orientations = None):
        if segs_to_clusters is None:
            su, segs_to_clusters = self.get_unique_clust_idxs()
        else:
            su, segs_to_clusters = self.get_unique_clust_idxs(segs_to_clusters)
        if phase_orientations is None:
            phase_orientations = np.r_[self.phase_orientations]

        # account for unassigned clusters
        min_clust_idx = 1 if (su == -1).any() else 0

        max_clust_idx = segs_to_clusters.max() + 1

        liks = np.full(segs_to_clusters.shape[0], np.nan)

        for i, (cl_samp, ph_samp) in enumerate(zip(segs_to_clusters, phase_orientations)):
            ## overall clustering likelihood

            A1 = npg.aggregate(cl_samp[ph_samp], self.S.loc[ph_samp, "maj"], size = max_clust_idx)
            A2 = npg.aggregate(cl_samp[~ph_samp], self.S.loc[~ph_samp, "maj"], size = max_clust_idx)

            B1 = npg.aggregate(cl_samp[ph_samp], self.S.loc[ph_samp, "min"], size = max_clust_idx)
            B2 = npg.aggregate(cl_samp[~ph_samp], self.S.loc[~ph_samp, "min"], size = max_clust_idx)

            count_prior = np.bincount(cl_samp, minlength = max_clust_idx).astype(np.double)
            count_prior /= count_prior.sum()

            clust_lik = (ss.betaln(A1 + 1, B1 + 1) + ss.betaln(A2 + 1, B2 + 1))[min_clust_idx:].sum()
            # account for unassigned clusters, if present
            if min_clust_idx == 1:
                clust_lik += ss.betaln(self.S.loc[cl_samp == 0, "maj"] + 1, self.S.loc[cl_samp == 0, "min"] + 1).sum()

#            ## segmentation likelihood
#
#            # get segment boundaries
#            bdy = np.flatnonzero(np.r_[1, np.diff(cl_samp) != 0, 1])
#            bdy = np.c_[bdy[:-1], bdy[1:]]
#
#            # sum log-likelihoods of each segment
#            seg_lik = 0
#            for st, en in bdy:
#                A, B = S_ph.iloc[st:en, [self.min_col, self.maj_col]].sum()
#
## for when self.S is not modified
##               A = self.S["min"].iloc[st:en].loc[~ph_samp[st:en]].sum() + \
##                   self.S["maj"].iloc[st:en].loc[ph_samp[st:en]].sum()
##               B = self.S["maj"].iloc[st:en].loc[~ph_samp[st:en]].sum() + \
##                   self.S["min"].iloc[st:en].loc[ph_samp[st:en]].sum()
#
#                seg_lik += ss.betaln(A + 1, B + 1)

            liks[i] = clust_lik

        return liks

    def run(self, n_iter = 50):
        #
        # assign segments to likeliest prior component {{{

        if len(self.clust_prior) > 1:
            for seg_idx in range(len(self.S)):
                seg_idx = np.r_[seg_idx] 

                # compute probability that segment belongs to each cluster prior element
                S_a = self._Siat_ph(seg_idx[0], min = True)
                S_b = self._Siat_ph(seg_idx[0], min = False)
                P_a = self.clust_prior_mat[1:, 0]
                P_b = self.clust_prior_mat[1:, 1]

                # prior likelihood ratios for both phase orientations
                P_l = np.c_[
                  ss.betaln(S_a + P_a + 1, S_b + P_b + 1) - (ss.betaln(S_a + 1, S_b + 1) + ss.betaln(P_a + 1, P_b + 1)),
                  ss.betaln(S_b + P_a + 1, S_a + P_b + 1) - (ss.betaln(S_b + 1, S_a + 1) + ss.betaln(P_a + 1, P_b + 1)),
                ]

                # get count prior
                ccp = np.c_[[v for k, v in self.clust_count_prior.items() if k != -1]]

                # posterior numerator
                num = P_l + np.log(ccp)
                num -= num.max()

                # probabilistically choose a cluster
                probs = np.exp(num)/np.exp(num).sum()
                idx = np.tile(np.r_[self.clust_prior.keys()][1:], [2, 1]).T*[1, -1]
                choice = np.random.choice(
                  idx.ravel(),
                  p = probs.ravel()
                )

                # rephase
                if choice < 0:
                    self.S.iloc[seg_idx, self.flip_col] = ~self.S.iloc[seg_idx, self.flip_col]
                    choice = -choice

                self.S.iloc[seg_idx, self.clust_col] = choice

        # }}}

        #
        # initialize cluster tracking hash tables
        self.clust_counts = sc.SortedDict(self.S["clust"].value_counts().drop(-1, errors = "ignore"))
        # for the first round of clustering, this is { 1 : 1 }

        x = self.S.groupby(["clust", "flipped"])[["min", "maj"]].sum()
        if (x.droplevel(0).index == True).any():
            x.loc[(slice(None), True), ["min", "maj"]] = x.loc[(slice(None), True), ["maj", "min"]].values
        self.clust_sums = sc.SortedDict({
          **{ k : np.r_[v["min"], v["maj"]] for k, v in x.groupby(level = "clust").sum().to_dict(orient = "index").items() },
          **{-1 : np.r_[0, 0]}
        })
        # for the first round, this is { -1 : np.r_[0, 0], 0 : np.r_[S[0, "min"], S[0, "maj"]] }

        self.clust_members = sc.SortedDict({ k : set(v) for k, v in self.S.groupby("clust").groups.items() if k != -1 })
        # for the first round, this is { 1 : {0} }

        unassigned_segs = sc.SortedList(self.S.index[self.S["clust"] == -1])

        # store this as numpy for speed
        self.clusts = self.S["clust"].values

        max_clust_idx = np.max(self.clust_members.keys() | self.clust_prior.keys() if self.clust_prior is not None else {})

        # containers for saving the MCMC trace
        self.segs_to_clusters = []
        self.phase_orientations = []

        burned_in = False
        all_assigned = False
        all_touched = False
        seg_touch_idx = np.zeros(len(self.S), dtype = bool)

        # likelihood trace
        self.lik_tmp = []
        self.post = 0

        n_it = 0
        n_it_last = 0
        while len(self.segs_to_clusters) < n_iter:
            if not n_it % 1000:
                print(self.S["clust"].value_counts().drop([-1, 0], errors = "ignore").value_counts().sort_index())
                print("n unassigned: {}".format((self.S["clust"] == -1).sum()))

            # poll every 100 iterations for burnin status
            if not n_it % 100:
                self.lik_tmp.append(self.post)
                if not all_assigned and len(unassigned_segs) == 0:
                    all_assigned = True
                if not burned_in and all_assigned:
                    # 1. have >90% of segments been adjacency corrected?
                    # print(seg_touch_idx.mean())
                    if seg_touch_idx.mean() > 0.9:
                        all_touched = True

                    # 2. if >90% of segments have been adjacency corrected, check for burnin
                    # does the smoothed derivative of the posterior numerator go below zero? this would indicate that we've solidly reached an optimum
                    # TODO: make this check more efficient?
                    if all_touched and (np.convolve(np.diff(self.lik_tmp), np.ones(50)/50, mode = "same") < 0).sum() > 2:
                        burned_in = True
                        breakpoint()

            #
            # pick either a segment or a cluster at random (50:50 prob.)
            move_clust = False

            # pick a segment at random
            if np.random.rand() < 0.5:
            #if np.random.rand() < 1:
                # bias picking unassigned segments if >90% of segments have been assigned
                if len(unassigned_segs) > 0 and len(unassigned_segs)/len(self.S) < 0.1 and np.random.rand() < 0.5:
                    seg_idx = sc.SortedSet({np.random.choice(unassigned_segs)})
                else:
                    seg_idx = sc.SortedSet({np.random.choice(len(self.S))})

                cur_clust = int(self.clusts[seg_idx])

                # expand segment to include all adjacent segments in the same cluster,
                # if it has already been assigned to a cluster
                if cur_clust >= 0 and np.random.rand() < 0.5:
                    si = seg_idx[0]

                    j = 1
                    while si - j > 0 and self.clusts[si - j] == cur_clust:
                        seg_idx.add(si - j)
                        j += 1
                    j = 1
                    while si + j < len(self.S) and self.clusts[si + j] == cur_clust:
                        seg_idx.add(si + j)
                        j += 1

                    # if we've expanded to include a large fraction (>10%) of segments 
                    # in this cluster, cluster indexing might become inconsistent.
                    # skip this iteration
#                    if len(seg_idx) >= 0.1*self.clust_counts[cur_clust]:
#                        breakpoint()
#                        n_it += 1
#                        continue

                # propose splitting out a contiguous interval of segments within the current cluster
                split_clust = False
                if all_assigned and np.random.rand() < 0.1:
                    # TODO: if we use cur_clust, this will be biased towards larger clusters. is this desireable?
                    clust_segs = np.sort(np.r_[list(self.clust_members[cur_clust])])
                    split_bdy = self.compute_cluster_splitpoints(clust_segs)

                    A_tot, B_tot = self.clust_sums[cur_clust]

                    lik0 = ss.betaln(A_tot + 1, B_tot + 1)

                    liks = np.zeros(len(split_bdy) + 1)
                    liks[-1] = lik0 # don't split at all

                    # likelihood ratios for splitting each region into a new cluster
                    for i, (st, en) in enumerate(split_bdy):
                        A = self._Ssum_ph(clust_segs[st:en], min = True)
                        B = self._Ssum_ph(clust_segs[st:en], min = False)

                        liks[i] = ss.betaln(A_tot - A + 1, B_tot - B + 1) + ss.betaln(A + 1, B + 1)

                    # pick a region to split
                    split_idx = np.random.choice(
                      len(split_bdy) + 1,
                      p = np.exp(liks - liks.max())/np.exp(liks - liks.max()).sum()
                    )

                    # don't split at all
                    if split_idx == len(split_bdy):
                        n_it += 1
                        continue

                    # seg_idx == segments to propose to split off
                    seg_idx = clust_segs[slice(*split_bdy[split_idx])]

                    split_clust = True

                seg_idx = np.r_[list(seg_idx)]

                n_move = len(seg_idx)

                # if segment was already assigned to a cluster, unassign it
                if cur_clust >= 0:
                    self.clust_counts[cur_clust] -= n_move
                    if self.clust_counts[cur_clust] == 0:
                        del self.clust_counts[cur_clust]
                        del self.clust_sums[cur_clust]
                        del self.clust_members[cur_clust]
                    else:
                        self.clust_sums[cur_clust] -= np.r_[self._Ssum_ph(seg_idx, min = True), self._Ssum_ph(seg_idx, min = False)]
                        self.clust_members[cur_clust] -= set(seg_idx)

                    unassigned_segs.update(seg_idx)
                    self.clusts[seg_idx] = -1

            # pick a cluster at random
            else:
                # it only makes sense to try joining two clusters if there are at least two of them!
                if len(self.clust_counts) < 2:
                    n_it += 1
                    continue

                cl_idx = np.random.choice(self.clust_counts.keys())
                seg_idx = np.r_[list(self.clust_members[cl_idx])]
                n_move = len(seg_idx)
                cur_clust = -1 # only applicable for individual segments, so we set to -1 here
                               # (this is so that subsequent references to clust_sums[cur_clust]
                               # will return (0, 0))

                # unassign all segments within this cluster
                # (it will either be joined with a new cluster, or remade again into its own cluster)
                del self.clust_counts[cl_idx]
                del self.clust_sums[cl_idx]
                del self.clust_members[cl_idx]
                unassigned_segs.update(seg_idx)
                self.clusts[seg_idx] = -1

                move_clust = True

            if not all_assigned:
                seg_touch_idx[seg_idx] += 1

            #
            # perform phase correction on segment/cluster
            # flip min/maj with probability that alleles are oriented the "wrong" way
            rephase_prob = self.compute_rephase_prob(seg_idx)

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
            C = ss.betaln(C_ab[:, 0] + 1, C_ab[:, 1] + 1)
            # A is likelihood cluster B is part of, minus B
            #A = ss.betaln(A_a + 1, A_b + 1)
            # B+C is likelihood of target cluster post-join, with both phase orientations
            BC = ss.betaln(C_ab[:, [0]] + np.c_[B_a, B_b] + 1, C_ab[:, [1]] + np.c_[B_b, B_a] + 1)

            MLs = BC - C[:, None] + np.log(np.maximum(1e-300, np.r_[1 - rephase_prob, rephase_prob]))

            #     L(join)           L(split)
            #MLs = A + BC + adj_BC - (AB + C + adj_AB)
            # TODO: remove extraneous calculations (e.g. adj_AB, AB, A);
            #       likelihood simplifies to this in the prior:
            #MLs = adj_BC + BC - C

            # if we are moving multiple contiguous segments assigned to the same
            # cluster, do not allow them to create a new cluster. this helps keep
            # cluster indices consistent
            # TODO: if we don't care about keeping indices consistent, then we can probably remove this line
            if n_move > 1 and not move_clust:
                MLs[self.clust_sums.index(-1)] = -np.inf

            #
            # priors

            ## prior on previous cluster fractions

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
                ]

                # prior marginal likelihoods for both phase orientations
                prior_MLs = ss.betaln( # prior clusters + segment
                  np.c_[self.clust_prior_mat[prior_idx, 0]] + np.c_[B_a, B_b] + 1,
                  np.c_[self.clust_prior_mat[prior_idx, 1]] + np.c_[B_b, B_a] + 1
                ) \
                - np.c_[ss.betaln(B_a + 1, B_b + 1) + np.r_[np.r_[self.clust_prior_liks.values()][prior_idx]]] # prior clusters, segment

                clust_prior_p = np.maximum(np.exp(prior_MLs - prior_MLs.max())/np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                # expand MLs to account for multiple new clusters
                MLs = np.r_[np.full([len(prior_diff), 2], MLs[0]), MLs[1:, :]]
                
            ## DP prior based on clusters sizes
            # DP alpha factor is split proportionally between prior_diff and -1 (brand new cluster)
            ccp = np.r_[[self.clust_count_prior[x] for x in prior_diff]]
            count_prior = np.r_[self.clust_count_prior[-1]*ccp/ccp.sum(), self.clust_counts.values()]
            count_prior /= count_prior.sum()

            # adjacent segment prior

            log_adj_prior = 0
            if not move_clust and not split_clust: # or (all_assigned and move_clust and np.random.rand() < 0.01):
                log_adj_prior = self.compute_adj_prob(seg_idx)
                if all_assigned:
                    seg_touch_idx[seg_idx] = True

            # choose to join a cluster or make a new one (choice_idx = 0)
            num = MLs + np.log(count_prior[:, None]) + np.log(clust_prior_p) + log_adj_prior
            choice_p = np.exp(num - num.max())/np.exp(num - num.max()).sum()
            # row major indexing: choice_idx//2 = cluster index, choice_idx & 1 = rephase true
            choice_idx = np.random.choice(
              np.r_[0:np.prod(choice_p.shape)],
              p = choice_p.ravel()
            )
            # -1 = brand new, -2, -3, ... = -(prior clust index) - 2
            choice = np.r_[-np.r_[prior_diff] - 2, self.clust_counts.keys()][choice_idx//2]

            # compute posterior delta between previous and current state
            post_delta = num.ravel()[choice_idx] - \
              num[self.clust_sums.index(cur_clust if cur_clust in self.clust_sums else -1), 0]
            self.post += post_delta

            # save rephasing status
            if choice_idx & 1:
                self.S.iloc[seg_idx, self.flip_col] = ~self.S.iloc[seg_idx, self.flip_col]

            # create new cluster
            if choice < 0:
                # if we are moving an entire cluster, give it the same index it used to have
                # otherwise, cluster indices will be inconsistent
                if move_clust:
                    new_clust_idx = cl_idx
                elif choice == -1: # totally new cluster
                    max_clust_idx += 1
                    new_clust_idx = max_clust_idx
                else: # match index of cluster in prior
                    new_clust_idx = -choice - 2

                self.clust_counts[new_clust_idx] = n_move
                self.S.iloc[seg_idx, self.clust_col] = new_clust_idx
                self.clusts[seg_idx] = new_clust_idx

                self.clust_sums[new_clust_idx] = np.r_[B_a, B_b] if not choice_idx & 1 else np.r_[B_b, B_a]
                self.clust_members[new_clust_idx] = set(seg_idx)

            # join existing cluster
            else:
                # if we are combining two clusters, take the index of the bigger one
                # this helps to keep cluster indices consistent
                if move_clust and self.clust_counts[choice] < n_move:
                    self.clust_counts[cl_idx] = self.clust_counts[choice]
                    self.clust_sums[cl_idx] = self.clust_sums[choice]
                    self.clust_members[cl_idx] = self.clust_members[choice]
                    self.S.iloc[np.flatnonzero(self.S["clust"] == choice), self.clust_col] = cl_idx
                    del self.clust_counts[choice]
                    del self.clust_sums[choice]
                    del self.clust_members[choice]
                    choice = cl_idx

                self.clust_counts[choice] += n_move 
                self.clust_sums[choice] += np.r_[B_a, B_b] if not choice_idx & 1 else np.r_[B_b, B_a]
                self.S.iloc[seg_idx, self.clust_col] = choice
                self.clusts[seg_idx] = choice

                self.clust_members[choice].update(set(seg_idx))

            for si in seg_idx:
                unassigned_segs.discard(si)

            # track global state of cluster assignments
            # on average, each segment will have been reassigned every n_seg/(n_clust/2) iterations
            if burned_in and n_it - n_it_last > len(self.S)/(len(self.clust_counts)*2):
                self.segs_to_clusters.append(self.S["clust"].copy())
                self.phase_orientations.append(self.S["flipped"].copy())
                n_it_last = n_it

            n_it += 1

        return np.r_[self.segs_to_clusters], np.r_[self.phase_orientations]

    #_colors = mpl.cm.get_cmap("tab10").colors
    _colors = ((np.c_[1:7] & np.r_[4, 2, 1]) > 0).astype(int)
#   _colors = np.r_[np.c_[87, 182, 55],
#   np.c_[253, 245, 81],
#   np.c_[238, 109, 45],
#   np.c_[204, 43, 30],
#   np.c_[221, 50, 132],
#   np.c_[0, 23, 204],
#   np.c_[75, 172, 227]]/255

    def get_unique_clust_idxs(self, segs_to_clusters = None):
        if segs_to_clusters is None:
            segs_to_clusters = np.r_[self.segs_to_clusters]
        s2cu, s2cu_j = np.unique(segs_to_clusters, return_inverse = True)
        return s2cu, s2cu_j.reshape(segs_to_clusters.shape)

    def get_colors(self):
        s2cu, s2cu_j = self.get_unique_clust_idxs()

        seg_terr = self.S["end_gp"] - self.S["start_gp"]
        tot_terr = np.zeros(len(s2cu))
        for r in s2cu_j:
           tot_terr += npg.aggregate(r, seg_terr, size = len(tot_terr))

        si = np.argsort(tot_terr)[::-1]
        terr_cs = np.cumsum(tot_terr[si])/tot_terr.sum()

        colors_to_use = np.array([mpl.cm.get_cmap("gist_rainbow")(x) for x in np.linspace(0, 1, (terr_cs < 0.99).sum())])
        colors = np.zeros([len(s2cu), 4])
        n_distinct = colors_to_use.shape[0] 
        colors[si[:n_distinct], :] = colors_to_use
        colors[si[n_distinct:], :] = colors_to_use[:(len(si) - n_distinct), :]

    def visualize_segs(self):
        plt.figure()
        ax = plt.gca()
        ax.set_xlim([0, self.S["end_gp"].max()])
        ax.set_ylim([0, 1])

        colors = self.get_colors()
        s2cu, s2cu_j = self.get_unique_clust_idxs()

        n_samp = len(self.segs_to_clusters)

        for s2c, s2ph in zip(s2cu_j, self.phase_orientations):
            # rephase segments according to phase orientation sample
            S_ph = self.S.copy()
            flip_idx = np.flatnonzero(s2ph != S_ph["flipped"])
            S_ph.iloc[flip_idx, [self.min_col, self.maj_col]] = S_ph.iloc[flip_idx, [self.maj_col, self.min_col]]

            for i, r in enumerate(S_ph.itertuples()):
                ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], r.min + 1, r.maj + 1)
                ax.add_patch(mpl.patches.Rectangle((r.start_gp, ci_lo), r.end_gp - r.start_gp, ci_hi - ci_lo, facecolor = colors[s2c[i] % len(colors)], fill = True, alpha = 1/n_samp, zorder = 1000))

    def visualize_adjacent_segs(self, f = None, n_samp = None):
        plt.figure(num = f, figsize = [17.56, 5.67])
        ax = plt.gca()
        ax.set_xlim([0, self.S["end_gp"].max()])
        ax.set_ylim([0, 1])

        colors = self.get_colors()
        s2cu, s2cu_j = self.get_unique_clust_idxs()

        n_samp = len(self.segs_to_clusters) if n_samp is None else n_samp

        for s2c, s2ph in zip(s2cu_j, self.phase_orientations):
            # rephase segments according to phase orientation sample
            S_ph = self.S.copy()
            flip_idx = np.flatnonzero(s2ph != S_ph["flipped"])
            S_ph.iloc[flip_idx, [self.min_col, self.maj_col]] = S_ph.iloc[flip_idx, [self.maj_col, self.min_col]]

            bdy = np.flatnonzero(np.r_[1, np.diff(s2c) != 0, 1])
            bdy = np.c_[bdy[:-1], bdy[1:]]

#            s2c_nz = s2c.copy()
#            zidx = np.flatnonzero(s2c[bdy[:, 0]] == 0)
#            for z in zidx:
#                s2c_nz[bdy[z, 0]:bdy[z, 1]] = s2c_nz[bdy[z - 1, 0]]
#            bdy_nz = np.flatnonzero(np.r_[1, np.diff(s2c_nz) != 0, 1])
#            bdy_nz = np.c_[bdy_nz[:-1], bdy_nz[1:]]

            for st, en in bdy:
                ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], S_ph.iloc[st:en, self.min_col].sum() + 1, S_ph.iloc[st:en, self.maj_col].sum() + 1)
                ax.add_patch(mpl.patches.Rectangle((S_ph.iloc[st]["start_gp"], ci_lo), S_ph.iloc[en - 1]["end_gp"] - S_ph.iloc[st]["start_gp"], np.maximum(0, ci_hi - ci_lo), facecolor = colors[s2c[st] % len(colors)], fill = True, alpha = 1/n_samp, zorder = 1000))

    def visualize_clusts(self, f = None, n_samp = None, thick = False, nocolor = False):
        plt.figure(num = f, figsize = [17.56, 5.67])
        ax = plt.gca()
        ax.set_xlim([0, self.S["end_gp"].max()])
        ax.set_ylim([0, 1])

        colors = self.get_colors()
        s2cu, s2cu_j = self.get_unique_clust_idxs()

        n_samp = len(self.segs_to_clusters) if n_samp is None else n_samp

        for s2c, s2ph in zip(s2cu_j, self.phase_orientations):
            # rephase segments according to phase orientation sample
            S_ph = self.S.copy()
            flip_idx = np.flatnonzero(s2ph != S_ph["flipped"])
            S_ph.iloc[flip_idx, [self.min_col, self.maj_col]] = S_ph.iloc[flip_idx, [self.maj_col, self.min_col]]

            # get overall cluster sums
            clust_min = npg.aggregate(s2c, S_ph["min"])
            clust_maj = npg.aggregate(s2c, S_ph["maj"])
            CIs = s.beta.ppf([0.05, 0.5, 0.95], clust_min[:, None] + 1, clust_maj[:, None] + 1)

            # get boundaries of contiguous segments
            bdy = np.flatnonzero(np.r_[1, np.diff(s2c) != 0, 1])
            bdy = np.c_[bdy[:-1], bdy[1:]]

#            s2c_nz = s2c.copy()
#            zidx = np.flatnonzero(s2c[bdy[:, 0]] == 0)
#            for z in zidx:
#                s2c_nz[bdy[z, 0]:bdy[z, 1]] = s2c_nz[bdy[z - 1, 0]]
#            bdy_nz = np.flatnonzero(np.r_[1, np.diff(s2c_nz) != 0, 1])
#            bdy_nz = np.c_[bdy_nz[:-1], bdy_nz[1:]]

            for st, en in bdy:
                if thick:
                    b = CIs[s2c[st], 1] - 0.01
                    t = CIs[s2c[st], 1] + 0.01
                else:
                    color = colors[s2c[st] % len(colors)]
                    b = CIs[s2c[st], 0]
                    t = CIs[s2c[st], 2]

                if nocolor:
                    color = [0, 1, 0]
                else:
                    color = colors[s2c[st] % len(colors)]

                ax.add_patch(mpl.patches.Rectangle(
                  xy = (S_ph.iloc[st]["start_gp"], b),
                  width = S_ph.iloc[en - 1]["end_gp"] - S_ph.iloc[st]["start_gp"],
                  height = t - b,
                  facecolor = color,
                  fill = True,
                  alpha = 1/n_samp,
                  zorder = 1000)
                )
