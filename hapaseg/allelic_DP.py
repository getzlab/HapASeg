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
        self.allelic_segs = pd.read_pickle(allelic_segs_pickle)
        self.n_samp = self.allelic_segs["results"].apply(lambda x : len(x.breakpoint_list)).min()
        self.ref_fasta = ref_fasta

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
            all_SNPs.append(pd.DataFrame({ "maj" : r.P["MAJ_COUNT"], "min" : r.P["MIN_COUNT"], "gpos" : seq.chrpos2gpos(r.P.loc[0, "chr"], r.P["pos"], ref = self.ref_fasta) }))

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
        S.iloc[0, S.columns.get_loc("clust")] = 1 # first segment is assigned to cluster 1

        # initial phasing orientation
        S["flipped"] = False

        return S, pd.concat(all_SNPs, ignore_index = True)

    def run_DP(self, S, clust_prior = sc.SortedDict(), clust_count_prior = sc.SortedDict(), n_iter = 50):
        #
        # define column indices
        clust_col = S.columns.get_loc("clust")
        min_col = S.columns.get_loc("min")
        maj_col = S.columns.get_loc("maj")
        aalt_col = S.columns.get_loc("A_alt")
        aref_col = S.columns.get_loc("A_ref")
        balt_col = S.columns.get_loc("B_alt")
        bref_col = S.columns.get_loc("B_ref")
        flip_col = S.columns.get_loc("flipped")

        #
        # initialize priors

        # store likelihoods for each cluster in the prior (from previous iterations)
        clust_prior[-1] = np.r_[0, 0]
        clust_prior_liks = sc.SortedDict({ k : ss.betaln(v[0] + 1, v[1] + 1) for k, v in clust_prior.items()})
        clust_prior_mat = np.r_[clust_prior.values()]

        clust_count_prior[-1] = 0.1 # DP alpha factor, i.e. relative probability of opening new cluster (TODO: make specifiable)
        clust_count_prior[0] = 0.1 # relative probability of sending a cluster to the garbage

        #
        # assign segments to likeliest prior component {{{

        if len(clust_prior) > 1:
            for seg_idx in range(len(S)):
                seg_idx = np.r_[seg_idx]
                # rephase segment
                x = s.beta.rvs(S.iloc[seg_idx, aalt_col].sum() + 1, S.iloc[seg_idx, aref_col].sum() + 1, size = [len(seg_idx), 30])
                y = s.beta.rvs(S.iloc[seg_idx, balt_col].sum() + 1, S.iloc[seg_idx, bref_col].sum() + 1, size = [len(seg_idx), 30])
                if np.random.rand() < (x > y).mean():
                    S.iloc[seg_idx, [min_col, maj_col]] = S.iloc[seg_idx, [min_col, maj_col]].values[:, ::-1]
                    S.iloc[seg_idx, [aalt_col, balt_col]] = S.iloc[seg_idx, [aalt_col, balt_col]].values[:, ::-1]
                    S.iloc[seg_idx, [aref_col, bref_col]] = S.iloc[seg_idx, [aref_col, bref_col]].values[:, ::-1]
                    S.iloc[seg_idx, flip_col] = ~S.iloc[seg_idx, flip_col]

                # compute probability that segment belongs to each cluster prior element
                S_a = S.iloc[seg_idx[0], min_col]
                S_b = S.iloc[seg_idx[0], maj_col]
                P_a = clust_prior_mat[1:, 0]
                P_b = clust_prior_mat[1:, 1]
                P_l = ss.betaln(S_a + P_a + 1, S_b + P_b + 1) - (ss.betaln(S_a + 1, S_b + 1) + ss.betaln(P_a + 1, P_b + 1))

                # probabilistically assign
                ccp = np.r_[[v for k, v in clust_count_prior.items() if k != -1 and k != 0]]
                S.iloc[seg_idx, clust_col] = np.random.choice(
                  np.r_[clust_prior.keys()][1:], 
                  p = np.exp(P_l)*ccp/(np.exp(P_l)*ccp).sum()
                )

            # TODO: send segments to garbage

        # }}}

        def SJliks(targ_clust, upstream_clust, downstream_clust, J_a, J_b, U_a, U_b, D_a, D_b):
#            if st == en:
#                J_a = S.iat[st, min_col].sum()
#                J_b = S.iat[st, maj_col].sum()
#            else:
#                J_a = S.iloc[st:(en + 1), min_col].sum()
#                J_b = S.iloc[st:(en + 1), maj_col].sum()
            SU_a = SU_b = SD_a = SD_b = 0
            # if target segments are being moved to the garbage, it is equivalent to making them their own segment, and joining the upstream and downstream segments
            if targ_clust == 0:
                SU_a = J_a
                SU_b = J_b
                J_a = 0
                J_b = 0

            if targ_clust != - 1 and st - 1 > 0 and (targ_clust == upstream_clust or targ_clust == 0):
                J_a += U_a
                J_b += U_b
            else:
                SU_a += U_a
                SU_b += U_b
            if targ_clust != - 1 and en + 1 < len(S) and (targ_clust == downstream_clust or targ_clust == 0):
                J_a += D_a
                J_b += D_b
            else:
                SD_a += D_a
                SD_b += D_b

            return ss.betaln(SU_a + 1, SU_b + 1) + ss.betaln(J_a + 1, J_b + 1) + ss.betaln(SD_a + 1, SD_b + 1)

        #
        # initialize cluster tracking hash tables
        clust_counts = sc.SortedDict(S["clust"].value_counts().drop([-1, 0], errors = "ignore"))
        # for the first round of clustering, this is { 1 : 1 }
        clust_sums = sc.SortedDict({
          **{ k : np.r_[v["min"], v["maj"]] for k, v in S.groupby("clust")[["min", "maj"]].sum().to_dict(orient = "index").items() },
          **{-1 : np.r_[0, 0], 0 : np.r_[0, 0]}
        })
        # for the first round, this is { -1/0 : np.r_[0, 0], 1 : np.r_[S[0, "min"], S[0, "maj"]] }
        clust_members = sc.SortedDict({ k : set(v) for k, v in S.groupby("clust").groups.items() if k != -1 and k != 0 })
        # for the first round, this is { 1 : {0} }
        unassigned_segs = sc.SortedList(S.index[S["clust"] == -1])

        # store this as numpy for speed
        clusts = S["clust"].values

        max_clust_idx = np.max(clust_members.keys() | clust_prior.keys() if clust_prior is not None else {})

        # containers for saving the MCMC trace
        segs_to_clusters = []
        phase_orientations = []

        burned_in = False

        n_it = 0
        n_it_last = 0
        while len(segs_to_clusters) < n_iter:
            if not n_it % 1000:
                print(S["clust"].value_counts().drop([-1, 0], errors = "ignore").value_counts().sort_index())
                print("n unassigned: {}".format((S["clust"] == -1).sum()))
                print("n garbage: {}".format((S["clust"] == 0).sum()))

            # we are burned in once all segments are assigned to a cluster
            if not burned_in and (S["clust"] != -1).all():
                burned_in = True

            #
            # pick either a segment or a cluster at random (50:50 prob.)
            move_clust = False

            # pick a segment at random
            if np.random.rand() < 0.5:
            #if np.random.rand() < 1:
                # bias picking unassigned segments if >90% of segments have been assigned
                if len(unassigned_segs) > 0 and len(unassigned_segs)/len(S) < 0.1 and np.random.rand() < 0.5:
                    seg_idx = sc.SortedSet({np.random.choice(unassigned_segs)})
                else:
                    seg_idx = sc.SortedSet({np.random.choice(len(S))})

                cur_clust = int(clusts[seg_idx])

                # expand segment to include all adjacent segments in the same cluster
                if np.random.rand() < 0.5:
                    si = seg_idx[0]

                    j = 1
                    while cur_clust != -1 and si - j > 0 and \
                      clusts[si - j] == cur_clust:
                        seg_idx.add(si - j)
                        j += 1
                    j = 1
                    while cur_clust != -1 and si + j < len(S) and \
                      clusts[si + j] == cur_clust:
                        seg_idx.add(si + j)
                        j += 1

                seg_idx = np.r_[list(seg_idx)]

                n_move = len(seg_idx)

                # if segment was already assigned to a cluster, unassign it
                if cur_clust > 0:
                    clust_counts[cur_clust] -= n_move
                    if clust_counts[cur_clust] == 0:
                        del clust_counts[cur_clust]
                        del clust_sums[cur_clust]
                        del clust_members[cur_clust]
                    else:
                        clust_sums[cur_clust] -= np.r_[S.iloc[seg_idx, min_col].sum(), S.iloc[seg_idx, maj_col].sum()]
                        clust_members[cur_clust] -= set(seg_idx)

                    unassigned_segs.update(seg_idx)
                    clusts[seg_idx] = -1

            # pick a cluster at random
            else:
                # it only makes sense to try joining two clusters if there are at least two of them!
                if len(clust_counts) < 2:
                    n_it += 1
                    continue

                cl_idx = np.random.choice(clust_counts.keys())
                seg_idx = np.r_[list(clust_members[cl_idx])]
                n_move = len(seg_idx)
                cur_clust = -1 # only applicable for individual segments, so we set to -1 here
                               # (this is so that subsequent references to clust_sums[cur_clust]
                               # will return (0, 0))

                # unassign all segments within this cluster
                # (it will either be joined with a new cluster, or remade again into its own cluster)
                del clust_counts[cl_idx]
                del clust_sums[cl_idx]
                del clust_members[cl_idx]
                unassigned_segs.update(seg_idx)
                clusts[seg_idx] = -1

                move_clust = True

            #
            # perform phase correction on segment/cluster
            # flip min/maj with probability that alleles are oriented the "wrong" way
            x = s.beta.rvs(S.iloc[seg_idx, aalt_col].sum() + 1, S.iloc[seg_idx, aref_col].sum() + 1, size = [n_move, 30])
            y = s.beta.rvs(S.iloc[seg_idx, balt_col].sum() + 1, S.iloc[seg_idx, bref_col].sum() + 1, size = [n_move, 30])
            if np.random.rand() < (x > y).mean():
                S.iloc[seg_idx, [min_col, maj_col]] = S.iloc[seg_idx, [min_col, maj_col]].values[:, ::-1]
                S.iloc[seg_idx, [aalt_col, balt_col]] = S.iloc[seg_idx, [aalt_col, balt_col]].values[:, ::-1]
                S.iloc[seg_idx, [aref_col, bref_col]] = S.iloc[seg_idx, [aref_col, bref_col]].values[:, ::-1]
                S.iloc[seg_idx, flip_col] = ~S.iloc[seg_idx, flip_col]

            #
            # choose to join a cluster or make a new one
            # probabilities determined by similarity of segment/cluster to existing ones

            # B is segment/cluster to move
            # A is cluster B is currently part of
            # C is all possible clusters to move to
            A_a = clust_sums[cur_clust][0] if cur_clust in clust_sums else 0
            A_b = clust_sums[cur_clust][1] if cur_clust in clust_sums else 0
            B_a = S.iloc[seg_idx, min_col].sum() # TODO: slow if seg_idx contains many SNPs
            B_b = S.iloc[seg_idx, maj_col].sum()
            C_ab = np.r_[clust_sums.values()] # first terms: (-1) = make new cluster, (0) = garbage cluster
            #C_ab = np.r_[[v for k, v in clust_sums.items() if k != cur_clust or cur_clust == -1]] # if we don't want to explicitly propose letting B rejoin cur_clust

            #
            # adjacent segment likelihoods

            adj_AB = 0
            adj_BC = np.zeros(len(clust_sums))

            if not move_clust:
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
                        # skip over adjacent segments that are in the garbage;
                        # we only care about adjacent segments actually assigned to clusters
                        j = 1
                        while st - j > 0 and clusts[st - j] == 0:
                            j += 1

                        U_cl = clusts[st - j]
                        adj_clusters[o, 0] = U_cl

                        while st - j > 0 and clusts[st - j] != -1 and \
                          (clusts[st - j] == U_cl or clusts[st - j] == 0):
                            # again, skip over segments in the garbage
                            if clusts[st - j] != 0:
                                UD_counts[o, 0] += S.iloc[st - j, min_col]
                                UD_counts[o, 1] += S.iloc[st - j, maj_col]

                            j += 1

                    # maj/min counts of contiguous downstream segments belonging to the same cluster
                    if en + 1 < len(S):
                        j = 1
                        while en + j < len(S) and clusts[en + j] == 0:
                            j += 1

                        D_cl = clusts[en + j]
                        adj_clusters[o, 1] = D_cl

                        while en + j < len(S) and clusts[en + j] != -1 and \
                          (clusts[en + j] == D_cl or clusts[en + j] == 0):
                            if clusts[en + j] != 0:
                                UD_counts[o, 2] += S.iloc[en + j, min_col]
                                UD_counts[o, 3] += S.iloc[en + j, maj_col]

                            j += 1

                # if there are any segments being moved adjacent to already existing clusters, get local split/join likelihoods
                adj_idx = ~(adj_clusters == -1).all(1)

                if adj_idx.any():
                    # maj/min counts of the segment(s) being moved
                    #S_a = S.iloc[st:(en + 1), min_col].sum()
                    S_a = S.iloc[:, min_col].values[st:(en + 1)].sum()
                    #S_b = S.iloc[st:(en + 1), maj_col].sum()
                    S_b = S.iloc[:, maj_col].values[st:(en + 1)].sum()

                    # for each segment/segment block within this cluster,
                    for j in np.flatnonzero(adj_idx):
                        cl_u = adj_clusters[j, 0]
                        cl_d = adj_clusters[j, 1]
                        U_a = UD_counts[j, 0]
                        U_b = UD_counts[j, 1]
                        D_a = UD_counts[j, 2]
                        D_b = UD_counts[j, 3]

                        # adjacency likelihood of this segment remaining where it is
                        adj_AB += SJliks(
                          targ_clust = cur_clust, 
                          upstream_clust = cl_u, 
                          downstream_clust = cl_d, 
                          J_a = S_a, 
                          J_b = S_b,
                          U_a = U_a,
                          U_b = U_b,
                          D_a = D_a,
                          D_b = D_b
                        )

                        # adjacency likelihood of this segment joining each possible cluster:
                        # 1. those it is actually adjacent to (+ new cluster, garbage)
                        for cl in {-1, 0, cl_u, cl_d}:
                            idx = clust_sums.index(cl)
                            adj_BC[idx] += SJliks(
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
                            # we cannot send a segment to the garbage adjacent to any unassigned segment
                            # TODO: this means we cannot throw the first or last segments in the garbage
                            if cl == 0 and (cl_u == -1 or cl_d == -1):
                                adj_BC[idx] = -np.inf

                        # 2. clusters it is not adjacent to (use default split value)
                        for cl in clust_sums.keys() - ({-1, 0} | set(adj_clusters[adj_idx].ravel())):
                            idx = clust_sums.index(cl)
                            adj_BC[idx] += SJliks(
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
                else:
                    # we cannot send a segment to the garbage adjacent to any unassigned segment
                    adj_BC[clust_sums.index(0)] = -np.inf
            else:
                adj_BC[clust_sums.index(0)] = -np.inf

            # A+B,C -> A,B+C

            # A+B is likelihood of current cluster B is part of
            AB = ss.betaln(A_a + B_a + 1, A_b + B_b + 1)
            # C is likelihood of target cluster pre-join
            C = ss.betaln(C_ab[:, 0] + 1, C_ab[:, 1] + 1)
            # A is likelihood cluster B is part of, minus B
            A = ss.betaln(A_a + 1, A_b + 1)
            # B+C is likelihood of target cluster post-join
            BC = ss.betaln(C_ab[:, 0] + B_a + 1, C_ab[:, 1] + B_b + 1)

            #     L(join)           L(split)
            MLs = A + BC + adj_BC - (AB + C + adj_AB)

            MLs_max = np.max(MLs)

            #
            # priors

            # prior on previous cluster fractions

            prior_diff = []
            prior_com = []
            clust_prior_p = 1
            if clust_prior is not None: 
                #
                # divide prior into three sections:
                # * clusters in prior not currently active (if picked, will open a new cluster with that ID)
                # * clusters in prior currently active (if picked, will weight that cluster's posterior probability)
                # * currently active clusters not in the prior (if picked, would weight cluster's posterior probability with prior probability of making brand new cluster)

                # not currently active
                prior_diff = clust_prior.keys() - clust_counts.keys()

                # currently active clusters in prior
                prior_com = clust_counts.keys() & clust_prior.keys()

                # currently active clusters not in prior
                prior_null = clust_counts.keys() - clust_prior.keys()

                # order of prior vector:
                # [-1 (totally new cluster), <prior_diff>, <prior_com + prior_null>]
                prior_idx = np.r_[
                  np.r_[[clust_prior.index(x) for x in prior_diff]],
                  np.r_[[clust_prior.index(x) if x in clust_prior else 0 for x in (prior_com | prior_null | {0})]]
                ]

                prior_MLs = ss.betaln( # prior clusters + segment
                  np.r_[clust_prior_mat[prior_idx, 0]] + B_a + 1,
                  np.r_[clust_prior_mat[prior_idx, 1]] + B_b + 1
                ) \
                - (ss.betaln(B_a + 1, B_b + 1) + np.r_[np.r_[clust_prior_liks.values()][prior_idx]]) # prior clusters, segment

                clust_prior_p = np.maximum(np.exp(prior_MLs - prior_MLs.max())/np.exp(prior_MLs - prior_MLs.max()).sum(), 1e-300)

                # expand MLs to account for multiple new clusters
                MLs = np.r_[np.full(len(prior_diff), MLs[0]), MLs[1:]]
                
            # DP prior based on clusters sizes
            count_prior_d = clust_counts.copy()
            for k in prior_com:
                count_prior_d[k] += clust_count_prior[k]
            count_prior = np.r_[[clust_count_prior[x] for x in prior_diff], clust_count_prior[0], count_prior_d.values()]
            #count_prior = np.r_[[clust_count_prior[x] for x in prior_diff], clust_count_prior[0]*(len(S) - len(unassigned_segs)), clust_counts.values()]
            count_prior /= count_prior.sum()

            # choose to join a cluster or make a new one (choice_idx = 0) 
            choice_p = np.exp(MLs - MLs_max + np.log(count_prior) + np.log(clust_prior_p))/np.exp(MLs - MLs_max + np.log(count_prior) + np.log(clust_prior_p)).sum()
            choice_idx = np.random.choice(
              np.r_[0:len(MLs)],
              p = choice_p
            )
            # -1 = brand new, -2, -3, ... = -(prior clust index) - 2
            # 0 = garbage
            choice = np.r_[-np.r_[prior_diff] - 2, 0, clust_counts.keys()][choice_idx]

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

                clust_counts[new_clust_idx] = n_move
                S.iloc[seg_idx, clust_col] = new_clust_idx
                clusts[seg_idx] = new_clust_idx

                clust_sums[new_clust_idx] = np.r_[B_a, B_b]
                clust_members[new_clust_idx] = set(seg_idx)

            # send to garbage
            elif choice == 0:
                S.iloc[seg_idx, clust_col] = 0
                clusts[seg_idx] = 0

            # join existing cluster
            else:
                # if we are combining two clusters, take the index of the bigger one
                # this helps to keep cluster indices consistent
                if move_clust and clust_counts[choice] < n_move:
                    clust_counts[cl_idx] = clust_counts[choice]
                    clust_sums[cl_idx] = clust_sums[choice]
                    clust_members[cl_idx] = clust_members[choice]
                    S.iloc[np.flatnonzero(S["clust"] == choice), clust_col] = cl_idx
                    del clust_counts[choice]
                    del clust_sums[choice]
                    del clust_members[choice]
                    choice = cl_idx

                clust_counts[choice] += n_move 
                clust_sums[choice] += np.r_[B_a, B_b]
                S.iloc[seg_idx, clust_col] = choice
                clusts[seg_idx] = choice

                clust_members[choice].update(set(seg_idx))

            for si in seg_idx:
                unassigned_segs.discard(si)

            # track global state of cluster assignments
            # on average, each segment will have been reassigned every n_seg/(n_clust/2) iterations
            if burned_in and n_it - n_it_last > len(S)/(len(clust_counts)*2):
                segs_to_clusters.append(S["clust"].copy())
                phase_orientations.append(S["flipped"].copy())
                n_it_last = n_it

            n_it += 1

        return np.r_[segs_to_clusters], np.r_[phase_orientations]

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

    def run(self, N_seg_samps = 50, N_clust_samps = 5):
        seg_sample_idx = np.random.choice(self.n_samp - 1, N_seg_samps, replace = False)
        S, SNPs = self.load_seg_samp(seg_sample_idx[0])
        N_SNPs = len(SNPs)
        
        snps_to_clusters = -1*np.ones((N_clust_samps*N_seg_samps, N_SNPs), dtype = np.int16)
        snps_to_phases = np.zeros((N_clust_samps*N_seg_samps, N_SNPs), dtype = bool)
        snp_counts = -1*np.ones((N_seg_samps, N_SNPs, 2))
        Segs = []

        clust_prior = sc.SortedDict()
        clust_count_prior = sc.SortedDict()

        for n_it in range(N_seg_samps):
            if n_it > 0:
                S, SNPs = self.load_seg_samp(seg_sample_idx[n_it])

            # run clustering
            s2c, ph = self.run_DP(S, clust_prior = clust_prior, clust_count_prior = clust_count_prior, n_iter = N_clust_samps)

            # assign clusters to individual SNPs, to use as segment assignment prior for next DP iteration
            snps_to_clusters[N_clust_samps*n_it:N_clust_samps*(n_it + 1), :] = self.map_seg_clust_assignments_to_SNPs(s2c, S)

            # assign phase orientations to individual SNPs
            snps_to_phases[N_clust_samps*n_it:N_clust_samps*(n_it + 1), :] = self.map_seg_phases_to_SNPs(ph, S)

            # compute prior on cluster locations/counts
            S_a = np.zeros(s2c.max() + 1)
            S_b = np.zeros(s2c.max() + 1)
            N_c = np.zeros(s2c.max() + 1)
            n_iter_clust_exist = np.zeros(np.maximum(s2c.max(), clust_prior.peekitem(-1)[0]) + 1)
            for seg_assignments, seg_phases in zip(s2c, ph):
                # reset phases
                S2 = S.copy()
                S2.loc[S2["flipped"], ["min", "maj"]] = S2.loc[S2["flipped"], ["min", "maj"]].values[:, ::-1]

                # match phases to current sample
                S2.loc[seg_phases, ["min", "maj"]] = S2.loc[seg_phases, ["min", "maj"]].values[:, ::-1]

                S_a += npg.aggregate(seg_assignments, S2["min"], size = s2c.max() + 1)
                S_b += npg.aggregate(seg_assignments, S2["maj"], size = s2c.max() + 1)

                N_c += npg.aggregate(seg_assignments, 1, size = s2c.max() + 1)

                n_iter_clust_exist[np.unique(seg_assignments)] += 1

            S_a /= N_clust_samps
            S_b /= N_clust_samps
            N_c /= N_clust_samps

            c = np.c_[S_a, S_b]

            next_clust_prior = sc.SortedDict(zip(np.flatnonzero(c.sum(1) > 0), c[c.sum(1) > 0]))
            next_clust_count_prior = sc.SortedDict(zip(np.flatnonzero(c.sum(1) > 0), N_c[N_c > 0]))

            # iteratively update priors
            for k, v in next_clust_prior.items():
                nccp = next_clust_count_prior[k]
                if k in clust_prior:
                    # iteratively update average
                    clust_prior[k] += (v - clust_prior[k])/(n_iter_clust_exist[k] + 1)
                    clust_count_prior[k] += (nccp - clust_count_prior[k])/(n_iter_clust_exist[k] + 1)
                else:
                    clust_prior[k] = v
                    clust_count_prior[k] = nccp
            # for clusters that don't exist in this iteration, average them with 0
            for k, v in clust_prior.items():
                if k != -1 and k not in next_clust_prior:
                    clust_prior[k] -= clust_prior[k]/(n_iter_clust_exist[k] + 1)
                    clust_count_prior[k] -= clust_count_prior[k]/(n_iter_clust_exist[k] + 1)

            # remove zero counts from priors
            for kk in [k for k, v in clust_count_prior.items() if v == 0]:
                del clust_prior[kk]
                del clust_count_prior[kk]

            # remove garbage cluster from priors
            del clust_prior[0]
            del clust_count_prior[0]

            # get probability that individual SNPs are flipped, to use as probability for
            # flipping segments for next DP iteration
            flipped = np.zeros(S.iloc[-1, S.columns.get_loc("SNP_en")] + 1, dtype = bool)
            for _, st, en in S.loc[S["flipped"], ["SNP_st", "SNP_en"]].itertuples():
                flipped[st:en] = True

            # save overall segmentation for this sample
            Segs.append(S)

        return snps_to_clusters, snps_to_phases
