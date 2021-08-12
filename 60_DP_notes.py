import colorama
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import ncls
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

from capy import seq

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.auto_ref_correct.overdispersion92.no_phase_correct.pickle")

def load_seg_sample(samp_idx):
    all_segs = []
    all_SNPs = []
    all_PIs = []
    all_BPs = []

    maj_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("MAJ_COUNT")
    min_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("MIN_COUNT")

    alt_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("ALT_COUNT")
    ref_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("REF_COUNT")

    chunk_offset = 0
    for _, H in allelic_segs.dropna(subset = ["results"]).iterrows():
        r = copy.deepcopy(H["results"])

        # set phasing orientation back to original
        for st, en in r.F.intervals():
            # code excised from flip_hap
            x = r.P.iloc[st:en, maj_idx].copy()
            r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
            r.P.iloc[st:en, min_idx] = x

        # save SNPs for this chunk
        all_SNPs.append(pd.DataFrame({ "maj" : r.P["MAJ_COUNT"], "min" : r.P["MIN_COUNT"], "gpos" : seq.chrpos2gpos(r.P.loc[0, "chr"], r.P["pos"]) }))

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

        # save breakpoints/phase orientations for this chunk
        all_BPs.append(bpl + chunk_offset)
        all_PIs.append(pi_samp.intervals() + chunk_offset if r.phase_correct else [])

        chunk_offset += len(r.P)

    # convert samples into dataframe
    S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj", "A_alt", "A_ref", "B_alt", "B_ref"])

    # convert chr-relative positions to absolute genomic coordinates
    S["start_gp"] = seq.chrpos2gpos(S["chr"], S["start"])
    S["end_gp"] = seq.chrpos2gpos(S["chr"], S["end"])

    # initial cluster assignments
    S["clust"] = -1 # initially, all segments are unassigned
    S.iloc[0, S.columns.get_loc("clust")] = 0 # first segment is assigned to cluster 0

    # initial phasing orientation
    S["flipped"] = False

    return S, pd.concat(all_SNPs, ignore_index = True), np.concatenate(all_BPs), np.concatenate(all_PIs)

def run_DP(S, seg_clust_prior = None, clust_prior = sc.SortedDict()):
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
    # initialize cluster tracking hash tables

    clust_counts = sc.SortedDict(S["clust"].value_counts().drop(-1, errors = "ignore"))
    # for the first round of clustering, this is { 0 : 1 }
    clust_sums = sc.SortedDict({
      **{ k : np.r_[v["min"], v["maj"]] for k, v in S.groupby("clust")[["min", "maj"]].sum().to_dict(orient = "index").items() },
      **{-1 : np.r_[0, 0]}
    })
    # for the first round, this is { -1 : np.r_[0, 0], 0 : np.r_[S[0, "min"], S[0, "maj"]] }
    clust_members = sc.SortedDict({ k : set(v) for k, v in S.groupby("clust").groups.items() if k != -1 })
    # for the first round, this is { 0 : {0} }
    unassigned_segs = sc.SortedList(S.index[S["clust"] == -1])

    # store likelihoods for each cluster in the prior (from previous iterations)
    clust_prior[-1] = np.r_[0, 0]
    clust_prior_liks = sc.SortedDict({ k : ss.betaln(v[0] + 1, v[1] + 1) for k, v in clust_prior.items()})
    clust_prior_mat = np.r_[clust_prior.values()]

    max_clust_idx = np.max(clust_members.keys() | clust_prior.keys() if clust_prior is not None else {})

    # containers for saving the MCMC trace
    clusters_to_segs = [[] for i in range(len(S))]
    segs_to_clusters = []
    phase_orientations = []

    burned_in = False

    n_it = 0
    n_it_last = 0
    while len(segs_to_clusters) < 50: # TODO: allow this to be tweaked
        if not n_it % 1000:
            print(S["clust"].value_counts().drop(-1, errors = "ignore").value_counts().sort_index())
            print("n unassigned: {}".format((S["clust"] == -1).sum()))

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
                seg_idx = np.r_[np.random.choice(unassigned_segs)]
            else:
                seg_idx = np.r_[np.random.choice(len(S))]

            n_move = 1

            # if segment was already assigned to a cluster, unassign it
            cur_clust = int(S.iloc[seg_idx, clust_col])
            if cur_clust != -1:
                clust_counts[cur_clust] -= 1
                if clust_counts[cur_clust] == 0:
                    del clust_counts[cur_clust]
                    del clust_sums[cur_clust]
                    del clust_members[cur_clust]
                else:
                    clust_sums[cur_clust] -= np.r_[S.iloc[seg_idx, min_col], S.iloc[seg_idx, maj_col]]
                    clust_members[cur_clust] -= set(seg_idx)

                unassigned_segs.add(seg_idx)
                S.iloc[seg_idx, clust_col] = -1

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
            S.iloc[seg_idx, clust_col] = -1

            move_clust = True

        #
        # perform phase correction on segment/cluster
        # flip min/maj with probability that alleles are oriented the "wrong" way
        x = s.beta.rvs(S.iloc[seg_idx, aalt_col].sum() + 1, S.iloc[seg_idx, aref_col].sum() + 1, size = [len(seg_idx), 30])
        y = s.beta.rvs(S.iloc[seg_idx, balt_col].sum() + 1, S.iloc[seg_idx, bref_col].sum() + 1, size = [len(seg_idx), 30])
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
        C_ab = np.r_[clust_sums.values()] # first term (-1) = make new cluster

        # A+B,C -> A,B+C

        # A+B is likelihood of current cluster B is part of
        AB = ss.betaln(A_a + B_a + 1, A_b + B_b + 1)
        # C is likelihood of target cluster pre-join
        C = ss.betaln(C_ab[:, 0] + 1, C_ab[:, 1] + 1)
        # A is likelihood cluster B is part of, minus B
        A = ss.betaln(A_a + 1, A_b + 1)
        # B+C is likelihood of target cluster post-join
        BC = ss.betaln(C_ab[:, 0] + B_a + 1, C_ab[:, 1] + B_b + 1)

        #
        # prior terms
        alpha0 = 1 # hyperparameter for similarities between segments TODO: make tunable
        join_prior_lik = 0
        if seg_clust_prior is not None:
            # seg_clust_prior: n_clust (\psi_{i-1}) x n_seg
            # assignments of segment/cluster being moved in previous iterations of DP
            alpha1 = seg_clust_prior[:, seg_idx].sum(1)[None, :]

            # dim(alpha1) = 1 x n_clust (\psi_{i-1})

            # assignments of all target clusters over previous iterations of DP
            # TODO: this should be dynamically updated, rather than recomputed every single iteration
            alpha2 = np.zeros([clust_counts.keys()[-1] + 2, seg_clust_prior.shape[0]])
            # dim(alpha2) = n_clust (\psi_i) x n_clust (\psi_{i-1})
            for k, v in clust_members.items():
                alpha2[k + 1, :] = seg_clust_prior[:, list(v)].sum(1) # k + 1 to let alpha2[0, :] correspond to probability of opening new cluster (all zeros)

            # subtract off values of segment/cluster being moved (XXX: not necessary -- we've already removed segment being moved from the cluster it was previously part of)
            #alpha2[[cur_clust + 1], :] -= alpha1
            # XXX: in the writeup, it says that we consider the marginal likelihood of every other possible cluster that S could join, excluding the one that S was already part of. we don't actually do this here. should we? S going back to the cluster it was originally part of is equivalent to not accepting the move. how does this factor into the MCMC?

            # subset rows of alpha2 to only clusters that currently exist
            alpha2 = alpha2[np.r_[-1, clust_members.keys()] + 1, :]

            join_prior_lik = multibetaln(alpha1 + alpha2 + alpha0) - (multibetaln(alpha1 + alpha0) + multibetaln(alpha2 + alpha0))

            # we do not impose any prior on opening a new cluster
            join_prior_lik[0] = 0

        #     L(join)  L(split)
        MLs = A + BC - (AB + C) + join_prior_lik

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
              np.r_[[clust_prior.index(x) if x in clust_prior else 0 for x in (prior_com | prior_null)]]
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
        count_prior = np.r_[np.full(len(prior_diff), 0.1/len(prior_diff)), clust_counts.values()]
        count_prior /= count_prior.sum()

        # choose to join a cluster or make a new one (choice_idx = 0) 
        T = 1 # temperature parameter for scaling choice distribution
        choice_p = np.exp(T*(MLs - MLs_max + np.log(count_prior) + np.log(clust_prior_p)))/np.exp(T*(MLs - MLs_max + np.log(count_prior) + np.log(clust_prior_p))).sum()
        choice_idx = np.random.choice(
          np.r_[0:len(MLs)],
          p = choice_p
        )
        #choice = np.r_[-1, clust_counts.keys()][choice_idx]
        # -1 = brand new, -2, -3, ... = -(prior clust index) - 2
        choice = np.r_[-np.r_[prior_diff] - 2, clust_counts.keys()][choice_idx]

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

            clust_sums[new_clust_idx] = np.r_[B_a, B_b]
            clust_members[new_clust_idx] = set(seg_idx)

        # join existing cluster
        else:
            clust_counts[choice] += n_move 
            clust_sums[choice] += np.r_[B_a, B_b]
            S.iloc[seg_idx, clust_col] = choice

            clust_members[choice].update(set(seg_idx))

        for si in seg_idx:
            unassigned_segs.remove(si)

        # track cluster assignment for segment(s) (XXX: may not be necessary anymore)
        if burned_in:
            for seg in seg_idx:
                clusters_to_segs[seg].append(choice if choice != -1 else max_clust_idx)

        # track global state of cluster assignments
        # on average, each segment will have been reassigned every n_seg/(n_clust/2) iterations
        if burned_in and n_it - n_it_last > len(S)/(len(clust_counts)*2):
            segs_to_clusters.append(S["clust"].copy())
            phase_orientations.append(S["flipped"].copy())
            n_it_last = n_it

        n_it += 1

    return np.r_[segs_to_clusters], np.r_[phase_orientations]

# map trace of segment cluster assignments to the SNPs within
def map_seg_clust_assignments_to_SNPs(segs_to_clusters, S):
    st_col = S.columns.get_loc("SNP_st")
    en_col = S.columns.get_loc("SNP_en")
    snps_to_clusters = np.zeros((segs_to_clusters.shape[0], S.iloc[-1, en_col] + 1), dtype = int)
    for i, seg_assign in enumerate(segs_to_clusters):
        for j, seg in enumerate(seg_assign):
            snps_to_clusters[i, S.iloc[j, st_col]:S.iloc[j, en_col]] = seg

    return snps_to_clusters

def map_seg_phases_to_SNPs(phase, S):
    st_col = S.columns.get_loc("SNP_st")
    en_col = S.columns.get_loc("SNP_en")
    snps_to_phase = np.zeros((phase.shape[0], S.iloc[-1, en_col] + 1), dtype = int)
    for i, phase_orient in enumerate(phase):
        for j, ph in enumerate(phase_orient):
            snps_to_phase[i, S.iloc[j, st_col]:S.iloc[j, en_col]] = ph

    return snps_to_phase

#
# test code for running multiple iterations of DP, implementing prior on clustering 

N_seg_samps = 10
N_clust_samps = 50
N_SNPs = 11768

snps_to_clusters = -1*np.ones((N_clust_samps*N_seg_samps, N_SNPs), dtype = np.int16)
snps_to_phases = np.zeros((N_clust_samps*N_seg_samps, N_SNPs), dtype = bool)
snp_counts = -1*np.ones((N_seg_samps, N_SNPs, 2))
Segs = []

seg_sample_idx = np.random.choice(N_clust_samps, N_seg_samps, replace = False) # FIXME: need to determine total number of segmentation samples
seg_sample_idx = np.r_[47, 17, 27, 39, 23, 37,  3, 18, 42,  1]

for n_it in range(10):
    S, SNPs, BPs, PIs = load_seg_sample(seg_sample_idx[n_it])

    # get prior on segments' cluster assignments, if we've already performed a clustering step
    seg_clust_prior = None
    if n_it > 0:
        n_clust_bins = snps_to_clusters.max() + 1
        seg_clust_prior = np.zeros([n_clust_bins, len(S)])
        for i, r in enumerate(S.itertuples()):
            seg_clust_prior[:, i] = np.bincount(snps_to_clusters[:N_clust_samps*n_it, r.SNP_st:r.SNP_en].ravel(), minlength = n_clust_bins)

        # initialize the cluster assignments in S based on the previous DP
        np.random.choice(seg_clust_prior.shape[0], p = seg_clust_prior[:, 0]/seg_clust_prior[:, 0].sum())
        S["clust"] = [np.random.choice(seg_clust_prior.shape[0], p = x/x.sum()) for x in seg_clust_prior.T]

        # flip segments based on the previous DP
        flip_frac = np.r_[[flipped[z].mean() for z in [slice(x, y) for x, y in S[["SNP_st", "SNP_en"]].values]]]
        S["flipped"] = flip_frac > np.random.rand(len(flip_frac))
        S.loc[S["flipped"], ["min", "maj"]] = S.loc[S["flipped"], ["min", "maj"]].values[:, ::-1]
        S.loc[S["flipped"], ["A_alt", "B_alt"]] = S.loc[S["flipped"], ["A_alt", "B_alt"]].values[:, ::-1]
        S.loc[S["flipped"], ["A_ref", "B_ref"]] = S.loc[S["flipped"], ["A_ref", "B_ref"]].values[:, ::-1]

        # unassign segments with ambiguous flip fraction
        S.loc[~np.isin(flip_frac, [0, 1]), "clust"] = -1

    # run clustering
    #s2c = run_DP(S, seg_clust_prior)
    s2c, ph = run_DP(S, None)

    # assign clusters to individual SNPs, to use as segment assignment prior for next DP iteration
    snps_to_clusters[N_clust_samps*n_it:N_clust_samps*(n_it + 1), :] = map_seg_clust_assignments_to_SNPs(s2c, S)

    # assign clusters to individual SNPs, to use as segment assignment prior for next DP iteration
    snps_to_phases[N_clust_samps*n_it:N_clust_samps*(n_it + 1), :] = map_seg_phases_to_SNPs(ph, S)

    # adjust SNPs according to phase orientation for this sample
    for st, en in PIs:
        SNPs.loc[st:en, ["maj", "min"]] = SNPs.loc[st:en, ["maj", "min"]].values[:, ::-1]

    # get probability that individual SNPs are flipped, to use as probability for
    # flipping segments for next DP iteration
    flipped = np.zeros(S.iloc[-1, S.columns.get_loc("SNP_en")] + 1, dtype = bool)
    for _, st, en in S.loc[S["flipped"], ["SNP_st", "SNP_en"]].itertuples():
        flipped[st:en] = True

    # save rephased A/B counts for each SNP (may not end up needing this?)
    snp_counts[n_it, :, :] = SNPs.loc[:, ["maj", "min"]]

    # save overall segmentation for this sample
    Segs.append(S)

# Dirichlet marginal likelihood:

def multibetaln(alpha):
    return ss.gammaln(alpha).sum(1) - ss.gammaln(alpha.sum(1))

#
# plot

#from glasbey import glasbey

colors = mpl.cm.get_cmap("tab10").colors

#
# single DP iteration plots {{{

# plot first 20 clusters overlaid (figure 2), and one cluster per subplot (figure 1)
f1 = plt.figure(2); plt.clf()
ax = plt.gca()
ax.set_xlim([0, S["end_gp"].max()])
ax.set_ylim([0, 1])

plt.figure(1); plt.clf()
f, axs = plt.subplots(20, 1, num = 1);
for a in axs:
    a.set_xlim([0, S["end_gp"].max()])
    a.set_ylim([0, 1])
    a.set_xticks([])
    a.set_yticks([])

for i, clust_idx in enumerate(S["clust"].value_counts().index[0:20]):
    if clust_idx == -1:
        continue
    for _, r in S.loc[S["clust"] == clust_idx].iterrows():
        ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], r["min"] + 1, r["maj"] + 1)
        axs[i].add_patch(mpl.patches.Rectangle((r["start_gp"], ci_lo), r["end_gp"] - r["start_gp"], ci_hi - ci_lo, facecolor = colors[i % len(colors)], fill = True, alpha = 0.9, zorder = 1000))
        ax.add_patch(mpl.patches.Rectangle((r["start_gp"], ci_lo), r["end_gp"] - r["start_gp"], ci_hi - ci_lo, facecolor = colors[i % len(colors)], fill = True, alpha = 0.1, zorder = 1000))

# plot beta dists. for clusters

r = np.linspace(0, 1, 2000)
plt.figure(3); plt.clf()
plts = []
for i, clust_idx in enumerate(S["clust"].value_counts().index[0:20]):
    if clust_idx == -1:
        continue
    plts.append(plt.plot(r, s.beta.pdf(r, S.loc[S["clust"] == clust_idx, "min"].sum(), S.loc[S["clust"] == clust_idx, "maj"].sum()), color = colors[i % len(colors)])[0])

plt.legend(plts, S["clust"].value_counts().index[0:20])

# plot beta dists. for segments, colored by cluster assignment

r = np.linspace(0, 1, 2000)
plt.figure(4); plt.clf()
plts = []
for i, clust_idx in enumerate(S["clust"].value_counts().index[0:20]):
    if clust_idx == -1:
        continue
    for _, mn, mj in S.loc[S["clust"] == clust_idx, ["min", "maj"]].itertuples():
        plt.plot(r, s.beta.pdf(r, mn, mj), color = colors[i % len(colors)], alpha = 0.3)

# }}}

#
# multi DP iteration plots {{{

# probabilistic assignment of segments to clusters {{{

f1 = plt.figure(20); plt.clf()
ax = plt.gca()
ax.set_xlim([0, S["end_gp"].max()])
ax.set_ylim([0, 1])

for i, r in S.iterrows():
    ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], r["min"] + 1, r["maj"] + 1)
    clust_trace = np.bincount(clusters_to_segs[i])
    clust_probs = clust_trace/clust_trace.sum()
    for prob, idx in zip(clust_probs[clust_trace > 0], np.flatnonzero(clust_trace)):
        ax.add_patch(mpl.patches.Rectangle((r["start_gp"], ci_lo), r["end_gp"] - r["start_gp"], ci_hi - ci_lo, facecolor = colors[idx % len(colors)], fill = True, alpha = prob, zorder = 1000))

# }}}

# use cluster mean for each segment {{{

f1 = plt.figure(21); plt.clf()
ax = plt.gca()
ax.set_xlim([0, S["end_gp"].max()])
ax.set_ylim([0, 1])

h_points = 72*ax.bbox.height/f1.dpi

clust_u = np.unique(s2c)
color_idx = dict(zip(clust_u, np.r_[0:len(clust_u)]))

for seg_assignments, seg_phases in zip(s2c, ph):
    # reset phases
    S2 = S.copy()
    S2.loc[S2["flipped"], ["min", "maj"]] = S2.loc[S2["flipped"], ["min", "maj"]].values[:, ::-1]

    # match phases to current sample
    S2.loc[seg_phases, ["min", "maj"]] = S2.loc[seg_phases, ["min", "maj"]].values[:, ::-1]

    S_a = npg.aggregate(seg_assignments, S2["min"])
    S_b = npg.aggregate(seg_assignments, S2["maj"])

    print(ss.betaln(S_a + 1, S_b + 1).sum())

    CIs = s.beta.ppf([0.025, 0.5, 0.975], S_a[:, None] + 1, S_b[:, None] + 1)

    for su in np.unique(seg_assignments):
        idx = seg_assignments == su

        plt.plot(
          np.c_[S2.loc[idx, "start_gp"], S2.loc[idx, "end_gp"], np.full(idx.sum(), np.nan)].ravel(),
          (CIs[su, 1]*np.ones([idx.sum(), 3])).ravel(),
          color = np.array(colors)[color_idx[su] % len(colors)],
          linewidth = h_points*(CIs[su, 2] - CIs[su, 0]),
          alpha = 1/50,
          zorder = 1000
        )

    # overlay SNPs
    SNPs2 = SNPs.copy()
    for _, st, en in S2.loc[seg_phases, ["SNP_st", "SNP_en"]].itertuples():
        SNPs2.iloc[st:en, [0, 1]] = SNPs2.iloc[st:en, [1, 0]]
    plt.scatter(SNPs2["gpos"], SNPs2["min"]/(SNPs2[["min", "maj"]].sum(1)), s = 0.01, color = 'k', zorder = 0, alpha = 0.05)

# chromosome boundaries
chrbdy = allelic_segs.dropna().loc[:, ["start", "end"]]
chr_ends = chrbdy.loc[chrbdy["start"] != 0, "end"].cumsum()
for chrbdy in chr_ends[:-1]:
    plt.axvline(chrbdy, color = 'k')

# allelic imbalances given purity (alpha)
alpha = 2*(0.8) - 1 # plug into (...) allelic imbalance corresponding to LoH eyeballed from plot
phis = np.r_[1, 1/2, 2/3, 3/4, 3/5, 4/5, 4/7, 5/6, 5/7, 5/8, 5/9]
for phi in phis:
    plt.axhline(alpha*phi + (1 - alpha)/2, color = 'k', linestyle = ':')

ax2 = plt.twinx()
ax2.set_yticks(alpha*phis + (1 - alpha)/2)
ax2.set_yticklabels(["0:1", "1:1", "1:2", "1:3", "2:3", "1:4", "3:4", "1:5", "2:5", "3:5", "4:5"])

# }}}

# histogram of segment assignments {{{

_, s2cu = np.unique(s2c, return_inverse = True)
s2cu = s2cu.reshape([50, -1])

s2c_hist = np.zeros((s2cu.max() + 1, s2cu.shape[1]))
for i in range(s2cu.shape[1]):
    s2c_hist[:, i] = np.bincount(s2cu[:, i], minlength = s2cu.max() + 1)

s2c_hist /= s2c_hist.sum(0)

# make prior?
S_a = np.zeros(s2c.max() + 1)
S_b = np.zeros(s2c.max() + 1)
for seg_assignments, seg_phases in zip(s2c, ph):
    # reset phases
    S2 = S.copy()
    S2.loc[S2["flipped"], ["min", "maj"]] = S2.loc[S2["flipped"], ["min", "maj"]].values[:, ::-1]

    # match phases to current sample
    S2.loc[seg_phases, ["min", "maj"]] = S2.loc[seg_phases, ["min", "maj"]].values[:, ::-1]

    S_a += npg.aggregate(seg_assignments, S2["min"], size = s2c.max() + 1)
    S_b += npg.aggregate(seg_assignments, S2["maj"], size = s2c.max() + 1)

S_a /= N_clust_samps
S_b /= N_clust_samps

c = np.c_[S_a, S_b]
plt.figure(333); plt.clf()
r = np.linspace(0, 1, 1000)
for a, b in c[c.sum(1) > 0]:
    plt.plot(r, s.beta.pdf(r, a + 1, b + 1))

clust_prior = sc.SortedDict(zip(np.flatnonzero(c.sum(1) > 0), c[c.sum(1) > 0]))

a_t = 200; b_t = 70
a_t = 200; b_t = 400
prior = ss.betaln(a_t + c[c.sum(1) > 0, 0] + 1, b_t + c[c.sum(1) > 0, 1] + 1) - (ss.betaln(a_t + 1, b_t + 1) + ss.betaln(c[c.sum(1) > 0, 0] + 1, c[c.sum(1) > 0, 1] + 1))
prior_norm = np.exp(prior)/np.exp(prior).sum()

prior = ss.betaln(a_t + c[c.sum(1) > 0, 0] + 1, b_t + c[c.sum(1) > 0, 1] + 1) - (ss.betaln(a_t + 1, b_t + 1) + ss.betaln(c[c.sum(1) > 0, 0] + 1, c[c.sum(1) > 0, 1] + 1))
prior_norm = np.exp(prior)/np.exp(prior).sum()

# }}}

#
# multi DP iteration/multi segment sample plots {{{

clust_u = np.unique(snps_to_clusters)
color_idx = dict(zip(clust_u, np.r_[0:len(clust_u)]))

plt.figure(1337); plt.clf()
for i, (snp_assignments, phase_assignments) in enumerate(zip(snps_to_clusters, snps_to_phases)):
    Seg = Segs[i//N_clust_samps].copy()
    seg_assignments = snp_assignments[Seg["SNP_st"]]
    seg_phases = phase_assignments[Seg["SNP_st"]]

    ph_idx = seg_phases != Seg["flipped"]
    Seg.loc[ph_idx, ["min", "maj"]] = Seg.loc[ph_idx, ["min", "maj"]].values[:, ::-1]

    S_a = npg.aggregate(seg_assignments, Seg["min"])
    S_b = npg.aggregate(seg_assignments, Seg["maj"])

    CIs = s.beta.ppf([0.025, 0.5, 0.975], S_a[:, None] + 1, S_b[:, None] + 1)

    for su in np.unique(seg_assignments):
        idx = seg_assignments == su

        plt.plot(
          np.c_[Seg.loc[idx, "start_gp"], Seg.loc[idx, "end_gp"], np.full(idx.sum(), np.nan)].ravel(),
          (CIs[su, 1]*np.ones([idx.sum(), 3])).ravel(),
          color = np.array(colors)[color_idx[su] % len(colors)],
          linewidth = 1,
          alpha = 0.1,
          zorder = 1000
        )

    plt.scatter(SNPs["gpos"], SNPs["min"]/(SNPs[["min", "maj"]].sum(1)), s = 0.01, color = 'k', zorder = 0, alpha = 0.1)

# }}}

# old scrap code

clust_col = S.columns.get_loc("clust")

clusts = { k : v for k, v in zip(range(0, len(S)),
  np.hstack([
    np.ones([len(S), 1], dtype = np.int),
    S.loc[:, ["min", "maj"]].values,
    #ss.betaln(S.loc[:, ["min"]] + 1, S.loc[:, ["maj"]] + 1)
  ])
)}
clust_keys = sc.SortedSet(clusts.keys())

clust_counts = sc.SortedDict({ 1 : len(clust_keys) })
clust_counts_map = sc.SortedDict({ 1 : clust_keys })

# number of segs in cluster, min. hap count, maj hap count

N = len(S)
alpha = 1
max_clust_idx = len(clusts)

for n_it in range(0, 5*len(S)):
    if not n_it % 1000:
        print(S["clust"].value_counts().value_counts().sort_index())

    i = np.random.choice(len(S)) # pick a segment at random

    min_c = S.loc[i, "min"]
    maj_c = S.loc[i, "maj"]

    old_idx = S.iat[i, clust_col]

    q = np.random.rand() < alpha/(N - 1 + alpha)

    # make a new cluster {{{
    if q == 1:
        # update counts

        # 1. decrement old_count; increment old_count - 1
        old_count = clusts[old_idx][0]
        clust_counts[old_count] -= 1
        clust_counts_map[old_count].remove(old_idx)
        if clust_counts[old_count] == 0:
            del clust_counts[old_count]
            del clust_counts_map[old_count]

        if old_count - 1 > 0:
            if old_count - 1 not in clust_counts:
                clust_counts[old_count - 1] = 0 
                clust_counts_map[old_count - 1] = sc.SortedSet()
            clust_counts[old_count - 1] += 1
            clust_counts_map[old_count - 1].add(old_idx)

        # 2. increment new count
        new_count = 1
        if new_count not in clust_counts:
            clust_counts[new_count] = 0
            clust_counts_map[new_count] = sc.SortedSet()
        clust_counts[new_count] += 1
        clust_counts_map[new_count].add(max_clust_idx)

        # remove from old cluster
        clusts[old_idx][0] -= 1
        clusts[old_idx][1] -= min_c
        clusts[old_idx][2] -= maj_c
        if clusts[old_idx][0] == 0:
            del clusts[old_idx]

        clusts[max_clust_idx] = [1, min_c, maj_c]
        S.iat[i, clust_col] = max_clust_idx
        max_clust_idx += 1
    # }}}

    # try to join an existing cluster
    else:
        # probability of picking a cluster of size d[:, 0]
        d = np.r_[clust_counts.items()]
        denom = d[:, 0]@d[:, 1]

        # pick a cluster size to try joining
        sz = np.random.choice(d[:, 0], p = d[:, 0]*d[:, 1]/denom)

        # pick a cluster within that size group
        j = clust_counts_map[sz][np.random.choice(len(clust_counts_map[sz]))]

        # we are trying to join this segment with its currently assigned cluster
        if j == old_idx:
            continue

        c = clusts[j]

        # accept proposal via Metropolis
        # A+B,C -> A,B+C
        # A+B is likelihood of current cluster B is part of
        AB = ss.betaln(clusts[old_idx][1] + 1, clusts[old_idx][2] + 1)
        # C is likelihood of target cluster pre-join
        C = ss.betaln(c[1] + 1, c[2] + 1)
        # A is likelihood cluster B is part of, minus B
        A = ss.betaln(clusts[old_idx][1] - min_c + 1, clusts[old_idx][2] - maj_c + 1)
        # B+C is likelihood of target cluster post-join
        BC = ss.betaln(c[1] + min_c + 1, c[2] + maj_c + 1)

        ML_join = A + BC
        ML_split = AB + C

        if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split):
            # add to new cluster
            clusts[j][0] += 1
            clusts[j][1] += min_c
            clusts[j][2] += maj_c

            S.iat[i, clust_col] = j

            # update counts

            # 1. decrement old_count; increment old_count - 1
            old_count = clusts[old_idx][0]
            clust_counts[old_count] -= 1
            clust_counts_map[old_count].remove(old_idx)
            if clust_counts[old_count] == 0:
                del clust_counts[old_count]
                del clust_counts_map[old_count]

            if old_count - 1 > 0:
                if old_count - 1 not in clust_counts:
                    clust_counts[old_count - 1] = 0 
                    clust_counts_map[old_count - 1] = sc.SortedSet()
                clust_counts[old_count - 1] += 1
                clust_counts_map[old_count - 1].add(old_idx)

            # 2. increment new count; decrement new_count - 1
            new_count = clusts[j][0]
            if new_count not in clust_counts:
                clust_counts[new_count] = 0
                clust_counts_map[new_count] = sc.SortedSet()
            clust_counts[new_count] += 1
            clust_counts_map[new_count].add(j)

            if new_count - 1 > 0:
                clust_counts[new_count - 1] -= 1
                clust_counts_map[new_count - 1].remove(j)
                if clust_counts[new_count - 1] == 0:
                    del clust_counts[new_count - 1]
                    del clust_counts_map[new_count - 1]

            # remove from old cluster
            clusts[old_idx][0] -= 1
            clusts[old_idx][1] -= min_c
            clusts[old_idx][2] -= maj_c
            if clusts[old_idx][0] == 0:
                del clusts[old_idx]
