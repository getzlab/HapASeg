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

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.auto_ref_correct.overdispersion92.pickle")

maj_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("MAJ_COUNT")
min_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("MIN_COUNT")

def load_seg_sample(samp_idx):
    all_segs = []

    chunk_offset = 0
    for _, H in allelic_segs.dropna(subset = ["results"]).iterrows():
        r = copy.deepcopy(H["results"])
        
        # set phasing orientation back to original
        for st, en in r.F.intervals():
            # code excised from flip_hap
            x = r.P.iloc[st:en, maj_idx].copy()
            r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
            r.P.iloc[st:en, min_idx] = x

        # draw breakpoint, phasing, and SNP inclusion sample from segmentation MCMC trace
        bp_samp, pi_samp, inc_samp = (r.breakpoint_list[samp_idx], r.phase_interval_list[samp_idx], r.include[samp_idx])
        # flip everything according to sample
        for st, en in pi_samp.intervals():
            x = r.P.iloc[st:en, maj_idx].copy()
            r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
            r.P.iloc[st:en, min_idx] = x

        bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]

        # get major/minor sums for each segment
        for st, en in bpl:
            all_segs.append([
              st + chunk_offset, en + chunk_offset,
              r.P.loc[st, "chr"], r.P.loc[st, "pos"], r.P.loc[en, "pos"],
              r._Piloc(st, en, min_idx, inc_samp).sum(),
              r._Piloc(st, en, maj_idx, inc_samp).sum()
            ])

        # flip everything back
        for st, en in pi_samp.intervals():
            # TODO: can replace with flip_hap()?
            x = r.P.iloc[st:en, maj_idx].copy()
            r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
            r.P.iloc[st:en, min_idx] = x

        chunk_offset += len(r.P)

    # convert samples into dataframe
    S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj"])

    # construct overlap matrix
    S["start_gp"] = seq.chrpos2gpos(S["chr"], S["start"])
    S["end_gp"] = seq.chrpos2gpos(S["chr"], S["end"])

    # other fields of S
    S["clust"] = -1 # initially, all segments are unassigned
    clust_col = S.columns.get_loc("clust")
    min_col = S.columns.get_loc("min")
    maj_col = S.columns.get_loc("maj")
    S.iloc[0, clust_col] = 0 # first segment is assigned to cluster 0

    return S

def run_DP(S, seg_prior = None):
    # define column indices
    clust_col = S.columns.get_loc("clust")
    min_col = S.columns.get_loc("min")
    maj_col = S.columns.get_loc("maj")

    # initialize cluster tracking hash tables

    # TODO: I don't think we actually need clust_counts anymore, since we aren't doing a true DP process where the probability of joining a cluster depends on the number of members
    clust_counts = sc.SortedDict(S["clust"].value_counts().drop(-1, errors = "ignore"))
    # for the first round of clustering, this is { 0 : 1 }
    clust_sums = sc.SortedDict({
      **{ k : np.r_[v["min"], v["maj"]] for k, v in S.groupby("clust")[["min", "maj"]].sum().to_dict(orient = "index").items() },
      **{-1 : np.r_[0, 0]}
    })
    # for the first round, this is { -1 : np.r_[0, 0], 0 : np.r_[S[0, "min"], S[0, "maj"]] }
    clust_members = sc.SortedDict({ k : set(v) for k, v in S.groupby("clust").groups.items() if k != -1 })
    # for the first round, this is { 0 : {0} }

    max_clust_idx = np.max(clust_members.keys())

    # containers for saving the MCMC trace
    clusters_to_segs = [[] for i in range(len(S))]
    segs_to_clusters = []

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

                S.iloc[seg_idx, clust_col] = -1

        # pick a cluster at random
        else:
            # it only makes sense to try joining two clusters if there are at least two of them!
            if len(clust_counts) < 2:
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
            S.iloc[seg_idx, clust_col] = -1

            # NOTE: in the previous code, this accidentally was -=, not =
            # leaving comment here for posterity
            #clust_sums[cl_idx] = np.r_[0, 0]

            move_clust = True

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

        #     L(join)  L(split)
        MLs = A + BC - (AB + C)

        # if we are moving an entire cluster, it does not make sense to let it
        # create a new cluster, since this will make cluster indices inconsistent.
        if move_clust:
            MLs[0] = -np.inf

        MLs_max = np.max(MLs)

        # choose to join a cluster or make a new one (choice_idx = 0) 
        T = 1 # temperature parameter for scaling choice distribution
        choice_p = np.exp(T*(MLs - MLs_max))/np.exp(T*(MLs - MLs_max)).sum()
        choice_idx = np.random.choice(
          np.r_[0:(len(clust_counts) + 1)],
          p = choice_p
        )
        choice = np.r_[-1, clust_counts.keys()][choice_idx]

        # accept proposal via Metropolis
        # A+B,C -> A,B+C
        # C is likelihood of target cluster pre-join
        C_c = C[choice_idx]
        # B+C is likelihood of target cluster post-join
        BC_c = BC[choice_idx]

        ML_join = A + BC_c
        ML_split = AB + C_c

        #AB+C <- A+BC

        # BC is likelihood of target cluster post-join
        # == BC_c

        # A is likelihood of all clusters pre-join
        # == C above

        # C is likelihood of target cluster without join
        # == C_c

        # AB is likelihood of all clusters post-join
        # == BC above

        MLs_rev = (BC + C_c) - (BC_c + C)

        # when moving an entire cluster, we cannot open a new one
        if move_clust:
            MLs_rev[0] = -np.inf

        MLs_rev_max = np.max(MLs_rev)

        choice_p_rev = np.exp(T*(MLs_rev - MLs_rev_max))/np.exp(T*(MLs_rev - MLs_rev_max)).sum()
        q_rat = np.log(choice_p_rev[choice_idx]) - np.log(choice_p[choice_idx]) 

        # accept proposal
        if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split + q_rat):
            # create new cluster
            if choice == -1:
                max_clust_idx += 1
                clust_counts[max_clust_idx] = n_move
                S.iloc[seg_idx, clust_col] = max_clust_idx

                clust_sums[max_clust_idx] = np.r_[B_a, B_b]
                clust_members[max_clust_idx] = set(seg_idx)

            # join existing cluster
            else:
                clust_counts[choice] += n_move 
                clust_sums[choice] += np.r_[B_a, B_b]
                S.iloc[seg_idx, clust_col] = choice

                clust_members[choice].update(set(seg_idx))

            # track cluster assignment for segment(s)
            if burned_in:
                for seg in seg_idx:
                    clusters_to_segs[seg].append(choice if choice != -1 else max_clust_idx)

        # otherwise, keep (restore) current chain configuration
        else:
            # previously unassigned segment was rejected from making a new cluster (should never happen)
            if choice == -1 and cur_clust == -1:
                breakpoint()

            # we proposed moving a single segment
            if not move_clust:
                # previously assigned segment was rejected from joining an existing cluster
                if cur_clust != -1:
                    if cur_clust not in clust_counts:
                        clust_counts[cur_clust] = n_move
                        clust_sums[cur_clust] = np.r_[B_a, B_b]
                        clust_members[cur_clust] = set(seg_idx)
                    else:
                        clust_counts[cur_clust] += n_move 
                        clust_sums[cur_clust] += np.r_[B_a, B_b]
                        clust_members[cur_clust].update(set(seg_idx))
            
                    S.iloc[seg_idx, clust_col] = cur_clust

                # if a previously unassigned segment was rejected from joining an existing cluster,
                # we don't need to do anything: it remains unassigned

            # we proposed moving a whole cluster
            else:
                clust_counts[cl_idx] = n_move
                clust_sums[cl_idx] = np.r_[B_a, B_b]
                clust_members[cl_idx] = set(seg_idx)

                S.iloc[seg_idx, clust_col] = cl_idx

            # track cluster assignment for segment(s)
            if burned_in:
                for seg in seg_idx:
                    clusters_to_segs[seg].append(cur_clust if not move_clust else cl_idx)

        # track global state of cluster assignments
        # on average, each segment will have been reassigned every n_seg/(n_clust/2) iterations
        if burned_in and n_it - n_it_last > len(S)/(len(clust_counts)*2):
            segs_to_clusters.append(S["clust"].copy())
            n_it_last = n_it

        n_it += 1

    _, segs_to_clusters = np.unique(np.r_[segs_to_clusters], return_inverse = True)
    return segs_to_clusters.reshape([-1, len(S)])

# map trace of segment cluster assignments to the SNPs within
def map_seg_clust_assignments_to_SNPs(segs_to_clusters, S):
    st_col = S.columns.get_loc("SNP_st")
    en_col = S.columns.get_loc("SNP_en")
    snps_to_clusters = np.zeros((segs_to_clusters.shape[0], S.iloc[-1, en_col]), dtype = int)
    for i, seg_assign in enumerate(segs_to_clusters):
        for j, seg in enumerate(seg_assign):
            snps_to_clusters[i, S.iloc[j, st_col]:S.iloc[j, en_col]] = seg

    return snps_to_clusters

#
# plot

#from glasbey import glasbey

colors = mpl.cm.get_cmap("tab10").colors

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

r = np.linspace(0.4, 1, 1000)
plt.figure(3); plt.clf()
plts = []
for i, clust_idx in enumerate(S["clust"].value_counts().index[0:20]):
    if clust_idx == -1:
        continue
    plts.append(plt.plot(r, s.beta.pdf(r, S.loc[S["clust"] == clust_idx, "min"].sum(), S.loc[S["clust"] == clust_idx, "maj"].sum()), color = colors[i % len(colors)])[0])

plt.legend(plts, S["clust"].value_counts().index[0:20])

# plot beta dists. for segments, colored by cluster assignment

r = np.linspace(0.4, 1, 1000)
plt.figure(4); plt.clf()
plts = []
for i, clust_idx in enumerate(S["clust"].value_counts().index[0:20]):
    if clust_idx == -1:
        continue
    for _, mn, mj in S.loc[S["clust"] == clust_idx, ["min", "maj"]].itertuples():
        plt.plot(r, s.beta.pdf(r, mn, mj), color = colors[i % len(colors)], alpha = 0.3)


# probabilistic assignment of segments to clusters

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

# use cluster mean for each segment

s2c = np.r_[segs_to_clusters]

f1 = plt.figure(21); plt.clf()
ax = plt.gca()
ax.set_xlim([0, S["end_gp"].max()])
ax.set_ylim([0, 1])

for seg_assignments in s2c:
    S_a = npg.aggregate(seg_assignments, S["min"])
    S_b = npg.aggregate(seg_assignments, S["maj"])

    for i, clust_idx in enumerate(seg_assignments):
        r = S.iloc[i]
        ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], S_a[clust_idx] + 1, S_b[clust_idx] + 1)
        ax.add_patch(
          mpl.patches.Rectangle(
            (r["start_gp"], ci_lo),
            r["end_gp"] - r["start_gp"],
            ci_hi - ci_lo,
            facecolor = colors[clust_idx % len(colors)],
            fill = True,
            zorder = 1000,
            alpha = 1/10
          )
        )

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
