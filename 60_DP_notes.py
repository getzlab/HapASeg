import colorama
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import ncls
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

from capy import seq

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.auto_ref_correct.pickle")

all_segs = []

maj_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("MAJ_COUNT")
min_idx = allelic_segs["results"].iloc[0].P.columns.get_loc("MIN_COUNT")

for _, H in allelic_segs.dropna(subset = ["results"]).iterrows():
    r = H["results"]
    
    # set phasing orientation back to original
    for st, en in r.F.intervals():
        # code excised from flip_hap
        x = r.P.iloc[st:en, maj_idx].copy()
        r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
        r.P.iloc[st:en, min_idx] = x

    #for bpl, pil in zip(r.breakpoint_list, r.phase_interval_list):
    for bp_samp, pi_samp, inc_samp in zip(r.breakpoint_list, r.phase_interval_list, r.include):
        # flip everything according to sample
        for st, en in pi_samp.intervals():
            x = r.P.iloc[st:en, maj_idx].copy()
            r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
            r.P.iloc[st:en, min_idx] = x

        bpl = np.array(bp_samp); bpl = np.c_[bpl[0:-1], bpl[1:]]

        # get major/minor sums for each segment
        for st, en in bpl:
            all_segs.append([
              st, en,
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

S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj"])

# aggregate duplicate segments (effectively weighting by occurrence)
# XXX: how "effective" is this? weighting probability of choosing cluster by number
#      of occurrences != weighting probability of choosing by increasing counts
S = S.groupby(["chr", "start", "end"])[["min", "maj"]].sum().join(
  S.drop(columns = ["min", "maj"]).set_index(["chr", "start", "end"]),
  how = "left"
).drop_duplicates().reset_index()

# construct overlap matrix
S["start_gp"] = seq.chrpos2gpos(S["chr"], S["start"])
S["end_gp"] = seq.chrpos2gpos(S["chr"], S["end"])

intervals = ncls.NCLS(S["start_gp"], S["end_gp"], S.index)
x, y = intervals.all_overlaps_both(S["start_gp"].values, S["end_gp"].values, S.index.values)
z = np.zeros_like(x)

O = sp.dok_matrix(sp.coo_matrix((ss.betaln(mn.sum(1) + 1, mj.sum(1) + 1) - ss.betaln(mn + 1, mj + 1).sum(1), (x, y))))
O_d = O.todense() # TODO: dense lookup are much faster; we might want to store this as a block matrix for space savings, since we don't expect elements far from the diagonal

# other fields of S
S["clust"] = -1 # initially, all segments are unassigned
clust_col = S.columns.get_loc("clust")
min_col = S.columns.get_loc("min")
maj_col = S.columns.get_loc("maj")
S.iloc[0, clust_col] = 0 # first segment is assigned to cluster 0
S["lik"] = ss.betaln(S.loc[:, "min"] + 1, S.loc[:, "maj"] + 1)

n_assigned = 1

clust_counts = sc.SortedDict({ 0 : 1 })
clust_sums = sc.SortedDict({ -1 : np.r_[0, 0], 0 : np.r_[S.loc[0, "min"], S.loc[0, "maj"]]})
clust_members = sc.SortedDict({ 0 : set({0}) })

for n_it in range(0, 10*len(S)):
    if not n_it % 1000:
        print(S["clust"].value_counts().drop(-1).value_counts().sort_index())

    #
    # pick either a segment or a cluster at random (50:50 prob.)

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
            else:
                clust_sums[cur_clust] -= np.r_[S.iloc[seg_idx, min_col], S.iloc[seg_idx, maj_col]]

            S.iloc[seg_idx, clust_col] = -1

            clust_members[cur_clust] -= set(seg_idx)
            
            n_assigned -= 1

    # pick a cluster at random
    else:
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
        S.iloc[seg_idx, clust_col] = -1

        # NOTE: in the previous code, this accidentally was -=, not =
        # leaving comment here for posterity
        #clust_sums[cl_idx] = np.r_[0, 0]
        clust_members[cur_clust] = set()
        
        n_assigned -= n_move 

    # choose to join a cluster or make a new one
    # probabilities determined by similarity of segment/cluster to existing ones

    # B is segment/cluster to move
    # A is cluster B is currently part of
    # C is all possible clusters to move to
    A_a = clust_sums[cur_clust][0] if cur_clust in clust_sums else 0
    A_b = clust_sums[cur_clust][1] if cur_clust in clust_sums else 0
    B_a = S.iloc[seg_idx, min_col].sum()
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
    MLs_max = np.max(MLs)
    p = np.exp(MLs - MLs_max)/np.exp(MLs - MLs_max).sum()

    # choose to join a cluster or make a new one (choice_idx = 0) 
    choice_idx = np.random.choice(
      np.r_[0:(len(clust_counts) + 1)],
      p = np.exp(MLs - MLs_max)/np.exp(MLs - MLs_max).sum()
    )
    choice = np.r_[-1, clust_counts.keys()][choice_idx]

    # propose to join a cluster
    if choice != -1:
        C_a = clust_sums[choice][0]
        C_b = clust_sums[choice][1]

        # compute prior odds ratio for all overlaps between segments
        p_odd = O_d[np.array(list(clust_members[choice]))][:, seg_idx].sum()

        # accept proposal via Metropolis
        # A+B,C -> A,B+C
        # C is likelihood of target cluster pre-join
        C = ss.betaln(C_a + 1, C_b + 1) 
        # B+C is likelihood of target cluster post-join
        BC = ss.betaln(C_a + B_a + 1, C_b + B_b + 1)

        ML_join = A + BC
        ML_split = AB + C

        # TODO: add proposal ratio here, since it is no longer symmetric

        # accept proposal to join
        if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split + p_odd):
            clust_counts[choice] += n_move 
            clust_sums[choice] += np.r_[B_a, B_b]
            S.iloc[seg_idx, clust_col] = choice

            clust_members[choice].update(set(seg_idx))

        # otherwise, keep where it is
        else:
            # if it was previously assigned to a cluster, keep it there (only applicable to single segments)
            if cur_clust != -1 and cur_clust in clust_counts.keys():
                clust_counts[cur_clust] += n_move
                clust_sums[cur_clust] += np.r_[B_a, B_b]
                S.iloc[seg_idx, clust_col] = cur_clust

                clust_members[cur_clust].update(set(seg_idx))

            # otherwise, assign it to a new cluster
            else: 
                new_clust_idx = len(clust_counts)
                while new_clust_idx in clust_counts:
                    new_clust_idx += 1
                clust_counts[new_clust_idx] = n_move
                S.iloc[seg_idx, clust_col] = new_clust_idx

                clust_sums[new_clust_idx] = np.r_[B_a, B_b]
                clust_members[new_clust_idx] = set(seg_idx)

    # add to a new cluster
    else:
        #print("new!")
        new_clust_idx = len(clust_counts)
        while new_clust_idx in clust_counts:
            new_clust_idx += 1
        clust_counts[new_clust_idx] = n_move
        S.iloc[seg_idx, clust_col] = new_clust_idx

        clust_sums[new_clust_idx] = np.r_[B_a, B_b]
        clust_members[new_clust_idx] = set(seg_idx)

    n_assigned += n_move

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
    for _, r in S.loc[S["clust"] == clust_idx].iterrows():
        ci_lo, med, ci_hi = s.beta.ppf([0.05, 0.5, 0.95], r["min"] + 1, r["maj"] + 1)
        axs[i].add_patch(mpl.patches.Rectangle((r["start_gp"], ci_lo), r["end_gp"] - r["start_gp"], ci_hi - ci_lo, facecolor = colors[i % len(colors)], fill = True, alpha = 0.9, zorder = 1000))
        ax.add_patch(mpl.patches.Rectangle((r["start_gp"], ci_lo), r["end_gp"] - r["start_gp"], ci_hi - ci_lo, facecolor = colors[i % len(colors)], fill = True, alpha = 0.1, zorder = 1000))

# plot beta dists. for clusters that ought to be merged

r = np.linspace(0.495, 0.53, 10000)
plt.figure(3); plt.clf()
plts = []
for clust_idx in S["clust"].value_counts().iloc[np.r_[0:5]].index:
    plts.append(plt.plot(r, s.beta.pdf(r, S.loc[S["clust"] == clust_idx, "min"].sum(), S.loc[S["clust"] == clust_idx, "maj"].sum()))[0])

plt.legend(plts, S["clust"].value_counts().iloc[np.r_[0:5]].index)




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
