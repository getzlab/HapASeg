import colorama
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.pickle")

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
    for bp_samp, pi_samp in zip(r.breakpoint_list, r.phase_interval_list):
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
              r.P.iloc[st:en, min_idx].sum(),
              r.P.iloc[st:en, maj_idx].sum()
            ])

        # flip everything back
        for st, en in pi_samp.intervals():
            # TODO: can replace with flip_hap()?
            x = r.P.iloc[st:en, maj_idx].copy()
            r.P.iloc[st:en, maj_idx] = r.P.iloc[st:en, min_idx]
            r.P.iloc[st:en, min_idx] = x

S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj"])
S = S.drop_duplicates()
S["clust"] = -1 # initially, all segments are unassigned
clust_col = S.columns.get_loc("clust")
min_col = S.columns.get_loc("min")
maj_col = S.columns.get_loc("maj")
S.iloc[0, clust_col] = 0 # first segment is assigned to cluster 0
S["lik"] = ss.betaln(S.loc[:, "min"] + 1, S.loc[:, "maj"] + 1)

n_assigned = 1

clust_counts = sc.SortedDict({ 0 : 1 })
clust_sums = sc.SortedDict({ -1 : np.r_[0, 0], 0 : np.r_[S.loc[0, "min"], S.loc[0, "maj"]]})

for n_it in range(0, 10*len(S)):
    if not n_it % 1000:
        print(S["clust"].value_counts().value_counts().sort_index())

    i = np.random.choice(len(S)) # pick a segment at random

    # if segment was already assigned to a cluster, unassign it
    cur_clust = S.iat[i, clust_col]
    if cur_clust != -1:
        clust_counts[cur_clust] -= 1
        if clust_counts[cur_clust] == 0:
            del clust_counts[cur_clust]
        S.iat[i, clust_col] = -1

        clust_sums[cur_clust] -= np.r_[S.iat[i, min_col], S.iat[i, maj_col]]
        
        n_assigned -= 1

    # choose to join a cluster or make a new one
#    p_0 = np.zeros(len(clust_counts), dtype = np.int) # this could be slow to allocate each time
#    idx = np.array(clust_counts.keys())
#    p_0[idx] = np.array(clust_counts.values())
    choice_idx = np.random.choice(
      np.r_[0:(len(clust_counts) + 1)],
      p = np.r_[np.r_[clust_counts.values()]/(n_assigned + alpha), alpha/(n_assigned + alpha)]
    )
    choice = np.r_[clust_counts.keys(), -1][choice_idx]

    # propose to join a cluster
    if choice != -1:
        # accept proposal via Metropolis
        # A+B,C -> A,B+C
        # A+B is likelihood of current cluster B is part of
        AB = ss.betaln(
          clust_sums[cur_clust][0] + S.iat[i, min_col] + 1,
          clust_sums[cur_clust][1] + S.iat[i, maj_col] + 1
        )
        # C is likelihood of target cluster pre-join
        C = ss.betaln(
          clust_sums[choice][0] + 1,
          clust_sums[choice][1] + 1 
        ) 
        # A is likelihood cluster B is part of, minus B
        A = ss.betaln(
          clust_sums[cur_clust][0] + 1,
          clust_sums[cur_clust][1] + 1
        )
        # B+C is likelihood of target cluster post-join
        BC = ss.betaln(
          clust_sums[choice][0] + S.iat[i, min_col] + 1,
          clust_sums[choice][1] + S.iat[i, maj_col] + 1
        )

        ML_join = A + BC
        ML_split = AB + C

        # accept proposal to join
        if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split):
            clust_counts[choice] += 1
            clust_sums[choice] += np.r_[S.iat[i, min_col], S.iat[i, maj_col]]
            S.iat[i, clust_col] = choice

        # otherwise, keep where it is
        else:
            # if it was previously assigned to a cluster, keep it there
            if cur_clust != -1 and cur_clust in clust_counts:
                clust_counts[cur_clust] += 1
                clust_sums[cur_clust] += np.r_[S.iat[i, min_col], S.iat[i, maj_col]]
                S.iat[i, clust_col] = cur_clust

            # otherwise, assign it to a new cluster
            else: 
                new_clust_idx = len(clust_counts)
                while new_clust_idx in clust_counts:
                    new_clust_idx += 1
                clust_counts[new_clust_idx] = 1
                S.iat[i, clust_col] = new_clust_idx

                clust_sums[new_clust_idx] = np.r_[S.iat[i, min_col], S.iat[i, maj_col]]

    # add to a new cluster
    else:
        print("new!")
        new_clust_idx = len(clust_counts)
        while new_clust_idx in clust_counts:
            new_clust_idx += 1
        clust_counts[new_clust_idx] = 1
        S.iat[i, clust_col] = new_clust_idx

        clust_sums[new_clust_idx] = np.r_[S.iat[i, min_col], S.iat[i, maj_col]]

    n_assigned += 1




    q = np.random.rand() < alpha/(n_assigned - 1 + alpha)


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
