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

allelic_segs = pd.read_pickle("exome/allelic_segs.pickle")

all_segs = []

for _, H in allelic_segs.dropna(subset = ["results"]).iterrows():
    r = H["results"]
    
    # set phasing orientation back to original
    r.P["MAJ_COUNT"] = pd.concat([r.P.loc[r.P["aidx"], "ALT_COUNT"], r.P.loc[~r.P["aidx"], "REF_COUNT"]])
    r.P["MIN_COUNT"] = pd.concat([r.P.loc[r.P["aidx"], "REF_COUNT"], r.P.loc[~r.P["aidx"], "ALT_COUNT"]])

    for bpl, pil in zip(r.breakpoint_list, r.phase_interval_list):
        bps = np.r_[bpl]; bps = np.c_[bps[:-1], bps[1:]]
        pss = np.array(list(pil.keys()))

        # set phasing for this iteration
        for st, en in pss:
            r.flip_hap(st, en)

        # get major/minor sums for each segment
        for st, en in bps:
            all_segs.append([
              st, en,
              r.P.loc[st, "chr"], r.P.loc[st, "pos"], r.P.loc[en, "pos"],
              r.P.iloc[st:en, r.min_idx].sum(),
              r.P.iloc[st:en, r.maj_idx].sum()
            ])

        # reset phasing
        for st, en in pss:
            r.flip_hap(st, en)

S = pd.DataFrame(all_segs, columns = ["SNP_st", "SNP_en", "chr", "start", "end", "min", "maj"])
S["clust"] = np.r_[0:len(S)]
S["lik"] = ss.betaln(S.loc[:, "min"] + 1, S.loc[:, "maj"] + 1)
clust_col = S.columns.get_loc("clust")

clusts = { k : v for k, v in zip(range(0, len(S)),
  np.hstack([
    np.ones([len(S), 1]),
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

for _ in range(0, len(S)):
    i = np.random.choice(len(S)) # pick a segment at random

    min_c = S.loc[i, "min"]
    maj_c = S.loc[i, "maj"]

    old_idx = S.iat[i, clust_col]

    q = np.random.rand() < alpha/(N - 1 + alpha)

    # make a new cluster
    if q == 1:
        # update counts
        old_count = clusts[old_idx][0]
        clust_counts[old_count] -= 1
        clust_counts_map[old_count].remove(old_idx)
        if clust_counts[old_count] == 0:
            clust_counts.__delitem__(old_count)
            clust_counts_map.__delitem__(old_count)
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
            clusts.__delitem__(old_idx)

        clusts[max_clust_idx] = [1, min_c, maj_c]
        S.iat[i, clust_col] = max_clust_idx
        max_clust_idx += 1

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
            print("self-join")
            continue

        c = clusts[j]

        # accept proposal via Metropolis
        # A+B,C -> A,B+C
        # A+B is likelihood of current cluster B is part of
        AB = ss.betaln(clusts[old_idx][1] + 1, clusts[old_idx][2] + 1)
        # C is likelihood of target cluster pre-join
        C = ss.betaln(c[1] + 1, c[2] + 1)
        # A is likelihood cluster B is part of minus B
        A = ss.betaln(clusts[old_idx][1] - min_c + 1, clusts[old_idx][2] - maj_c + 1)
        # B+C is likelihood of target cluster post-join
        BC = ss.betaln(c[1] + min_c + 1, c[2] + maj_c + 1)

        ML_join = A + BC
        ML_split = AB + C

        if np.log(np.random.rand()) < np.minimum(0, ML_join - ML_split):
            clusts[j][0] += 1
            clusts[j][1] += min_c
            clusts[j][2] += maj_c

            S.iat[i, clust_col] = j

            # update counts
            old_count = clusts[old_idx][0]
            clust_counts[old_count] -= 1
            clust_counts_map[old_count].remove(old_idx)
            if clust_counts[old_count] == 0:
                clust_counts.__delitem__(old_count)
                clust_counts_map.__delitem__(old_count)
            new_count = clusts[j][0]
            if new_count not in clust_counts:
                clust_counts[new_count] = 0
                clust_counts_map[new_count] = sc.SortedSet()
            clust_counts[new_count] += 1
            clust_counts_map[new_count].add(j)

            # remove from old cluster
            clusts[old_idx][0] -= 1
            clusts[old_idx][1] -= min_c
            clusts[old_idx][2] -= maj_c
            if old_count == 0:
                clusts.__delitem__(old_idx)
