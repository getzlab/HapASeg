import numpy as np
import pandas as pd

def parse_cytoband(cytoband):
    cband = pd.read_csv(cytoband, sep = "\t", names = ["chr", "start", "end", "band", "stain"])
    cband["chr"] = cband["chr"].apply(lambda x : _chrmap[x])

    chrs = cband["chr"].unique()
    ints = dict(zip(chrs, [{0} for _ in range(0, len(chrs))]))
    last_end = None
    last_stain = None
    last_chrom = None
    for _, chrom, start, end, _, stain in cband.itertuples():
        if start == 0:
            if last_end is not None:
                ints[last_chrom].add(last_end)
        if stain == "acen" and last_stain != "acen":
            ints[chrom].add(start)
        if stain != "acen" and last_stain == "acen":
            ints[chrom].add(start)
        
        last_end = end
        last_stain = stain
        last_chrom = chrom
    ints[chrom].add(end)

    CI = np.full([len(ints), 4], 0)
    for c in chrs:
        CI[c - 1, :] = sorted(ints[c])

    return pd.DataFrame(
      np.c_[np.tile(np.c_[np.r_[1:25]], [1, 2]).reshape(-1, 1), CI.reshape(-1, 2)],
      columns = ["chr", "start", "end"]
    )

def plot_chrbdy(cytoband_file):
    chrbdy = parse_cytoband(cytoband_file)

    # plot chromosome boundaries
    chr_ends = chrbdy.loc[1::2, "end"].cumsum()
    for end in chr_ends[:-1]:
        plt.axvline(end, color = 'k')
    for st, en in np.c_[chr_ends[:-1:2], chr_ends[1::2]]:
        plt.fill_between([st, en], 0, 1, color = [0.9, 0.9, 0.9], zorder = 0)

    # plot centromere locations
    for cent in (np.c_[chrbdy.loc[1::2, "start"], chrbdy.loc[::2, "end"]] + np.c_[np.r_[0, chr_ends[:-1]]]).ravel():
        plt.axvline(cent, color = 'k', linestyle = ":", linewidth = 0.5)

    # add xticks
    xt = (np.r_[0, chr_ends[:-1]] + chr_ends)/2
    xtl = chrbdy.loc[chr_ends.index, "chr"]
    plt.xticks(xt, xtl)

    # alternately stagger xticks 
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks()[1::2]:
        t.set_pad(15)

    ax.tick_params(axis = "x", length = 0)
