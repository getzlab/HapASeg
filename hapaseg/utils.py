import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_chrmap = dict(zip(["chr" + str(x) for x in list(range(1, 23)) + ["X", "Y"]], range(1, 25)))

def parse_cytoband(cytoband):
    # some cytoband files have a header, some don't; we need to check
    has_header = False
    with open(cytoband, "r") as f:
        if f.readline().startswith("chr\t"):
            has_header = True

    cband = pd.read_csv(cytoband, sep = "\t", names = ["chr", "start", "end", "band", "stain"] if not has_header else None)
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
    yl_0 = plt.ylim()[0]
    yl_1 = plt.ylim()[1]
    chr_ends = chrbdy.loc[1::2, "end"].cumsum()
    for end in chr_ends[:-1]:
        plt.axvline(end, color = 'k', zorder=100)
    for st, en in np.c_[chr_ends[:-1:2], chr_ends[1::2]]:
        plt.fill_between([st, en], yl_0, yl_1, color = [0.9, 0.9, 0.9], zorder = 0)
    plt.ylim([yl_0, yl_1])

    # plot centromere locations
    for cent in (np.c_[chrbdy.loc[1::2, "start"], chrbdy.loc[::2, "end"]] + np.c_[np.r_[0, chr_ends[:-1]]]).ravel():
        plt.axvline(cent, color = 'k', linestyle = ":", linewidth = 0.5, zorder=100)

    # add xticks
    xt = (np.r_[0, chr_ends[:-1]] + chr_ends)/2
    xtl = chrbdy.loc[chr_ends.index, "chr"]
    plt.xticks(xt, xtl)

    # alternately stagger xticks 
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks()[1::2]:
        t.set_pad(15)

    ax.tick_params(axis = "x", length = 0)


def plot_chrbdy_rm_chrs(cytoband_file, chr_rm_list = []):
    chrbdy = parse_cytoband(cytoband_file)

    # plot chromosome boundaries
    yl_0 = plt.ylim()[0]
    yl_1 = plt.ylim()[1]
    chr_ends = chrbdy.loc[1::2, "end"].cumsum()

    for end in chr_ends[:-1]:
        plt.axvline(end, color = 'k')
    for i, (st, en) in enumerate(np.c_[chr_ends[:-1:2], chr_ends[1::2]]):
        if 2*(i+1) in chr_rm_list:
            continue
        plt.fill_between([st, en], yl_0, yl_1, color = [0.9, 0.9, 0.9], zorder = 0)
    plt.ylim([yl_0, yl_1])

    # plot centromere locations
    for i, cent in enumerate((np.c_[chrbdy.loc[1::2, "start"], chrbdy.loc[::2, "end"]] + np.c_[np.r_[0, chr_ends[:-1]]]).ravel()):
        if int(i / 2) + 1  in chr_rm_list:
            continue
        plt.axvline(cent, color = 'k', linestyle = ":", linewidth = 0.5)

    # add xticks
    xt = (np.r_[0, chr_ends[:-1]] + chr_ends)/2
    xtl = chrbdy.loc[chr_ends.index, "chr"]
    # add crossout for dropped chrs
    xtl = ['\u0336'.join(str(c)) + '\u0336' if c in chr_rm_list else str(c) for c in xtl]
    plt.xticks(xt, xtl)

    # alternately stagger xticks 
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks()[1::2]:
        t.set_pad(15)

    # fill missing chrs with grey rectangles
    cp_chr_ends = chr_ends.copy()
    cp_chr_ends.index = range(1,25)
    cp_chr_ends[0] = 0
    d=0.85
    for i in chr_rm_list:
        #plt.fill_between([cp_chr_ends[i-1], cp_chr_ends[i]], yl_0, yl_1, color = [d, d, d], zorder = 0)
        plt.fill_between([cp_chr_ends[i-1], cp_chr_ends[i]], yl_0, yl_1, color = [1, 1, 1], hatch='////', edgecolor=[d,d,d], zorder =0)
    ax.tick_params(axis = "x", length = 0)

