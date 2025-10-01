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


def exclude_region_from_bed(Cov, exclude_region_bed_path):
    # read in the exclude region bed
    exclude_region_df = pd.read_csv(exclude_region_bed_path, sep="\t", header=None)
    # ensure bed contains chr, start, end columns
    if len(exclude_region_df.columns) < 3:
        raise ValueError("Exclude region bed must contain chr, start, end columns")
    exclude_region_df.columns = ["chr", "start", "end"]
    exclude_region_df["start"] = exclude_region_df["start"].astype(int)
    exclude_region_df["end"] = exclude_region_df["end"].astype(int) 

    # perform intersection between exclude region bed and coverage bins
    # create a mask for rows to keep (those that don't overlap with any excluded region)
    keep_mask = np.ones(len(Cov), dtype=bool)
    
    for chr_name in Cov["chr"].unique():
        # get coverage bins for this chromosome
        cov_chr_mask = Cov["chr"] == chr_name
        cov_chr = Cov.loc[cov_chr_mask]
        
        # get exclude regions for this chromosome
        exclude_chr = exclude_region_df[exclude_region_df["chr"] == chr_name]
        
        if len(exclude_chr) == 0:
            continue
        
        # for each coverage bin, check if it overlaps with any exclude region
        for idx in cov_chr.index:
            cov_start = Cov.loc[idx, "start"]
            cov_end = Cov.loc[idx, "end"]
            
            # check for overlap: bins overlap if cov_start < exclude_end AND cov_end > exclude_start
            overlaps = ((cov_start < exclude_chr["end"]) & (cov_end > exclude_chr["start"])).any()
            
            if overlaps:
                keep_mask[idx] = False

    print(f"Excluding {len(Cov[~keep_mask])} bins from {len(Cov)} total bins using exclusion region bed file: {exclude_region_bed}")
    
    return Cov[keep_mask]


def exclude_region_from_segfile(segfile_df, exclude_region_bed_path):
    # read in the exclude region bed
    exclude_region_df = pd.read_csv(exclude_region_bed_path, sep="\t", header=None)
    # ensure bed contains chr, start, end columns
    if len(exclude_region_df.columns) < 3:
        raise ValueError("Exclude region bed must contain chr, start, end columns")
    exclude_region_df.columns = ["chr", "start", "end"]
    exclude_region_df["start"] = exclude_region_df["start"].astype(int)
    exclude_region_df["end"] = exclude_region_df["end"].astype(int) 

    # perform intersection between exclude region bed and segfile segments
    # segments completely contained within excluded regions will be removed
    # segments partially overlapping excluded regions will be trimmed
    # segments with excluded regions in the middle will be split
    
    result_rows = []
    
    for chr_name in segfile_df["Chromosome"].unique():
        # get segments for this chromosome
        seg_chr_mask = segfile_df["Chromosome"] == chr_name
        seg_chr = segfile_df.loc[seg_chr_mask].copy()
        
        # get exclude regions for this chromosome
        exclude_chr = exclude_region_df[exclude_region_df["chr"] == chr_name]
        
        if len(exclude_chr) == 0:
            result_rows.append(seg_chr)
            continue
        
        # for each segment, check if it overlaps with any exclude region
        for idx in seg_chr.index:
            seg_start = segfile_df.loc[idx, "Start.bp"]
            seg_end = segfile_df.loc[idx, "End.bp"]
            
            # check if segment is completely contained within any exclude region
            # completely contained: exclude_start <= seg_start AND seg_end <= exclude_end
            completely_contained = ((exclude_chr["start"] <= seg_start) & (seg_end <= exclude_chr["end"])).any()
            
            if completely_contained:
                # skip this segment entirely
                continue
            
            # check if segment has no overlap with any exclude region
            # no overlap: seg_end <= exclude_start OR seg_start >= exclude_end for all exclude regions
            has_any_overlap = ((seg_start < exclude_chr["end"]) & (seg_end > exclude_chr["start"])).any()
            
            if not has_any_overlap:
                # no overlap, keep the segment as is
                result_rows.append(pd.DataFrame([segfile_df.loc[idx]]))
                continue
            
            # handle partial overlaps - may result in multiple segments if excluded region is in the middle
            # collect all exclude regions that overlap with this segment
            overlapping_excludes = exclude_chr[
                (seg_start < exclude_chr["end"]) & (seg_end > exclude_chr["start"])
            ].sort_values("start")
            
            # create intervals by removing excluded regions
            intervals = [(seg_start, seg_end)]
            
            for _, exclude_row in overlapping_excludes.iterrows():
                exclude_start = exclude_row["start"]
                exclude_end = exclude_row["end"]
                
                new_intervals = []
                for interval_start, interval_end in intervals:
                    # check if exclude region overlaps with this interval
                    if interval_start < exclude_end and interval_end > exclude_start:
                        # there is overlap - split the interval
                        if interval_start < exclude_start:
                            # keep the left portion
                            new_intervals.append((interval_start, exclude_start))
                        if interval_end > exclude_end:
                            # keep the right portion
                            new_intervals.append((exclude_end, interval_end))
                    else:
                        # no overlap, keep the interval as is
                        new_intervals.append((interval_start, interval_end))
                
                intervals = new_intervals
            
            # create a segment for each remaining interval
            for interval_start, interval_end in intervals:
                if interval_start < interval_end:
                    row = segfile_df.loc[idx].copy()
                    row["Start.bp"] = interval_start
                    row["End.bp"] = interval_end
                    result_rows.append(pd.DataFrame([row]))

    if len(result_rows) == 0:
        print("Warning: All segments excluded using exclusion region bed file")
        return segfile_df.iloc[0:0]  # return empty dataframe with same structure
    
    result_df = pd.concat(result_rows, ignore_index=True)
    
    print(f"Excluding/trimming {len(segfile_df) - len(result_df)} segments from {len(segfile_df)} total segments using exclusion region bed file")
    
    return result_df