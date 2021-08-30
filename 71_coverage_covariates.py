import liftover
import pandas as pd
import pyfaidx
import tqdm
from capy import mut

#
# replication timing

F = pd.read_csv("/mnt/j/proj/cnv/20201018_hapseg2/covars/GSE137764_H1_GaussiansGSE137764_mooth_scaled_autosome.mat", sep = "\t", header = None).T.rename(columns = { 0 : "chr", 1 : "start", 2 : "end" })
F.iloc[:, 3:] = F.loc[:, 3:].astype(float)
F.loc[:, ["start", "end"]] = F.loc[:, ["start", "end"]].astype(int)
F["chr"] = mut.convert_chr(F["chr"])
F.to_pickle("covars/GSE137764_H1.hg38.pickle")

# liftover to hg19
F["chr_start_lift"] = 0
F["chr_end_lift"] = 0
F["start_lift"] = 0
F["end_lift"] = 0
F["start_strand_lift"] = 'x'
F["end_strand_lift"] = 'x'
converter = liftover.ChainFile("/mnt/j/db/hg38/liftover/hg38ToHg19.over.chain", "hg38", "hg19")
for x in tqdm.tqdm(F.itertuples()):
    conv = converter[x.chr][x.start]
    F.loc[x.Index, ["chr_start_lift", "start_lift", "start_strand_lift"]] = (-1, -1, '?') if len(conv) == 0 else conv[0]
    conv = converter[x.chr][x.end]
    F.loc[x.Index, ["chr_end_lift", "end_lift", "end_strand_lift"]] = (-1, -1, '?') if len(conv) == 0 else conv[0]

F.to_pickle("covars/GSE137764_H1.hg19_raw_liftover.pickle")

# fix weird intervals

# flip reverse strands
idx = ((F["start_strand_lift"] == "-") & (F["end_strand_lift"] == "-")).values
st_col = F.columns.get_loc("start_lift")
en_col = F.columns.get_loc("end_lift")
F.iloc[idx, [st_col, en_col]] = F.iloc[idx, [en_col, st_col]]
F.loc[idx, ["start_strand_lift", "end_strand_lift"]] = "+"

# get length distribution
plt.figure(1); plt.clf()
plt.hist(F["end_lift"] - F["start_lift"] + 1, np.linspace(0, 100000, 100))
plt.yscale("log")

# for now, let's not bother with any funky intervals
good_idx = (F["chr_start_lift"] == F["chr_end_lift"]) & (F["start_lift"] < F["end_lift"]) & (F["end_lift"] - F["start_lift"] < 60000)

F_lift = F.loc[good_idx]
F_lift = F_lift.drop(columns = ["chr", "start", "end"]).rename(columns = { "chr_start_lift" : "chr", "start_lift" : "start", "end_lift" : "end" }).iloc[:, np.r_[16, 18, 19, 0:16]]

# remove non-primary chromosomes
F_lift["chr"] = mut.convert_chr(F_lift["chr"])
F_lift = F_lift.loc[F_lift["chr"].apply(lambda x : type(x) == int)]

F_lift = F_lift.sort_values(["chr", "start", "end"])

F_lift.to_pickle("covars/GSE137764_H1.hg19_liftover.pickle")

# check for weird intervals
bad_idx = (F["chr_start_lift"] != F["chr"]) | \
(F["chr_end_lift"] != F["chr"]) | \
(F["start_strand_lift"].notin(["+", "?"])) | \
(F["start_lift"] > F["end_lift"])

#
# GC content

B = pd.read_csv("/mnt/j/proj/cnv/20210326_coverage_collector/targets.bed", sep = "\t", header = None, names = ["chr", "start", "end"])
B["chr"] = mut.convert_chr(B["chr"])
B["gc"] = np.nan
B = B.loc[B["chr"].apply(type) == int]
B = B.loc[B["chr"] > 0]
F = pyfaidx.Fasta("/mnt/j/db/hg19/ref/hs37d5.fa")

for (i, chrm, start, end, _) in B.itertuples():
    B.iat[i, -1] = F[chrm - 1][(start - 1):end].gc

B.to_pickle("covars/GC.pickle")

