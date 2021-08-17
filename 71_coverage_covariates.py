import pandas as pd
import pyfaidx
from capy import mut

#
# replication timing

F = pd.read_csv("/mnt/j/proj/cnv/20201018_hapseg2/covars/GSE137764_H1_GaussiansGSE137764_mooth_scaled_autosome.mat", sep = "\t", header = None).T.rename(columns = { 0 : "chr", 1 : "start", 2 : "end" })
F.iloc[:, 3:] = F.loc[:, 3:].astype(float)
F.loc[:, ["start", "end"]] = F.loc[:, ["start", "end"]].astype(int)
F["chr"] = mut.convert_chr(F["chr"])
F.to_pickle("covars/GSE137764_H1.pickle")

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
