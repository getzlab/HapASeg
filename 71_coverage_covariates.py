import liftover
import numpy as np
import pandas as pd
import pyfaidx
import pyBigWig
import tqdm
from capy import mut, seq

#
# replication timing {{{

F = pd.read_csv("/mnt/j/proj/cnv/20201018_hapseg2/covars/GSE137764_H1_GaussiansGSE137764_mooth_scaled_autosome.mat", sep = "\t", header = None).T.rename(columns = { 0 : "chr", 1 : "start", 2 : "end" })
F.iloc[:, 3:] = F.loc[:, 3:].astype(float)
F.loc[:, ["start", "end"]] = F.loc[:, ["start", "end"]].astype(int)
F["chr"] = mut.convert_chr(F["chr"])
F.to_pickle("covars/GSE137764_H1.hg38.pickle")

# liftover to hg19 {{{
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

# }}}

# }}}

#
# GC content {{{

# note: this is obsolete; GC content is now computed on the fly

B = pd.read_csv("/mnt/j/proj/cnv/20210326_coverage_collector/targets.bed", sep = "\t", header = None, names = ["chr", "start", "end"])
B["chr"] = mut.convert_chr(B["chr"])
B["gc"] = np.nan
B = B.loc[B["chr"].apply(type) == int]
B = B.loc[B["chr"] > 0]
F = pyfaidx.Fasta("/mnt/j/db/hg19/ref/hs37d5.fa")

for (i, chrm, start, end, _) in B.itertuples():
    B.iat[i, -1] = F[chrm - 1][(start - 1):end].gc

B.to_pickle("covars/GC.pickle")

# }}}

#
# DNAse HS/FAIRE {{{

## DNAse {{{

bw = pyBigWig.open("covars/wgEncodeUwDnaseGm12878RawRep1.bigWig")

# WGS (2kb chunks)
clen = seq.get_chrlens()
C = []
for i, chrname in enumerate(["chr" + str(x) for x in list(range(1, 23)) + ["X", "Y"]]):
    bins = np.r_[0:clen[i]:2000, clen[i]]; bins = np.c_[bins[:-1], bins[1:]]
    tmp = pd.DataFrame({ "chr" : chrname, "start" : bins[:, 0], "end" : bins[:, 1], "DNAse" : 0 })
    for j, (st, en) in enumerate(tqdm.tqdm(bins)):
        tmp.loc[j, "DNAse"] = np.nanmean(np.r_[bw.values(chrname, st, en)])
    C.append(tmp)

# preliminary results not so great; stick with FAIRE for now

# TODO: liftover to hg38

# WES

# }}}

## FAIRE {{{

## convert bigWig to FWB

# for some reason pyBigWig can't process this file
# bw = pyBigWig.open("covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.bigwig")

# use bigWig2FWB instead
# git clone git@github.com:getzlab/bigWig2FWB.git

# figure out range of file
# bigWig2FWB/bigWig2FWB covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.bigWig covars/bwtest
# ./getmax
# -> max = 5478
# set scale factor to 11
# bigWig2FWB/bigWig2FWB covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.bigWig covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal

## WGS
from capy import fwb, mut

F = fwb.FWB("covars/wgEncodeOpenChromFaireGm12878BaseOverlapSignal.fwb");

clen = seq.get_chrlens()
C = []
for i, chrname in enumerate(["chr" + str(x) for x in list(range(1, 23)) + ["X"]]):
    bins = np.r_[0:clen[i]:2000, clen[i]]; bins = np.c_[bins[:-1], bins[1:]]
    tmp = pd.DataFrame({ "chr" : chrname, "start" : bins[:, 0], "end" : bins[:, 1], "FAIRE" : 0 })
    for j, (st, en) in enumerate(tqdm.tqdm(bins)):
        tmp.loc[j, "FAIRE"] = F.get(chrname, np.r_[st:en] + 1).mean()
    C.append(tmp)

FAIRE = pd.concat(C, ignore_index = True)
FAIRE["chr"] = mut.convert_chr(FAIRE["chr"])
FAIRE.to_pickle("covars/FAIRE_GM12878.hg19.pickle")

# smoothed version
FAIRE_smooth = FAIRE.copy()
FAIRE_smooth["FAIRE"] = np.convolve(FAIRE["FAIRE"], np.ones(5), mode = "same")/5
FAIRE_smooth.to_pickle("covars/FAIRE_GM12878.smooth5.hg19.pickle")

#
# re-process all FAIRE files using samtools
import wolf, itertools, glob, prefect

## make interval list
clen = seq.get_chrlens()
for i, chrname in enumerate(["chr" + str(x) for x in list(range(1, 23)) + ["X"]]):
    bins = np.r_[0:clen[i]:2000, clen[i]]; bins = np.c_[bins[:-1], bins[1:]]
    tmp = pd.DataFrame({ "chr" : chrname, "start" : bins[:, 0], "end" : bins[:, 1] })
    tmp.to_csv(f"FAIRE/intervals/{chrname}.bed", sep = "\t", header = None, index = False)

## define coverage workflow

class markdups(wolf.Task):
    inputs = { "bamin" }
    script = "samtools markdup ${bamin} $(basename ${bamin}).dedup.bam && samtools index *dedup.bam"
    outputs = { "bam" : "*.bam", "bai" : "*.bai" }
    docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"

intervals = glob.glob("/mnt/j/proj/cnv/20201018_hapseg2/covars/FAIRE/intervals/*.bed")

def BedCovFlow(bams, intervals):
    # mark duplicates
    mark_dups = []
    for b in bams:
        mark_dups.append(markdups(
          inputs = { "bamin" : b },
          overrides = { "bamin" : "string" },
          use_scratch_disk = True,
          scratch_disk_size = 10
        ))

    # run bedcov on all BAMs (gather)
    @prefect.task(nout = 2)
    def bl(md):
        return [m["bam"] for m in md], [m["bai"] for m in md]
    bam_list, bai_list = bl(mark_dups)

    BedCov = wolf.Task(
      name = "BedCov",
      inputs = { "intervals" : intervals, "bams" : [bam_list], "bais" : [bai_list] },
      script = """
      samtools bedcov -Q1 ${intervals} $(cat ${bams}) > coverage.bed
      """,
      outputs = { "coverage" : "coverage.bed" },
      docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"
    )
#    for b in bam_list:
#        wolf.DeleteDisk(b, BedCov["coverage"])

    # gather BedCovs
    BedCovGather = wolf.Task(
      name = "BedCovGather",
      inputs = { "beds" : [BedCov["coverage"]] },
      script = """
      cat $(cat ${beds}) | sort -k1,1V -k2,2n | \
        awk -F'\t' 'BEGIN { OFS = FS } { tot = 0; for(i = 4; i <= NF; i++) { tot += $i }; print $0, tot }' > concat.bed
      """,
      outputs = { "concat" : "concat.bed" },
    )

## run workflow 

base_url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromFaire/"

BAMs = ["wgEncodeOpenChromFaireA549AlnRep1.bam", # {{{
"wgEncodeOpenChromFaireA549AlnRep2.bam",
"wgEncodeOpenChromFaireAstrocyAlnRep1.bam",
"wgEncodeOpenChromFaireAstrocyAlnRep2.bam",
"wgEncodeOpenChromFaireColonocAlnRep1.bam",
"wgEncodeOpenChromFaireColonocAlnRep2.bam",
"wgEncodeOpenChromFaireEndometriumocAlnRep1.bam",
"wgEncodeOpenChromFaireEndometriumocAlnRep2.bam",
"wgEncodeOpenChromFaireFrontalcortexocAlnRep1.bam",
"wgEncodeOpenChromFaireFrontalcortexocAlnRep2.bam",
"wgEncodeOpenChromFaireGlioblaAlnRep1.bam",
"wgEncodeOpenChromFaireGlioblaAlnRep2.bam",
"wgEncodeOpenChromFaireGlioblaAlnRep3.bam",
"wgEncodeOpenChromFaireGm12878AlnRep1.bam",
"wgEncodeOpenChromFaireGm12878AlnRep2.bam",
"wgEncodeOpenChromFaireGm12878AlnRep3.bam",
"wgEncodeOpenChromFaireGm12891AlnRep1.bam",
"wgEncodeOpenChromFaireGm12891AlnRep2.bam",
"wgEncodeOpenChromFaireGm12892AlnRep1.bam",
"wgEncodeOpenChromFaireGm12892AlnRep2.bam",
"wgEncodeOpenChromFaireGm18507AlnRep1.bam",
"wgEncodeOpenChromFaireGm18507AlnRep2.bam",
"wgEncodeOpenChromFaireGm19239AlnRep1.bam",
"wgEncodeOpenChromFaireGm19239AlnRep2.bam",
"wgEncodeOpenChromFaireH1hescAlnRep1.bam",
"wgEncodeOpenChromFaireH1hescAlnRep2.bam",
"wgEncodeOpenChromFaireHelas3AlnRep1.bam",
"wgEncodeOpenChromFaireHelas3AlnRep2.bam",
"wgEncodeOpenChromFaireHelas3Ifna4hAlnRep1.bam",
"wgEncodeOpenChromFaireHelas3Ifna4hAlnRep2.bam",
"wgEncodeOpenChromFaireHelas3Ifng4hAlnRep1.bam",
"wgEncodeOpenChromFaireHelas3Ifng4hAlnRep2.bam",
"wgEncodeOpenChromFaireHepg2AlnRep1.bam",
"wgEncodeOpenChromFaireHepg2AlnRep2.bam",
"wgEncodeOpenChromFaireHepg2AlnRep3.bam",
"wgEncodeOpenChromFaireHtr8AlnRep1.bam",
"wgEncodeOpenChromFaireHtr8AlnRep2.bam",
"wgEncodeOpenChromFaireHuvecAlnRep1.bam",
"wgEncodeOpenChromFaireHuvecAlnRep2.bam",
"wgEncodeOpenChromFaireK562AlnRep1.bam",
"wgEncodeOpenChromFaireK562AlnRep2.bam",
"wgEncodeOpenChromFaireK562NabutAlnRep1.bam",
"wgEncodeOpenChromFaireK562NabutAlnRep2.bam",
"wgEncodeOpenChromFaireK562OhureaAlnRep1.bam",
"wgEncodeOpenChromFaireK562OhureaAlnRep2.bam",
"wgEncodeOpenChromFaireKidneyocAlnRep1.bam",
"wgEncodeOpenChromFaireKidneyocAlnRep2.bam",
"wgEncodeOpenChromFaireMcf7Est10nm30mAlnRep1.bam",
"wgEncodeOpenChromFaireMcf7Est10nm30mAlnRep2.bam",
"wgEncodeOpenChromFaireMcf7HypoxlacAlnRep1.bam",
"wgEncodeOpenChromFaireMcf7HypoxlacAlnRep2.bam",
"wgEncodeOpenChromFaireMcf7VehAlnRep1.bam",
"wgEncodeOpenChromFaireMcf7VehAlnRep2.bam",
"wgEncodeOpenChromFaireMedulloAlnRep1.bam",
"wgEncodeOpenChromFaireMedulloAlnRep2.bam",
"wgEncodeOpenChromFaireMrta2041AlnRep1.bam",
"wgEncodeOpenChromFaireMrta2041AlnRep2.bam",
"wgEncodeOpenChromFaireMrtg4016AlnRep1.bam",
"wgEncodeOpenChromFaireMrtg4016AlnRep2.bam",
"wgEncodeOpenChromFaireMrtttc549AlnRep1.bam",
"wgEncodeOpenChromFaireMrtttc549AlnRep2.bam",
"wgEncodeOpenChromFaireNhaAlnRep1.bam",
"wgEncodeOpenChromFaireNhaAlnRep2.bam",
"wgEncodeOpenChromFaireNhbeAlnRep1.bam",
"wgEncodeOpenChromFaireNhbeAlnRep2.bam",
"wgEncodeOpenChromFaireNhekAlnRep1.bam",
"wgEncodeOpenChromFaireNhekAlnRep2.bam",
"wgEncodeOpenChromFairePancreasocAlnRep1.bam",
"wgEncodeOpenChromFairePancreasocAlnRep2.bam",
"wgEncodeOpenChromFairePanisletsAlnRep1.bam",
"wgEncodeOpenChromFaireRcc7860AlnRep1.bam",
"wgEncodeOpenChromFaireRcc7860AlnRep2.bam",
"wgEncodeOpenChromFaireSmallintestineocAlnRep1.bam",
"wgEncodeOpenChromFaireSmallintestineocAlnRep2.bam",
"wgEncodeOpenChromFaireUrotsaAlnRep1.bam",
"wgEncodeOpenChromFaireUrotsaAlnRep2.bam",
"wgEncodeOpenChromFaireUrotsaUt189AlnRep1.bam",
"wgEncodeOpenChromFaireUrotsaUt189AlnRep2.bam"] # }}}

B = pd.Series(BAMs).str.extract("(?P<bam>.*Faire(?P<cell_line>.*)AlnRep(?P<rep>\d+)\.bam)")

with wolf.Workflow(workflow = BedCovFlow, namespace = "FAIRE_cov") as w:
    for cell_line, b in B.groupby("cell_line"):
        w.run(RUN_NAME = cell_line, bams = base_url + b["bam"], intervals = intervals)

## parse in coverages; make covariate table
from capy import mut

w = wolf.Workflow(workflow = BedCovFlow, namespace = "FAIRE_cov")
for cell_line, b in B.groupby("cell_line"):
    w.load_results(RUN_NAME = cell_line, bams = base_url + b["bam"], intervals = intervals)

T = w.tasks.loc[(slice(None), "BedCovGather"), ["results"]].droplevel(1)
T["covpath"] = T["results"].apply(lambda x : x["concat"])

for i, (cell_line, cov) in enumerate(T.iterrows()):
    X = pd.read_csv(cov["covpath"], sep = "\t", header = None)
    X = X.rename(columns = { len(X.columns) - 1 : cell_line })
    # get common lines
    if i == 0:
        C = X.iloc[:, np.r_[0:3, -1]].rename(columns = { 0 : "chr", 1 : "start", 2 : "end" })
    else:
        C = pd.concat([C, X.iloc[:, -1]], axis = 1)

C["chr"] = mut.convert_chr(C["chr"])

C.to_pickle("covars/FAIRE/coverage.dedup.raw.pickle")

# rebin to 10k
C["index_r"] = C.index//5
C10k = C.groupby(["chr", "index_r"]).agg({
   "start" : min, "end" : max,
   **{ k : sum for k in C.columns[3:] }
}).droplevel(1).reset_index().drop(columns = "index_r")

C10k.to_pickle("covars/FAIRE/coverage.dedup.raw.10kb.pickle")

# gsutil cp covars/FAIRE/coverage.dedup.raw.10kb.pickle gs://getzlab-workflows-reference_files-oa/hg19/hapaseg/FAIRE/coverage.dedup.raw.10kb.pickle

# 100k?
C["index_r"] = C.index//50
C100k = C.groupby(["chr", "index_r"]).agg({
   "start" : min, "end" : max,
   **{ k : sum for k in C.columns[3:] }
}).droplevel(1).reset_index().drop(columns = "index_r")

C100k.to_pickle("covars/FAIRE/coverage.dedup.raw.100kb.pickle")

# }}}

# }}}
