import dalmatian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import subprocess
import sys
from wolf import fc

# grab a low purity Corcoran exome

WM = dalmatian.WorkspaceManager("corcoran-sada/Corcoran_IO_resistance")
P = WM.get_pairs()
S = WM.get_samples()

# get callstats file
subprocess.check_call("gsutil cp " + P.loc["18144_6_C1D1_CFDNA_BB", 'MUTECT1_CS_SNV'] + " exome", shell = True)

# get BAM/BAI
subprocess.check_call("gsutil cp " + S.loc[P.loc["18144_6_C1D1_CFDNA_BB", "case_sample"], "cram_or_bam_path"] + " exome", shell = True)
subprocess.check_call("gsutil cp " + S.loc[P.loc["18144_6_C1D1_CFDNA_BB", "case_sample"], "crai_or_bai_path"] + " exome", shell = True)

# pulldown het sites
/mnt/j/proj/cnv/20200909_hetpull/hetpull.py -c exome/18144_6_C1D1_CFDNA_BB.MuTect1.call_stats.txt \
 -s /mnt/j/db/hg19/gnomad/ACNV_sites/gnomAD_MAF1.txt -r /mnt/j/db/hg19/ref/hs37d5.fa -o exome/6_C1D1_CFDNA -g

# get coverage




# run
sys.path.append(".")
import hapaseg

import dask.distributed as dd

c = dd.Client()

refs = hapaseg.load.HapasegReference(phased_VCF = "exome/phased.vcf", allele_counts = "exome/6_C1D1_CFDNA.tumor.tsv", ref_bias = 0.938)

# with whatshap correction
refs = hapaseg.load.HapasegReference(phased_VCF = "exome/phased.vcf", readbacked_phased_VCF = "exome/whatshap.vcf", allele_counts = "exome/6_C1D1_CFDNA.tumor.tsv")

runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(refs.allele_counts, refs.chromosome_intervals, c, misphase_prior = 3e-3)
allelic_segs = runner.run_all()

allelic_segs["results"].iloc[0].visualize()
allelic_segs["results"].iloc[1].visualize()

allelic_segs["results"].iloc[1].correct_phases()

#
# infer reference bias
chunk = runner.chunks["results"].iloc[1]
bdy = np.array(chunk.breakpoints); bdy = np.c_[bdy[:-1], bdy[1:]]

a = chunk.P.iloc[333:513]["MIN_COUNT"].sum()
b = chunk.P.iloc[333:513]["MAJ_COUNT"].sum()
(s.beta.rvs(a, b, 1000) - s.beta.rvs(b, a, 1000)).mean()

plt.figure(10); plt.clf()
plt.plot(np.linspace(0.48, 0.52, 200), s.beta.pdf(np.linspace(0.48, 0.52, 200), a, b))


refs = hapaseg.load.HapasegReference(phased_VCF = "exome/phased.vcf", allele_counts = "exome/6_C1D1_CFDNA.normal.tsv", ref_bias = 0.0)

st = 0; en = 118

H.P.iloc[st:en, H.P.columns.get_loc("ALT_COUNT")]
H.P.iloc[st:en, H.P.columns.get_loc("REF_COUNT")]

P = H.P.copy()

bias = P["ALT_COUNT"].sum()/P["REF_COUNT"].sum()
P["REF_COUNT"] *= bias

aidx = P["allele_A"] > 0
bidx = P["allele_B"] > 0

P["MAJ_COUNT"] = pd.concat([P.loc[aidx, "ALT_COUNT"], P.loc[bidx, "REF_COUNT"]])
P["MIN_COUNT"] = pd.concat([P.loc[aidx, "REF_COUNT"], P.loc[bidx, "ALT_COUNT"]])

bpl = np.array(H.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]
for st, en in bpl:
    print(s.beta.ppf([0.05, 0.5, 0.95], P.iloc[st:en, H.maj_idx].sum() + 1, P.iloc[st:en, H.min_idx].sum() + 1))

refs = hapaseg.load.HapasegReference(phased_VCF = "exome/phased.vcf", allele_counts = "exome/6_C1D1_CFDNA.normal.tsv", ref_bias = 0.936365296327212)
runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(refs.allele_counts, refs.chromosome_intervals, c, misphase_prior = 0.00001)
allelic_segs2 = runner.run_all()


alt_idx = H.P.columns.get_loc("ALT_COUNT")
ref_idx = H.P.columns.get_loc("REF_COUNT")
bpl = np.array(H.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]
probs = np.full(len(bpl), np.nan)
for i, (st, en) in enumerate(bpl):
    p = s.beta.cdf(0.5,
      H.P.iloc[st:en, alt_idx].sum() + 1,
      H.P.iloc[st:en, ref_idx].sum() + 1
    )
    probs[i] = np.min(2*np.r_[p, 1 - p])
