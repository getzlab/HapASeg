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

refs = hapaseg.load.HapasegReference(phased_VCF = "exome/eagle.vcf", readbacked_phased_VCF = "exome/whatshap.vcf", allele_counts = "exome/6_C1D1_CFDNA.tumor.tsv")

runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(
  refs.allele_counts,
  refs.chromosome_intervals,
  c,
  misphase_prior = 3e-3,
  _ref_bias = 0.936 # tmp: will be automatically inferred later
)
allelic_segs = runner.run_all()

#allelic_segs.to_pickle("exome/allelic_segs.pickle")
allelic_segs = pd.read_pickle("exome/allelic_segs.pickle")

allelic_segs["results"].iloc[0].visualize()
allelic_segs["results"].iloc[1].visualize()

allelic_segs["results"].iloc[1].correct_phases()
