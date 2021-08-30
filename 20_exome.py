import dalmatian
import dask.distributed as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import sortedcontainers as sc
import subprocess
import sys
from wolf import fc

sys.path.append(".")
import hapaseg


# # Load in Corcoran IO workspace

WM = dalmatian.WorkspaceManager("corcoran-sada/Corcoran_IO_resistance")
P = WM.get_pairs()
S = WM.get_samples()

# # Sample 1: a low purity exome

# ## Load

# +
# get callstats file
subprocess.check_call("gsutil cp " + P.loc["18144_6_C1D1_CFDNA_BB", 'MUTECT1_CS_SNV'] + " exome", shell = True)

# get BAM/BAI
subprocess.check_call("gsutil cp " + S.loc[P.loc["18144_6_C1D1_CFDNA_BB", "case_sample"], "cram_or_bam_path"] + " exome", shell = True)
subprocess.check_call("gsutil cp " + S.loc[P.loc["18144_6_C1D1_CFDNA_BB", "case_sample"], "crai_or_bai_path"] + " exome", shell = True)

# pulldown het sites
# /mnt/j/proj/cnv/20200909_hetpull/hetpull.py -c exome/18144_6_C1D1_CFDNA_BB.MuTect1.call_stats.txt \
# -s /mnt/j/db/hg19/gnomad/ACNV_sites/gnomAD_MAF1.txt -r /mnt/j/db/hg19/ref/hs37d5.fa -o exome/6_C1D1_CFDNA -g

# get coverage
# /mnt/j/proj/cnv/20210326_coverage_collector/covcollect -b /mnt/j/proj/cnv/20201018_hapseg2/exome/18144_6_C1D1_ctDNA.bam \
# -i targets.bed -o exome/18144_6_C1D1_ctDNA.cov
# -

# The phasing (both imputed and physical) performed in another script I haven't yet exported

# ## Run

# ### Load SNPs/phasing info

refs = hapaseg.load.HapasegReference(
  phased_VCF = "exome/eagle.vcf",
  readbacked_phased_VCF = "exome/whatshap.vcf",
  allele_counts = "exome/6_C1D1_CFDNA.tumor.tsv",
  allele_counts_N = "exome/6_C1D1_CFDNA.normal.tsv"
)

# ### Add overdispersion
#
# For now, we are empirically estimating this at 0.92. In the future, we should be able to infer this.

refs.allele_counts[[
  "REF_COUNT",
  "ALT_COUNT",
  "REF_COUNT_N",
  "ALT_COUNT_N",
  "MAJ_COUNT",
  "MIN_COUNT"
]] *= 0.92

# ### Run segmentation

# +
c = dd.Client(n_workers = 36)
runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(
  refs.allele_counts,
  refs.chromosome_intervals,
  c,
  phase_correct = False
)
allelic_segs = runner.run_all()

allelic_segs.to_pickle("exome/6_C1D1_CFDNA.allelic_segs.auto_ref_correct.overdispersion92.no_phase_correct.pickle")
# -

# # Sample 2: a higher purity exome from the same individual

# ## Load

# get callstats file
subprocess.check_call("gsutil cp " + P.loc["18144_6_C1D1_tissue_DNA", 'MUTECT1_CS_SNV'] + " exome", shell = True)

# get BAM/BAI
subprocess.check_call("gsutil cp " + S.loc[P.loc["18144_6_C1D1_tissue_DNA", "case_sample"], "cram_or_bam_path"] + " exome", shell = True)
subprocess.check_call("gsutil cp " + S.loc[P.loc["18144_6_C1D1_tissue_DNA", "case_sample"], "crai_or_bai_path"] + " exome", shell = True)

# Once again, genotyping/coverage collection/phasing performed in another script I haven't yet exported


# ## Run

# ### Load SNPs/phasing info

refs = hapaseg.load.HapasegReference(
  phased_VCF = "exome/6_C1D1_META.eagle.vcf",
  # read-backed phasing not yet performed for this sample
  allele_counts = "exome/6_C1D1_META.tumor.tsv",
  allele_counts_N = "exome/6_C1D1_META.normal.tsv"
)

# ### Add overdispersion
#
# (again, empirically estimated at ~0.92)

refs.allele_counts[[
  "REF_COUNT",
  "ALT_COUNT",
  "REF_COUNT_N",
  "ALT_COUNT_N",
  "MAJ_COUNT",
  "MIN_COUNT"
]] *= 0.92

# ### Run segmentation

# +
c = dd.Client(n_workers = 36)
runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(
  refs.allele_counts,
  refs.chromosome_intervals,
  c,
  phase_correct = False
)
allelic_segs = runner.run_all()

allelic_segs.to_pickle("exome/6_C1D1_META.allelic_segs.auto_ref_correct.overdispersion92.no_phase_correct.pickle")
# -

# # (scrap code)
#
# Debugging why reverting intervals in F won't restore us to the original state

refs = hapaseg.load.HapasegReference(phased_VCF = "exome/6_C1D1_META.eagle.vcf", allele_counts = "exome/6_C1D1_META.tumor.tsv")

runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(
  refs.allele_counts.loc[refs.allele_counts["chr"] == 1],
  refs.chromosome_intervals,
  c,
  #phase_correct = False,
  misphase_prior = 3e-3,
  #_ref_bias = 0.936 # tmp: will be automatically inferred later
)

self = runner

#
# code excised from run_allelic_MCMC

from hapaseg import A_MCMC

chunks = [slice(*x) for x in self.chunks[["start", "end"]].values]

futures = self.client.map(self._run_on_chunks, chunks, P = self.P_shared)
self.chunks["results"] = self.client.gather(futures)

#
# concatenate burned in chunks for each arm
H = [None]*len(self.chunks["arm"].unique())
for i, (arm, A) in enumerate(self.chunks.groupby("arm")):
    # concatenate allele count dataframes
    H[i] = A_MCMC(
      pd.concat([x.P for x in A["results"]], ignore_index = True),
      n_iter = self.n_iter,
      phase_correct = self.phase_correct,
      misphase_prior = self.misphase_prior,
      ref_bias = self._ref_bias # TODO: infer dynamically from burnin chunks
    )

    # replicate constructor steps to define initial breakpoint set and
    # marginal likelihood dict
    breakpoints = [None]*len(A)
    H[i].seg_marg_liks = sc.SortedDict()
    for j, (_, _, start, _, r) in enumerate(A.itertuples()):
        start -= A["start"].iloc[0]
        breakpoints[j] = np.array(r.breakpoints) + start
        for k, v in r.seg_marg_liks.items():
            H[i].seg_marg_liks[k + start] = v
    H[i].breakpoints = sc.SortedSet(np.hstack(breakpoints))

    H[i].marg_lik = np.full(H[i].n_iter, np.nan)
    H[i].marg_lik[0] = np.array(H[i].seg_marg_liks.values()).sum()

H[0].run()

#
# run on just a single chunk

refs = hapaseg.load.HapasegReference(phased_VCF = "exome/6_C1D1_META.eagle.vcf", allele_counts = "exome/6_C1D1_META.tumor.tsv", allele_counts_N = "exome/6_C1D1_META.normal.tsv")

A = hapaseg.allelic_MCMC.A_MCMC(P = refs.allele_counts.iloc[0:543], n_iter = 20000, phase_correct = True, ref_bias = 0.93)
A.run()

A = hapaseg.allelic_MCMC.A_MCMC(P = refs.allele_counts.iloc[0:543], n_iter = 20000)
A.run()
