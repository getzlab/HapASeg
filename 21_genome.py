import dalmatian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import subprocess
import sys

import hapaseg

import dask.distributed as dd

c = dd.Client()

refs = hapaseg.load.HapasegReference(phased_VCF = "genome/3328.eagle.vcf", allele_counts = "3328.tumor.tsv")

runner = hapaseg.run_allelic_MCMC.AllelicMCMCRunner(
  refs.allele_counts,
  refs.chromosome_intervals,
  c,
  misphase_prior = 3e-3
)
allelic_segs = runner.run_all()

#allelic_segs.to_pickle("genome/allelic_segs.pickle")
allelic_segs = pd.read_pickle("genome/allelic_segs.pickle")

allelic_segs["results"].iloc[0].visualize()
