import glob
import os
import pandas as pd
import wolf

import dalmatian
import dask.distributed as dd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as s
import sortedcontainers as sc
import subprocess
import sys

# import stock workflow
from wolF import workflow

# run, in order to get non-simulated inputs
with wolf.Workflow(workflow = workflow.workflow) as w:
    w.run(RUN_NAME = "05bd347a", callstats_file = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/pair/05bd347a/05bd347a-3da7-4d1f-9bc6-4375226a0cb4_de3962db-0bd7-4126-85d9-da9ffe131088.MuTect1.call_stats.txt")

results = w.results

# simulate tumor het coverage of 05bd... with zero allelic imbalance anywhere
H = pd.read_csv(results.loc[("05bd347a", "get_het_coverage_from_callstats", "0"), ("outputs", "tumor_hets")], sep = "\t")

H["cov"] = H["REF_COUNT"] + H["ALT_COUNT"]

H["ALT_COUNT_SIM"] = s.binom.rvs(H["cov"], 0.5)
H["REF_COUNT_SIM"] = H["cov"] - H["ALT_COUNT_SIM"]

H.drop(columns = ["cov", "REF_COUNT", "ALT_COUNT"]).rename(columns = { "ALT_COUNT_SIM" : "ALT_COUNT", "REF_COUNT_SIM" : "REF_COUNT" }).loc[:, ["CONTIG", "POSITION", "REF_COUNT", "ALT_COUNT"]].to_csv("genome/05bd347a.tumor_hets.sim.tsv", sep = "\t", index = False)

# workflow code that runs everything from hapaseg_load onwards
# excised from workflow.py
def sim_workflow():
    hapaseg_load_task = hapaseg.Hapaseg_load(
      inputs = {
        "phased_VCF" : results.loc[("05bd347a", "combine_vcfs", "0"), ("outputs", "combined_vcf")],
        "tumor_allele_counts" : "genome/05bd347a.tumor_hets.sim.tsv",
        "normal_allele_counts" : results.loc[("05bd347a", "get_het_coverage_from_callstats", "0"), ("outputs", "normal_hets")],
        "cytoband_file" : "/mnt/j/db/hg38/ref/cytoBand_primary.txt" # TODO: allow to be specified
      }
    )

    # get intervals for burnin
    @prefect.task
    def get_chunks(scatter_chunks):
        return pd.read_csv(scatter_chunks, sep = "\t")

    chunks = get_chunks(hapaseg_load_task["scatter_chunks"])

    # burnin chunks
    hapaseg_burnin_task = hapaseg.Hapaseg_burnin(
     inputs = {
       "allele_counts" : hapaseg_load_task["allele_counts"],
       "start" : chunks["start"],
       "end" : chunks["end"]
     }
    )

    # concat burned in chunks, infer reference bias
    hapaseg_concat_task = hapaseg.Hapaseg_concat(
     inputs = {
       "chunks" : [hapaseg_burnin_task["burnin_MCMC"]],
       "scatter_intervals" : hapaseg_load_task["scatter_chunks"]
     }
    )

    # run on arms
    hapaseg_arm_AMCMC_task = hapaseg.Hapaseg_amcmc(
     inputs = {
       "amcmc_object" : hapaseg_concat_task["arms"],
       "ref_bias" : hapaseg_concat_task["ref_bias"]
     }
    )

    # concat arm level results
    @prefect.task
    def concat_arm_level_results(arm_results):
        A = []
        for arm_file in arm_results:
            with open(arm_file, "rb") as f:
                H = pickle.load(f)
                A.append(pd.Series({ "chr" : H.P["chr"].iloc[0], "start" : H.P["pos"].iloc[0], "end" : H.P["pos"].iloc[-1], "results" : H }))

        # get into order
        A = pd.concat(A, axis = 1).T.sort_values(["chr", "start", "end"]).reset_index(drop = True)

        # save
        _, tmpfile = tempfile.mkstemp(  )
        A.to_pickle(tmpfile) 

        return tmpfile

    arm_concat = concat_arm_level_results(hapaseg_arm_AMCMC_task["arm_level_MCMC"])

with wolf.Workflow(workflow = sim_workflow) as w:
    w.run(RUN_NAME = "05bd347a_sim")
