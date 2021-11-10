import dalmatian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as s
import subprocess
import sys

import hapaseg

import dask.distributed as dd

#
# scrap notes for running manually

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

#
# run with workflow
import wolf
from wolF import workflow

with wolf.Workflow(workflow = workflow.workflow, conf = { "clust_frac" : 0.5 }, common_task_opts = { "retry" : 2 } ) as w:
    w.run(RUN_NAME = "DLBCL_0138500e", callstats_file = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/pair/0138500e-e992-4036-ace4-ddedcd3e7785_ea550049-0c6d-4f04-92e4-ecb9d092661c/gather_M1__2021-07-13--23-17-11_nsuvzba_4ni5beq_ibegd1vbq5k0w/jobs/0/workspace/0138500e-e992-4036-ace4-ddedcd3e7785_ea550049-0c6d-4f04-92e4-ecb9d092661c.MuTect1.call_stats.txt")
    w.run(RUN_NAME = "DLBCL_06590a5f", callstats_file = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/pair/06590a5f-e523-4391-b3fd-59a315cf57d5_0b7d6938-bf3b-4399-bfb4-2d6c8a252e13/gather_M1__2021-07-13--23-01-32_nsuvzba_4ni5beq_ssrormn1nsebm/jobs/0/workspace/06590a5f-e523-4391-b3fd-59a315cf57d5_0b7d6938-bf3b-4399-bfb4-2d6c8a252e13.MuTect1.call_stats.txt")

with wolf.Workflow(workflow = workflow.workflow, conf = { "clust_frac" : 0.5 }, common_task_opts = { "retry" : 2 } ) as w:
    w.run(
      RUN_NAME = "DLBCL_0138500e",
      callstats_file = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/pair/0138500e-e992-4036-ace4-ddedcd3e7785_ea550049-0c6d-4f04-92e4-ecb9d092661c/gather_M1__2021-07-13--23-17-11_nsuvzba_4ni5beq_ibegd1vbq5k0w/jobs/0/workspace/0138500e-e992-4036-ace4-ddedcd3e7785_ea550049-0c6d-4f04-92e4-ecb9d092661c.MuTect1.call_stats.txt",
      tumor_bam = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/7d418b8a-5b21-443e-8d9b-93081096dc7a/gdc_api_file_download/5c0900df-672a-4f46-a7d4-7992f980f8bf/call-download_file/attempt-2/3ed03341-56bc-4192-a638-ca03de4778e7_wgs_gdc_realn.bam",
      tumor_bai = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/3f2e4ec2-5eac-4ee2-9180-6896f354b423/gdc_api_file_download/f41de056-a7a2-4280-8ee6-6fa78f538711/call-download_file/3ed03341-56bc-4192-a638-ca03de4778e7_wgs_gdc_realn.bai",
      target_list = 200
    )
