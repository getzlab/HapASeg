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

# alchemist test
with wolf.Workflow(workflow = workflow.workflow, conf = { "clust_frac" : 0.5 }, common_task_opts = { "retry" : 2 } ) as w:
    w.run(
      RUN_NAME = "ALCH_000b5e0e",
      tumor_bam = "gs://fc-secure-e2772064-386d-4911-b242-d6ade82bf172/a07b3704-7a4f-440b-8a70-205c72719674/gdc_api_file_download/89fcac22-33aa-4fe4-ad8f-1bcf35b1d301/call-download_file/554038ea-f80d-495a-b3e3-5e8869cad2ec_wgs_gdc_realn.bam",
      tumor_bai = "gs://fc-secure-e2772064-386d-4911-b242-d6ade82bf172/9048149b-3da3-44e5-974c-3ecf4759777b/gdc_api_file_download/f6184583-3c35-4910-89a6-a721c1185442/call-download_file/554038ea-f80d-495a-b3e3-5e8869cad2ec_wgs_gdc_realn.bai",
      normal_bam = "gs://fc-secure-e2772064-386d-4911-b242-d6ade82bf172/3536bc08-4783-4827-8763-9e65a87e1508/gdc_api_file_download/ecf26baf-01f2-4907-bef2-3bcab6304345/call-download_file/98e061cd-0586-4e56-85fb-c6cc6688dbff_wgs_gdc_realn.bam",
      normal_bai = "gs://fc-secure-e2772064-386d-4911-b242-d6ade82bf172/360c5959-3827-4b24-92e3-d57dbc5de2f6/gdc_api_file_download/15788922-9cf8-4c83-8040-47fa60b7d374/call-download_file/98e061cd-0586-4e56-85fb-c6cc6688dbff_wgs_gdc_realn.bai",
      target_list = 200
    )

# Richter's test (hg19)
import wolf
from wolF import workflow

import dalmatian
wm = dalmatian.WorkspaceManager("broad-firecloud-ibmwatson/Getz_Wu_Richters_WGS_UK")

wic = wolf.fc.WorkspaceInputConnector("broad-firecloud-ibmwatson/Getz_Wu_Richters_WGS_UK")
Pj = wic.get_pairs_as_joint_samples()

with wolf.Workflow(workflow = workflow.workflow, namespace = "HapASeg_Richters") as w:
    for pair, p in Pj.loc[Pj["sample_type_T"] == "Richter"].iterrows():
        w.run(
          RUN_NAME = pair,
          tumor_bam = p["output_bam_T"],
          tumor_bai = p["output_bam_index_T"],
          normal_bam = p["output_bam_N"],
          normal_bai = p["output_bam_index_N"],
          target_list = 2000,
          ref_genome_build = "hg19"
        )
        break
