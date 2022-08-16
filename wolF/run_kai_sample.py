import numpy as np
import pandas as pd
import dalmatian
import wolf

import workflow

with wolf.Workflow(
  workflow = workflow.workflow,
  conf = { "clust_frac": 0.7 }, # if you want to use more machines in the elastic cluster
  common_task_opts = { "retry" : 2 }, # will retry every task up to 5 times
  scheduler_processes=1
) as w:
    w.run(
        run_name = "hapaseg_kai_ffpe_exome",
        tumor_bam = "gs://opriebe-tmp/DFCI_CAR_020_FFPE_01.bam",
        tumor_bai = "gs://opriebe-tmp/DFCI_CAR_020_FFPE_01.bai",
        normal_bam = "gs://opriebe-tmp/DFCI_CAR_020_N_01.bam",
        normal_bai = "gs://opriebe-tmp/DFCI_CAR_020_N_01.bai",
        # used to name files output by the workflow
        ref_genome_build = "hg19",
        target_list='/home/opriebe/dev/HapASeg/exome/TWIST_intervals_list.bed',
        persistent_dry_run = False
    )
