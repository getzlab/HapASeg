import wolf
from benchmarking_workflows import *

with wolf.Workflow(workflow=Run_Sim_Workflows) as w:
    w.run(run_name='recomb_easy_sim_1',
          sim_profile='/home/opriebe/data/cnv_sim/benchmarking/sim_samples/cnv_profiles/easy_profile_1.pickle',
          purity = 0.7,
          sample_label='easy_sim_1',
          normal_vcf_path = '/home/opriebe/data/cnv_sim/NA12878/NA12878.vcf',
          ref_build = "hg38",
          ref_fasta = "/home/opriebe/data/ref/hg38/GRCh38.d1.vd1.fa",
          cytoband_file = "/home/opriebe/data/ref/hg38/cytoBand.txt",
          hapaseg_hetsite_depth_path='/home/opriebe/data/cnv_sim/benchmarking/hapaseg/NA12878_hetsites_depth.tsv',
          hapaseg_covcollect_path='/home/opriebe/data/cnv_sim/NA12878/NA12878_2kb_coverage_num_reads_g300.bed',
          hapaseg_phased_vcf_path='/home/opriebe/data/cnv_sim/benchmarking/hapaseg/NA12878_eagle_phasing.vcf',
          gatk_variant_depth_path = '/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_gatk_var_depth.tsv',
          gatk_coverage_tsv_path = '/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_gatk_cov_counts.tsv',
          gatk_sim_normal_allelecounts_path='/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_gatk_cs_sim_normal.tsv',
          gatk_raw_gatk_allelecounts_path='/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_platinum_all_vars_no_sex.allelecounts.tsv',
          gatk_raw_gatk_coverage_path='/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_hg38_wgs_1kb_gatk_no_sex_frag.counts.hdf5',
          gatk_sequence_dictionary='/home/opriebe/data/cnv_sim/benchmarking/gatk/1kG_PoN/Homo_sapiens_assembly38.dict',
          gatk_count_panel='/home/opriebe/data/cnv_sim/benchmarking/gatk/1kG_PoN/GATK_PoN_50samples_1kG.hdf5',
          facets_variant_depth_path = '/home/opriebe/data/cnv_sim/benchmarking/facets/facets_cs_variant_depths.tsv',
          facets_filtered_variants_path = '/home/opriebe/data/cnv_sim/benchmarking/facets/facets_cs_variant_filtered.tsv',
          ascat_variant_depth_path = '/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_cs_variant_depths.tsv',
          ascat_filtered_variants_path = '/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_cs_variant_filtered.tsv',
          ascat_GC_correction_file = '/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_loci/GC_G1000_hg38.txt',
          ascat_RT_correction_file='/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_loci/RT_G1000_hg38.txt'
          )

## Run on large range of simulated profiles
import numpy as np
import pandas as pd
import subprocess

purities = pd.Series(np.r_[0.1:1:0.1], name = "purities")
sim_profiles = pd.Series(
  subprocess.run(
    "gsutil ls gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile*.p*",
    capture_output = True,
    shell = True
  ).stdout.decode().rstrip().split("\n")
).str.extract(
  r"(?P<file>.*profile_(?P<entropy>\d+)_(?P<id>\d+)\.(?P<ext>pickle|png))"
).astype(
  { "entropy" : int, "id" : int }
).pivot(index = ["id", "entropy"], columns = "ext", values = "file").reset_index()

sim_profiles = sim_profiles.sample(50, random_state = 1337)

sim_profiles = sim_profiles.merge(purities, how = "cross")

with wolf.Workflow(workflow=Run_Sim_Workflows, namespace = "CNV_benchmark", scheduler_processes = 10, common_task_opts = { "cleanup_job_workdir" : True }) as w:
    for _, profile in sim_profiles.iterrows():
        purity = np.around(profile['purities'], 1)
        name = f"{profile['entropy']}_{profile['id']}_{purity}"
        w.run(run_name = name,
              sim_profile = profile["pickle"],
              purity = profile["purities"],
              sample_label = name,
              normal_vcf_path = 'gs://hapaseg-pub/cnv_sim/NA12878/NA12878.vcf',
              ref_build = "hg38",
              ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
              cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",

              hapaseg_hetsite_depth_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_hetsites_depth.tsv',
              hapaseg_covcollect_path='gs://hapaseg-pub/cnv_sim/NA12878/NA12878_2kb_coverage_num_reads_g300.bed',
              hapaseg_phased_vcf_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_eagle_phasing.vcf',

              gatk_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_var_depth.tsv',
              gatk_coverage_tsv_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cov_counts.tsv',
              gatk_sim_normal_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cs_sim_normal.tsv',
              gatk_raw_gatk_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_platinum_all_vars_no_sex.allelecounts.tsv',
              gatk_raw_gatk_coverage_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_hg38_wgs_1kb_gatk_no_sex_frag.counts.hdf5',
              gatk_sequence_dictionary='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/Homo_sapiens_assembly38.dict',
              gatk_count_panel='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/GATK_PoN_50samples_1kG.hdf5',

              facets_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_depths.tsv',
              facets_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_filtered.tsv',

              ascat_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_depths.tsv',
              ascat_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_filtered.tsv',
              ascat_GC_correction_file = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/GC_G1000_hg38.txt',
              ascat_RT_correction_file='gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/RT_G1000_hg38.txt'
              )

pd.to_pickle(w.flow_results, "benchmarking.50.pickle")
