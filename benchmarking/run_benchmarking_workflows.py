import wolf
from benchmarking_workflows import *
import glob
import numpy as np
import os
import pandas as pd
import pickle

# 50 profiles
profile_paths = glob.glob('/home/opriebe/data/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_*.pickle')
purity_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # using arange leads to floating point weirdness when printing

with wolf.Workflow(workflow=Run_Sim_Workflows, 
                   namespace="benchmarking_dir",
                   scheduler_processes=32,
                   max_concurrent_flow_tasks = 3,
                   max_concurrent_flows = 200,
                   common_task_opts = { "retry" : 8 }) as w:
    for profile_path in profile_paths:
        for purity in purity_range:
            # Fresh frozen pcr free
            w.run(run_name = 'FF_' + os.path.basename(profile_path).rstrip('.pickle') + '_' + str(np.around(purity,2)),
                  sim_profile=profile_path,
                  purity = purity,
                  sample_label= 'FF_' + os.path.basename(profile_path).rstrip('.pickle'),
                  normal_vcf_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                  normal_callstats_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/SRR6691666/NA12878_SRR6691666_mutect_callstats.tsv',
                  ref_build = "hg38",
                  ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
                  cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
                  hapaseg_hetsite_depth_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hapaseg/hapaseg_NA12878_platinum_realigned_hetsites_depth.tsv',
                  hapaseg_covcollect_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878_platinum_realigned_covcollect.bed',
                  hapaseg_normal_covcollect_path="gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/SRR6691666/NA12878_SRR6691666_covcollect.bed",
                  hapaseg_phased_vcf_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878_platinum_realigned_eagle_phased.vcf',
                  gatk_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/NA12878_platnium_realigned_gatk_var_depth.tsv',
                  gatk_coverage_tsv_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/NA12878_platnium_realigned_gatk_cov_counts.tsv',
                  gatk_raw_gatk_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/NA12878_platnium_realigned_gatk.allelecounts.tsv',
                  gatk_raw_gatk_coverage_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/NA12878_platnium_realigned_gatk.frag.counts.hdf5',
                  gatk_sequence_dictionary = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                  gatk_count_panel = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/GATK_PoN_50samples_1kG.hdf5',
                  facets_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/facets/facets_NA12878_platinum_realigned_cs_variant_depths.tsv',
                  facets_filtered_variants_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/facets/facets_NA12878_platinum_realigned_cs_variant_filtered.tsv',
                  ascat_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/ascat/ascat_NA12878_platinum_realigned_cs_variant_depths.tsv',
                  ascat_filtered_variants_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/ascat/ascat_NA12878_platinum_realigned_cs_variant_filtered.tsv',
                  ascat_GC_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/GC_G1000_hg38.txt',
                  ascat_RT_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/RT_G1000_hg38.txt',
                  hatchet_tumor_baf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/raw_data/allele_counts/tumor_snps.txt',
                  hatchet_thresholds_files_dir = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/raw_data/read_counts/thresholds/',
                  hatchet_int_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/preprocess_data/NA12878_platinum_SRR6691666_interval_counts.for_simulation_input.txt',
                  hatchet_pos_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/preprocess_data/NA12878_platinum_SRR6691666_position_counts.for_simulation_input.txt',
                  hatchet_snp_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/preprocess_data/NA12878_platinum_SRR6691666_snp_counts.for_simulation_input.txt',
                  hatchet_read_combined_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/preprocess_data/NA12878_platinum_SRR6691666_read_combined_df.txt',
                  hatchet_phased_vcf = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/phased.vcf.gz"      
            )
            #FFPE 1022
            w.run(run_name = 'FFPE_CH1022_' + os.path.basename(profile_path).rstrip('.pickle') + '_' + str(np.around(purity,2)),
              sim_profile=profile_path,
              purity = purity,
              sample_label= 'FFPE_CH1022_' + os.path.basename(profile_path).rstrip('.pickle'),
              normal_vcf_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
              normal_callstats_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022GL/CH1022GL_NA12878_mutect_callstats.tsv',
              unmatched_normal_callstats=True,
              ref_build = "hg38",
              ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
              cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
              hapaseg_hetsite_depth_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hapaseg/hapaseg_CH1022LN_NA12878_hetsites_depth.tsv',
              hapaseg_covcollect_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/CH1022LN_NA12878_covcollect_chrs_excluded.bed',
              hapaseg_normal_covcollect_path="gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022GL/CH1022GL_NA12878_covcollect.bed",
              hapaseg_phased_vcf_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878_platinum_realigned_eagle_phased.vcf',
              gatk_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/CH1022LN_gatk_var_depth.tsv',
              gatk_coverage_tsv_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/CH1022LN_gatk_cov_counts.tsv',
              gatk_raw_gatk_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/CH1022LN_gatk.allelecounts.tsv',
              gatk_raw_gatk_coverage_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/CH1022LN_gatk.frag.counts.hdf5',
              gatk_sequence_dictionary = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/hg38_1kG_wgs_gatk.annotated_intervals_no_sex.tsv',
              gatk_count_panel = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022GL/CH1022GL_normal_sample_PoN.hdf5',
              facets_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/facets/facets_CH1022LN_NA12878_cs_variant_depths.tsv',
              ascat_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/ascat/ascat_CH1022LN_NA12878_cs_variant_depths.tsv',
              ascat_GC_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/GC_G1000_hg38.txt',
              ascat_RT_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/RT_G1000_hg38.txt',
              hatchet_tumor_baf = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hatchet/raw_data/allele_counts/tumor_snps.txt',
              hatchet_thresholds_files_dir = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hatchet/raw_data/read_counts/thresholds/',
              hatchet_int_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hatchet/preprocess_data/NA12878_platinum_CH1022GL_CH1022LN_interval_counts.for_simulation_input.txt',
              hatchet_pos_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hatchet/preprocess_data/NA12878_platinum_CH1022GL_CH1022LN_position_counts.for_simulation_input.txt',
              hatchet_snp_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hatchet/preprocess_data/NA12878_platinum_CH1022GL_CH1022LN_snp_counts.for_simulation_input.txt',
              hatchet_read_combined_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/hatchet/preprocess_data/NA12878_platinum_CH1022GL_CH1022LN_read_combined_df.txt',
              hatchet_phased_vcf = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/phased.vcf.gz",     
              is_ffpe=True
            )
            #FFPE 1032
            w.run(run_name = 'FFPE_CH1032_' + os.path.basename(profile_path).rstrip('.pickle') + '_' + str(np.around(purity,2)),
              sim_profile=profile_path,
              purity = purity,
              sample_label= 'FFPE_CH1032_' + os.path.basename(profile_path).rstrip('.pickle'),
              normal_vcf_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
              normal_callstats_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032GL/CH1032GL_NA12878_mutect_callstats.tsv',
              unmatched_normal_callstats=True,
              ref_build = "hg38",
              ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
              cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
              hapaseg_hetsite_depth_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hapaseg/hapaseg_CH1032LN_NA12878_hetsites_depth.tsv',
              hapaseg_covcollect_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/CH1032LN_NA12878_covcollect_chrs_excluded.bed',
              hapaseg_normal_covcollect_path="gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032GL/CH1032GL_NA12878_covcollect.bed",
              hapaseg_phased_vcf_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878_platinum_realigned_eagle_phased.vcf',
              gatk_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/CH1032LN_gatk_var_depth.tsv',
              gatk_coverage_tsv_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/CH1032LN_gatk_cov_counts.tsv',
              gatk_raw_gatk_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/CH1032LN_gatk.allelecounts.tsv',
              gatk_raw_gatk_coverage_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/CH1032LN_gatk.frag.counts.hdf5',
              gatk_sequence_dictionary = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/hg38_1kG_wgs_gatk.annotated_intervals_no_sex.tsv',
              gatk_count_panel = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032GL/CH1032GL_normal_sample_PoN.hdf5',
              facets_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/facets/facets_CH1032LN_NA12878_cs_variant_depths.tsv',
              ascat_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/ascat/ascat_CH1032LN_NA12878_cs_variant_depths.tsv',
              ascat_GC_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/GC_G1000_hg38.txt',
              ascat_RT_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/RT_G1000_hg38.txt',
              hatchet_tumor_baf = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hatchet/raw_data/allele_counts/tumor_snps.txt',
              hatchet_thresholds_files_dir = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hatchet/raw_data/read_counts/thresholds/',
              hatchet_int_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hatchet/preprocess_data/NA12878_platinum_CH1032GL_CH1032LN_interval_counts.for_simulation_input.txt',
              hatchet_pos_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hatchet/preprocess_data/NA12878_platinum_CH1032GL_CH1032LN_position_counts.for_simulation_input.txt',
              hatchet_snp_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hatchet/preprocess_data/NA12878_platinum_CH1032GL_CH1032LN_snp_counts.for_simulation_input.txt',
              hatchet_read_combined_file = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/hatchet/preprocess_data/NA12878_platinum_CH1032GL_CH1032LN_read_combined_df.txt',
              hatchet_phased_vcf = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/phased.vcf.gz", 
              is_ffpe=True
        )
            # finally, run exome
            w.run(run_name = 'exome_TWIST_' + os.path.basename(profile_path).rstrip('.pickle') + '_' + str(np.around(purity,2)),
                  sim_profile=profile_path,
                  purity = purity,
                  sample_label= 'exome_TWIST_' + os.path.basename(profile_path).rstrip('.pickle'),
                  normal_vcf_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                  normal_callstats_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/TWIST_18/NA12878_TWIST_18_mutect_callstats.tsv',
                  ref_build = "hg38",
                  ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
                  cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
                  hapaseg_hetsite_depth_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hapaseg/hapaseg_NA12878_TWIST_36_hetsites_depth.tsv',
                  hapaseg_covcollect_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/NA12878_TWIST_36_covcollect.bed',
                  hapaseg_normal_covcollect_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/TWIST_18/NA12878_TWIST_18_covcollect.bed',
                  hapaseg_phased_vcf_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/NA12878_TWIST_36_eagle_phased.vcf',
                  hapaseg_target_list='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/broad_custom_exome_v1.Homo_sapiens_assembly38.targets.bed',
                  hapaseg_run_cdp=False, ## try not running cdp on WES
                  gatk_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/TWIST_36/NA12878_TWIST_36_gatk_var_depth.tsv',
                  gatk_coverage_tsv_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/TWIST_36/NA12878_TWIST_36_gatk_cov_counts.tsv',
                  gatk_raw_gatk_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/TWIST_36/NA12878_TWIST_36_gatk.allelecounts.tsv',
                  gatk_raw_gatk_coverage_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/TWIST_36/NA12878_TWIST_36_gatk.frag.counts.hdf5',
                  gatk_sequence_dictionary = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                  gatk_count_panel = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/twist_hg38_PoN.hdf5',
                  facets_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/facets/facets_NA12878_TWIST_36_cs_variant_depths.tsv',
                  facets_filtered_variants_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/facets/facets_NA12878_TWIST_36_cs_variant_filtered.tsv',
                  ascat_variant_depth_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/ascat/ascat_NA12878_TWIST_36_cs_variant_depths.tsv',
                  ascat_filtered_variants_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/ascat/ascat_NA12878_TWIST_36_cs_variant_filtered.tsv',
                  ascat_GC_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/GC_G1000_hg38.txt',
                  ascat_RT_correction_file = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/RT_G1000_hg38.txt',
                  hatchet_tumor_baf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/raw_data/allele_counts/tumor_snps.txt',
                  hatchet_thresholds_files_dir = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/raw_data/read_counts/thresholds/',
                  hatchet_int_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/preprocess_data/NA12878_TWIST_36_18_interval_counts.for_simulation_input.txt',
                  hatchet_pos_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/preprocess_data/NA12878_TWIST_36_18_position_counts.for_simulation_input.txt',
                  hatchet_snp_counts_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/preprocess_data/NA12878_TWIST_36_18_snp_counts.for_simulation_input.txt',
                  hatchet_read_combined_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/preprocess_data/NA12878_TWIST_36_18_read_combined_df.txt',
                  hatchet_phased_vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/hatchet/phased.vcf.gz'      
            )

    with open('./final_benchmakring_results.pickle', 'wb') as f:
        pickle.dump(w.flow_results, f)

### Run on large range of simulated profiles
#import numpy as np
#import pandas as pd
#import subprocess
#
#purities = pd.Series(np.r_[0.1:1:0.1], name = "purities")
#sim_profiles = pd.Series(
#  subprocess.run(
#    "gsutil ls gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile*.p*",
#    capture_output = True,
#    shell = True
#  ).stdout.decode().rstrip().split("\n")
#).str.extract(
#  r"(?P<file>.*profile_(?P<entropy>\d+)_(?P<id>\d+)\.(?P<ext>pickle|png))"
#).astype(
#  { "entropy" : int, "id" : int }
#).pivot(index = ["id", "entropy"], columns = "ext", values = "file").reset_index()
#
#sim_profiles = sim_profiles.sample(50, random_state = 1337)
#
#sim_profiles = sim_profiles.merge(purities, how = "cross")
#
#with wolf.Workflow(workflow=Run_Sim_Workflows, namespace = "CNV_benchmark", scheduler_processes = 10, common_task_opts = { "cleanup_job_workdir" : True }) as w:
#    for _, profile in sim_profiles.iterrows():
#        purity = np.around(profile['purities'], 1)
#        name = f"{profile['entropy']}_{profile['id']}_{purity}"
#        w.run(run_name = name,
#              sim_profile = profile["pickle"],
#              purity = profile["purities"],
#              sample_label = name,
#              normal_vcf_path = 'gs://hapaseg-pub/cnv_sim/NA12878/NA12878.vcf',
#              ref_build = "hg38",
#              ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
#              cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
#
#              hapaseg_hetsite_depth_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_hetsites_depth.tsv',
#              hapaseg_covcollect_path='gs://hapaseg-pub/cnv_sim/NA12878/NA12878_2kb_coverage_num_reads_g300.bed',
#              hapaseg_phased_vcf_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_eagle_phasing.vcf',
#
#              gatk_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_var_depth.tsv',
#              gatk_coverage_tsv_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cov_counts.tsv',
#              gatk_sim_normal_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cs_sim_normal.tsv',
#              gatk_raw_gatk_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_platinum_all_vars_no_sex.allelecounts.tsv',
#              gatk_raw_gatk_coverage_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_hg38_wgs_1kb_gatk_no_sex_frag.counts.hdf5',
#              gatk_sequence_dictionary='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/Homo_sapiens_assembly38.dict',
#              gatk_count_panel='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/GATK_PoN_50samples_1kG.hdf5',
#
#              facets_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_depths.tsv',
#              facets_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_filtered.tsv',
#
#              ascat_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_depths.tsv',
#              ascat_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_filtered.tsv',
#              ascat_GC_correction_file = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/GC_G1000_hg38.txt',
#              ascat_RT_correction_file='gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/RT_G1000_hg38.txt'
#              )
#
#pd.to_pickle(w.flow_results, "benchmarking.50.pickle")
#
### re-run comparator with standardized purity
#
#F = pd.read_pickle("benchmarking.50.pickle")
#
#def comparison_workflow(
#  sim_profile=None,
#  sample_label=None,
#
#  normal_vcf_path=None,
#  hapaseg_hetsite_depth_path=None,
#  hapaseg_covcollect_path=None,
#  input_purity=None,
#
#  ref_fasta=None,
#  cytoband_file=None,
#
#  hapaseg_seg_file = None,
#
#  gatk_tumor_coverage = None,
#  gatk_tumor_allele_counts = None,
#  gatk_seg_file = None,
#
#  facets_input_counts = None,
#  facets_seg_file = None,
#
#  ascat_tumor_logR = None,
#  ascat_tumor_BAF = None,
#  ascat_raw_segments = None,
#):
#    seg_file_gen_task = Generate_Groundtruth_Segfile(inputs= {
#                            "sample_label": sample_label,
#                            "purity":0.7,
#                            "sim_profile":sim_profile,
#                            "normal_vcf_path":normal_vcf_path,
#                            "hapaseg_hetsite_depth_path": hapaseg_hetsite_depth_path,
#                            "hapaseg_coverage_tsv":hapaseg_covcollect_path
#                        },
#                        extra_localization_args = { "localize_to_persistent_disk" : True }
#                    ) 
#
#    localization_task = wolf.LocalizeToDisk(files = {
#                                    "ref_fasta" : ref_fasta,
#                                    "cytoband_file" : cytoband_file
#                                    })
#
#    hapaseg_downstream = Downstream_HapASeg_Analysis(inputs = {
#                                   "hapaseg_seg_file": hapaseg_seg_file,
#                                   "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
#                                   "sample_name": f"{sample_label}_{input_purity}",
#                                   "ref_fasta": localization_task["ref_fasta"],
#                                   "cytoband_file": localization_task["cytoband_file"]
#                        }, job_avoid = False)
#
#    gatk_downstream = Downstream_GATK_Analysis(
#              inputs={"gatk_sim_cov_input": gatk_tumor_coverage,
#                      "gatk_sim_acounts" : gatk_tumor_allele_counts,
#                      "gatk_seg_file" : gatk_seg_file,
#                      "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
#                      "sample_name": f"{sample_label}_{input_purity}",
#                      "ref_fasta": localization_task["ref_fasta"],
#                      "cytoband_file": localization_task["cytoband_file"]
#                    }
#            )
#
#    facets_downstream = Downstream_Facets_Analysis(
#                            inputs={"facets_input_counts": facets_input_counts,
#                                    "facets_seg_file": facets_seg_file,
#                                     "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
#                                     "sample_name": f"{sample_label}_{input_purity}",
#                                     "ref_fasta": localization_task["ref_fasta"],
#                                     "cytoband_file": localization_task["cytoband_file"]
#                                   }
#                            )
#
#    ascat_downstream = Downstream_ASCAT_Analysis(
#                                inputs={"ascat_t_logr": ascat_tumor_logR,
#                                        "ascat_t_baf": ascat_tumor_BAF,
#                                        "ascat_seg_file": ascat_raw_segments,
#                                        "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
#                                        "sample_name": f"{sample_label}_{input_purity}",
#                                        "ref_fasta": localization_task["ref_fasta"],
#                                        "cytoband_file": localization_task["cytoband_file"]
#                                       }
#                                )
#
#with wolf.Workflow(workflow = comparison_workflow, namespace = "CNV_benchmark") as w:
#    for samp in F.keys():
#        sim_inputs = F[samp]["Generate_Groundtruth_Segfile"].results["inputs"].iloc[0].to_dict()
#
#        # in case some samples don't have various outputs
#        try:
#            w.run(
#              RUN_NAME = samp + "_recompare",
#
#              sim_profile=sim_inputs["sim_profile"],
#              sample_label=samp,
#
#              normal_vcf_path=sim_inputs["normal_vcf_path"],
#              hapaseg_hetsite_depth_path=sim_inputs["hapaseg_hetsite_depth_path"],
#              hapaseg_covcollect_path=sim_inputs["hapaseg_coverage_tsv"],
#              input_purity=sim_inputs["purity"],
#
#              ref_fasta="gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
#              cytoband_file="gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
#
#              hapaseg_seg_file = F[samp]["Hapaseg_run_acdp"]["acdp_segfile"],
#
#              gatk_tumor_coverage = F[samp]["Generate_GATK_Sim_Data"]["tumor_coverage_tsv"],
#              gatk_tumor_allele_counts = F[samp]["Generate_GATK_Sim_Data"]["tumor_allele_counts"],
#              gatk_seg_file = F[samp]["GATK_CNV_model_segments"]["model_segments_post_smoothing"],
#
#              facets_input_counts = F[samp]["Generate_Facets_Sim_Data"]["facets_input_counts"],
#              facets_seg_file = F[samp]["Facets"]["facets_seg_file"],
#
#              ascat_tumor_logR = F[samp]["Generate_ASCAT_Sim_Data"]["ascat_tumor_logR"],
#              ascat_tumor_BAF = F[samp]["Generate_ASCAT_Sim_Data"]["ascat_tumor_BAF"],
#              ascat_raw_segments = F[samp]["ASCAT"]["ascat_raw_segments"]
#            )
#        except:
#            pass
#
#pd.to_pickle(w.flow_results, "benchmarking.50.rescale.pickle")
#
### re-run comparator with standardized purity
#
#F2 = pd.read_pickle("benchmarking.50.rescale.pickle")
#
#methods = ['Downstream_ASCAT_Analysis', 'Downstream_Facets_Analysis', 'Downstream_GATK_Analysis', 'Downstream_HapASeg_Analysis']
#D = { samp : { method : F2[samp][method] for method in methods } for samp in F2.keys() }
#
#D = pd.DataFrame.from_dict(D, orient = "index")
#D = pd.concat([D.reset_index(drop = True), D.index.str.extract(r"(?P<entropy>\d+)_\d+_(?P<purity>[\d\.]+)_recompare")], axis = 1).astype({ "entropy" : int, "purity" : float })
#
#for method in methods:
#    D[method + "_MAD"] = np.nan
#
#for idx, row in D.iterrows():
#    for method in methods:
#        try:
#            score = pd.read_csv(row[method]["comparison_results"], sep = "\t")
#            D.loc[idx, method + "_MAD"] = score["mad_score"][0]
#        except:
#            pass
#
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#plt.figure(11); plt.clf()
#
#minent = D["entropy"].min()
#
#for ent, idx in D.groupby("entropy").groups.items():
#    jit = lambda x : np.random.rand(len(x))/20 - 0.025
#    sz = 5 + (ent - minent)**2/500
#    l2 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_GATK_Analysis_MAD"], s = sz, color = colors[1], alpha = 0.8)
#    l3 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_ASCAT_Analysis_MAD"], s = sz, color = colors[2], alpha = 0.8)
#    l4 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_Facets_Analysis_MAD"], s = sz, color = colors[3], alpha = 0.8)
#    l1 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_HapASeg_Analysis_MAD"], s = sz, color = colors[0], alpha = 0.8)
#
#plt.xticks(np.r_[0.1:1:0.1])
#
#plt.yscale("log")
#plt.xlabel("Purity")
#plt.ylabel("MAD")
#
#plt.legend([l1, l2, l3, l4], ["HapASeg", "GATK4 CNV", "ASCAT", "FACETS"], loc = "lower center", ncol = 4)
#plt.ylim([0.07, 140])
#
#ent_quant = np.quantile(D["entropy"], [0.33, 0.66])
#
#plt.figure(
#
### TODO: move elsewhere
#import sys
#from capy import plots, seq, mut
#
## corrected coverage
#N = pd.read_csv(F["48214_21_0.2"]["GATK_CNV_denoise"]["std_copy_ratios"], comment = "@", sep = "\t")
#sys.path.append("/home/jhess/j/proj/cnv/20201018_hapseg2")
#import hapaseg.utils
#
#N["gpos"] = seq.chrpos2gpos(mut.convert_chr(N["CONTIG"]), N["START"])
#
## raw coverage
#M = pd.read_csv(F["48214_21_0.2"]["Generate_GATK_Sim_Data"]["tumor_coverage_tsv"], comment = "@", sep = "\t")
#M["gpos"] = seq.chrpos2gpos(M["chr"], M["start"])
#
#plt.figure(1, figsize = (16, 4)); plt.clf()
#plots.pixplot(N["gpos"], np.exp2(N["LOG2_COPY_RATIO"]))
#hapaseg.utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")
#
#plt.figure(2, figsize = (16, 4)); plt.clf()
#plots.pixplot(M["gpos"], M["covcorr"], color = "r")
#plt.ylim([0, 1000])
#hapaseg.utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")
#
## plot input caller
#
#C = pd.read_csv(F["48214_21_0.2"]["Generate_Groundtruth_Segfile"].results["inputs"]["hapaseg_coverage_tsv"][0], sep = "\t", names = ["chr", "start", "end", "covcorr", "x", "y", "z", "u", "v"])
#C = C.loc[C["chr"] != "chrM"]
#C["gpos"] = seq.chrpos2gpos(mut.convert_chr(C["chr"]), C["start"])
#
#plt.figure(3, figsize = (16, 4)); plt.clf()
#plots.pixplot(C["gpos"], C["covcorr"])
#plt.ylim([0, 300000])
#hapaseg.utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")
#
## hets
#H = pd.read_csv(F["48214_21_0.2"]["Generate_HapASeg_Sim_Data"]["hapaseg_hets"], sep = "\t")
#H["gpos"] = seq.chrpos2gpos(mut.convert_chr(H["CONTIG"]), H["POSITION"])
#
## scaled coverage
#S = pd.read_csv(F["48214_21_0.2"]["Generate_HapASeg_Sim_Data"]["hapaseg_coverage_bed"], sep = "\t", names = ["chr", "start", "end", "covcorr", "x", "y", "z", "u", "v"])
#S = S.loc[S["chr"] != "chrM"]
#S["gpos"] = seq.chrpos2gpos(mut.convert_chr(S["chr"]), S["start"])
#
#plt.figure(4, figsize = (16, 4)); plt.clf()
#plots.pixplot(S["gpos"], S["covcorr"])
#plt.ylim([0, 300000])
#hapaseg.utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")
#
#plt.figure(5, figsize = (16, 4)); plt.clf()
#plots.pixplot(H["gpos"], H["ALT_COUNT"]/H[["REF_COUNT", "ALT_COUNT"]].sum(1), color = "orange")
#plt.ylim([0, 1])
#hapaseg.utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")
#
#
###
#X = pd.read_csv("gs://opriebe-tmp/RP-2550_GTEX-13OW5-0126_v1_WGS_GCP.covcollect.tsv", sep = "\t", names = ["chr", "start", "end", "covcorr", "x", "y", "z", "u", "v"])
#X = X.loc[X["chr"] != "chrM"]
#X["gpos"] = seq.chrpos2gpos(mut.convert_chr(X["chr"]), X["start"])
#
#plt.figure(40, figsize = (16, 4)); plt.clf()
#plots.pixplot(X["gpos"], X["covcorr"])
#plt.ylim([0, 3.3e6])
#hapaseg.utils.plot_chrbdy("/mnt/j/db/hg38/ref/cytoBand_primary.txt")
#plt.xlim([0, 3093173846])
#
### plot all MADs
#methods = ['Downstream_ASCAT_Analysis', 'Downstream_Facets_Analysis', 'Downstream_GATK_Analysis', 'Downstream_HapASeg_Analysis']
#D = { samp : { method : F[samp][method] for method in methods } for samp in F.keys() }
#
#D = pd.DataFrame.from_dict(D, orient = "index")
#D = pd.concat([D.reset_index(drop = True), D.index.str.extract(r"(?P<entropy>\d+)_\d+_(?P<purity>[\d\.]+)")], axis = 1).astype({ "entropy" : int, "purity" : float })
#
#for method in methods:
#    D[method + "_MAD"] = np.nan
#
#for idx, row in D.iterrows():
#    for method in methods:
#        try:
#            score = pd.read_csv(row[method]["comparison_results"], sep = "\t")
#            D.loc[idx, method + "_MAD"] = score["mad_score"][0]
#        except:
#            pass
#
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
#plt.figure(11); plt.clf()
#
#minent = D["entropy"].min()
#
#for ent, idx in D.groupby("entropy").groups.items():
#    jit = lambda x : np.random.rand(len(x))/20 - 0.025
#    sz = 5 + (ent - minent)/500
#    l1 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_HapASeg_Analysis_MAD"], s = sz, color = colors[0], alpha = 0.8)
#    l2 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_GATK_Analysis_MAD"], s = sz, color = colors[1], alpha = 0.8)
#    l3 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_ASCAT_Analysis_MAD"], s = sz, color = colors[2], alpha = 0.8)
#    l4 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_Facets_Analysis_MAD"], s = sz, color = colors[3], alpha = 0.8)
#
#plt.xticks(np.r_[0.1:1:0.1])
#
#plt.yscale("log")
#plt.xlabel("Purity")
#plt.ylabel("MAD")
#
#plt.legend([l1, l2, l3, l4], ["HapASeg", "GATK4 CNV", "ASCAT", "FACETS"], loc = "lower center", ncol = 4)
#plt.ylim([0.05, 100])
#
#
### Run on FFPE
#
#ffpe_sim_profiles = pd.DataFrame(dict(
#pickle = ["gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_32882_146.pickle",
#"gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_33708_76.pickle",
#"gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_39976_17.pickle",
#"gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_46924_92.pickle"],
#desc = ["easiest", "easy", "med", "hard"]
#))
#
#purities = pd.Series([0.15, 0.30, 0.6, 0.9], name = "purities")
#
#ffpe_sim_profiles = ffpe_sim_profiles.merge(purities, how = "cross")
#
#with wolf.Workflow(workflow=Run_Sim_Workflows, namespace = "CNV_benchmark", scheduler_processes = 10, max_concurrent_flows = 200) as w:
#    for _, profile in ffpe_sim_profiles.iterrows():
#        purity = np.around(profile['purities'], 1)
#        name = f"FFPE_{profile['desc']}_{purity}"
#        w.run(run_name = name,
#              sim_profile = profile["pickle"],
#              purity = profile["purities"],
#              sample_label = name,
#              normal_vcf_path = 'gs://hapaseg-pub/cnv_sim/NA12878/NA12878.vcf',
#              ref_build = "hg38",
#              ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
#              cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",
#
#              # UPDATE
#              hapaseg_hetsite_depth_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_hetsites_depth.tsv',
#              hapaseg_covcollect_path='../benchmarking_data/1022_cov.bed',
#              hapaseg_phased_vcf_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_eagle_phasing.vcf',
#              hapaseg_is_ffpe=True,
#
#              # UPDATE
#              gatk_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_var_depth.tsv',
#              gatk_coverage_tsv_path = '../benchmarking_data/1022_cov_totreads.bed',
#              gatk_sim_normal_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cs_sim_normal.tsv',
#              gatk_raw_gatk_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_platinum_all_vars_no_sex.allelecounts.tsv',
#              gatk_raw_gatk_coverage_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_hg38_wgs_1kb_gatk_no_sex_frag.counts.hdf5',
#              gatk_sequence_dictionary='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/Homo_sapiens_assembly38.dict',
#              gatk_count_panel='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/GATK_PoN_50samples_1kG.hdf5',
#
#              # UPDATE
#              facets_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_depths.tsv',
#              facets_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_filtered.tsv',
#
#              # UPDATE
#              ascat_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_depths.tsv',
#              ascat_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_filtered.tsv',
#              ascat_GC_correction_file = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/GC_G1000_hg38.txt',
#              ascat_RT_correction_file='gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/RT_G1000_hg38.txt'
#              )
