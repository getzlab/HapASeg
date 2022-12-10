import wolf
from benchmarking_workflows import *
import glob
import numpy as np
import os

# 50 profiles
profile_paths = glob.glob('/home/opriebe/data/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_*.pickle')
purity_range = np.r_[0.1:1:0.1]

with wolf.Workflow(workflow=Run_Sim_Workflows, namespace="benchmarking_dir", scheduler_processes=3) as w:
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
                  gatk_sim_normal_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/NA12878_platnium_realigned_gatk_sim_normal_allele_counts.tsv',
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
              gatk_sim_normal_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/CH1022LN_gatk_sim_normal_allele_counts.tsv',
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
              hatchet_phased_vcf = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/phased.vcf.gz"      
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
              gatk_sim_normal_allelecounts_path='gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/CH1032LN_gatk_sim_normal_allele_counts.tsv',
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
              hatchet_phased_vcf = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/hatchet/phased.vcf.gz"      
        )

