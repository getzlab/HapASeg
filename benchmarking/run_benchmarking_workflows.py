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

