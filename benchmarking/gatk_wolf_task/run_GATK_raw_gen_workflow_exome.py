# coding: utf-8
from gatk_generate_raw_task import GATK_Generate_Raw_Data
import wolf
with wolf.Workflow(workflow = GATK_Generate_Raw_Data) as w:
    w.run(run_name = 'GATK_exome_NA12878_TWIST_36_raw_data_gen',
                       input_bam = 'gs://jh-xfer/realignment/NA12878_TWIST_36/NA12878_TWIST_36.bam',
                       input_bai = 'gs://jh-xfer/realignment/NA12878_TWIST_36/NA12878_TWIST_36.bam.bai',
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/broad_custom_exome_v1.Homo_sapiens_assembly38.targets.interval_list',
                       interval_name = 'hg38_broad_twist',
                       sample_name = 'NA12878_TWIST_36',
                       bin_length = 0, # for WES
                       padding=250,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/TWIST_36/',
                       preprocess_tumor_bam = True)

    w.run(run_name = 'GATK_exome_NA12878_TWIST_18_raw_data_gen',
                       input_bam = 'gs://jh-xfer/realignment/NA12878_TWIST_18/NA12878_TWIST_18.bam',
                       input_bai = 'gs://jh-xfer/realignment/NA12878_TWIST_18/NA12878_TWIST_18.bam.bai',
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/broad_custom_exome_v1.Homo_sapiens_assembly38.targets.interval_list',
                       interval_name = 'hg38_broad_twist',
                       sample_name = 'NA12878_TWIST_18',
                       bin_length = 0, # for WES
                       padding=250,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/GATK/TWIST_18/',
                       preprocess_tumor_bam=False)

                       
