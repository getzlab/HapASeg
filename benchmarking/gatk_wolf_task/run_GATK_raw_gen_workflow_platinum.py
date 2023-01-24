# coding: utf-8
from gatk_generate_raw_task import GATK_Generate_Raw_Data
import wolf
with wolf.Workflow(workflow = GATK_Generate_Raw_Data) as w:
    w.run(run_name = 'GATK_generate_raw_realigned',
                       input_bam = 'gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam',
                       input_bai = 'gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam.bai',
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/wgs_calling_regions.hg38.interval_list',
                       interval_name = 'hg38_1kG_wgs',
                       sample_name = 'NA12878_platnium_realigned',
                       bin_length = 1000,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/platinum/',
                       persistent_dry_run = True,
                       preprocess_tumor_bam = True)

#    w.run(run_name = 'GATK_generate_raw_SRR',
#                       input_bam = 'gs:',
#                       input_bai = 'gs:',
#                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
#                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
#                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
#                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
#                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/wgs_calling_regions.hg38.interval_list',
#                       interval_name = 'hg38_1kG_wgs',
#                       sample_name = 'NA12878_SRR6691666',
#                       bin_length = 1000,
#                       exclude_sex = True,
#                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/SRR6691666/',
#                       persistent_dry_run = True,
#                       preprocess_tumor_bam=False)

                       
