# coding: utf-8
from gatk_generate_raw_task import GATK_Generate_Raw_Data
import wolf
with wolf.Workflow(workflow = GATK_Generate_Raw_Data) as w:
    w.run(run_name = 'GATK_generate_raw_FFPE_CH1022LN',
                       input_bam = "gs://jh-xfer/realignment/CH1022LN/CH1022LN.bam",
                       input_bai = "gs://jh-xfer/realignment/CH1022LN/CH1022LN.bam.bai",
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/wgs_calling_regions.hg38.interval_list',
                       exclude_chroms = "chr4 chr6 chr7 chr10 chr13",
                       interval_name = 'hg38_1kG_wgs',
                       sample_name = 'CH1022LN',
                       bin_length = 1000,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/GATK/',
                       
                       persistent_dry_run = False,
                       preprocess_tumor_bam = True)

    w.run(run_name = 'GATK_generate_raw_FFPE_CH1032LN',
                       input_bam = 'gs://jh-xfer/realignment/CH1032LN/CH1032LN.bam',
                       input_bai = 'gs://jh-xfer/realignment/CH1032LN/CH1032LN.bam.bai',
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/wgs_calling_regions.hg38.interval_list',
                       exclude_chroms = "chr9 chr14",
                       interval_name = 'hg38_1kG_wgs',
                       sample_name = 'CH1032LN',
                       bin_length = 1000,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/GATK/',
               
                    
                       persistent_dry_run = False,
                       preprocess_tumor_bam=True)

    w.run(run_name = 'GATK_generate_raw_FFPE_CH1022GL',
                       input_bam = 'gs://jh-xfer/realignment/CH1022GL/CH1022GL.bam',
                       input_bai = 'gs://jh-xfer/realignment/CH1022GL/CH1022GL.bam.bai',
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/wgs_calling_regions.hg38.interval_list',
                       exclude_chroms = "chr4 chr6 chr7 chr10 chr13",
                       interval_name = 'hg38_1kG_wgs',
                       sample_name = 'CH1022GL',
                       bin_length = 1000,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022GL/',
               
                    
                       persistent_dry_run = False,
                       preprocess_tumor_bam=False)

    w.run(run_name = 'GATK_generate_raw_FFPE_CH1032GL',
                       input_bam = 'gs://jh-xfer/realignment/CH1032GL/CH1032GL.bam',
                       input_bai = 'gs://jh-xfer/realignment/CH1032GL/CH1032GL.bam.bai',
                       vcf_file = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
                       ref_fasta = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
                       ref_fasta_idx = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
                       ref_fasta_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
                       interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/GATK/wgs_calling_regions.hg38.interval_list',
                       exclude_chroms = "chr9 chr14",
                       interval_name = 'hg38_1kG_wgs',
                       sample_name = 'CH1032GL',
                       bin_length = 1000,
                       exclude_sex = True,
                       upload_bucket = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032GL/',
               
                    
                       persistent_dry_run = False,
                       preprocess_tumor_bam=False)
                
                
