# coding: utf-8
import wolf
from generate_raw_data import *
with wolf.Workflow(workflow = Generate_Hapaseg_Raw_Data_Workflow) as w:
    w.run(run_name = 'exome_NA12878_TWIST_36_raw_data_gen_hapsaeg',
    sample_name = 'NA12878_TWIST_36',
    bam = 'gs://jh-xfer/realignment/NA12878_TWIST_36/NA12878_TWIST_36.bam',
    bai = 'gs://jh-xfer/realignment/NA12878_TWIST_36/NA12878_TWIST_36.bam.bai',
    ref_genome_build="hg38",
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/broad_custom_exome_v1.Homo_sapiens_assembly38.targets.bed',
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/',
    preprocess=True
    )

    # second sample
    w.run(run_name = 'exome_NA12878_TWIST_18_raw_data_gen_hapaseg',
    sample_name = 'NA12878_TWIST_18',
    bam = 'gs://jh-xfer/realignment/NA12878_TWIST_18/NA12878_TWIST_18.bam',
    bai = 'gs://jh-xfer/realignment/NA12878_TWIST_18/NA12878_TWIST_18.bam.bai', 
    ref_genome_build="hg38",
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/broad_custom_exome_v1.Homo_sapiens_assembly38.targets.bed',
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/TWIST_18/',
    preprocess=False
    )

