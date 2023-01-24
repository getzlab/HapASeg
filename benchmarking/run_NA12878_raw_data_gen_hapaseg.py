# coding: utf-8
import wolf
from generate_raw_data import *
with wolf.Workflow(workflow = Generate_Hapaseg_Raw_Data_Workflow) as w:
    w.run(run_name = 'NA12878_raw_data_gen_hapsaeg_realign',
    sample_name = 'NA12878_platinum_realigned',
    bam = 'gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam',
    bai = 'gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam.bai',
    ref_genome_build="hg38",
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 2000,
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/',
    preprocess=True
    )

    # second sample
    w.run(run_name = 'NA12878_SRR6691666_raw_data_gen_hapaseg',
    sample_name = 'NA12878_SRR6691666',
    bam = 'rodisk://canine-scratch-i1azvfa-ypljbha-qdycd1xbkod0c-0/NA12878_SRR6691666.bam',
    bai = 'rodisk://canine-scratch-i1azvfa-ypljbha-qdycd1xbkod0c-0/NA12878_SRR6691666.bam.bai', 
    ref_genome_build="hg38",
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 2000,
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/SRR6691666/',
    preprocess=False
    )
    
