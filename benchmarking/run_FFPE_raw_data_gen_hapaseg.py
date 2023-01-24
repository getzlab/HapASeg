import wolf
from generate_raw_data import *
with wolf.Workflow(workflow = Generate_Hapaseg_Raw_Data_Workflow) as w:
    w.run(run_name = 'CH1022LN_NA12878_raw_data_gen_hapsaeg',
    sample_name = 'CH1022LN_NA12878',
    bam = 'gs://jh-xfer/realignment/CH1022LN/CH1022LN.bam',
    bai = 'gs://jh-xfer/realignment/CH1022LN/CH1022LN.bam.bai',
    ref_genome_build="hg38",
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 2000,
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/',
    preprocess=True,
    chrs_to_exclude="chr4 chr6 chr7 chr10 chr13",
    dummy_normal=True
    )

    # second sample
    w.run(run_name = 'CH1022GL_NA12878_raw_data_gen_hapsaeg',
    sample_name = 'CH1022GL_NA12878',
    bam = 'gs://jh-xfer/realignment/CH1022GL/CH1022GL.bam', 
    bai = 'gs://jh-xfer/realignment/CH1022GL/CH1022GL.bam.bai',
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 2000,
    ref_genome_build="hg38",
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022GL/',
    preprocess=False
    )
    
    w.run(run_name = 'CH1032LN_NA12878_raw_data_gen_hapsaeg',
    sample_name = 'CH1032LN_NA12878',
    bam = 'gs://jh-xfer/realignment/CH1032LN/CH1032LN.bam',
    bai = 'gs://jh-xfer/realignment/CH1032LN/CH1032LN.bam.bai',
    ref_genome_build="hg38",
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 2000,
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/',
    preprocess=True,
    chrs_to_exclude="chr9 chr14",
    dummy_normal=True
    )

    # second sample
    w.run(run_name = 'CH1032GL_NA12878_raw_data_gen_hapsaeg',
    sample_name = 'CH1032GL_NA12878',
    bam = 'gs://jh-xfer/realignment/CH1032GL/CH1032GL.bam', 
    bai = 'gs://jh-xfer/realignment/CH1032GL/CH1032GL.bam.bai',
    vcf = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf',
    ref_genome_build="hg38",
    ascat_loci_list = 'gs://opriebe-tmp/HapASeg/benchmarking/ascat_loci/G1000_loci_hg38.txt',
    db_snp_vcf = 'https://ftp.ncbi.nih.gov/snp/organisms/human_9606_b151_GRCh38p7/VCF/00-common_all.vcf.gz',
    target_list = 2000,
    upload_bucket_gs_path = 'gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032GL/',
    preprocess=False
    )
