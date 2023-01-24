import wolf
from hatchet_wolf_task.workflow import Hatchet_Generate_Raw

with wolf.Workflow(workflow = Hatchet_Generate_Raw) as w:
    w.run(run_name = "Hatchet_NA12878_paired",
          ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
          ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
          ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
          reference_genome_version="hg38",
          tumor_bam = 'gs://jh-xfer/realignment/NA12878_TWIST_36/NA12878_TWIST_36.bam',
          tumor_bai = 'gs://jh-xfer/realignment/NA12878_TWIST_36/NA12878_TWIST_36.bam.bai',
          normal_bam = 'gs://jh-xfer/realignment/NA12878_TWIST_18/NA12878_TWIST_18.bam', 
          normal_bai = 'gs://jh-xfer/realignment/NA12878_TWIST_18/NA12878_TWIST_18.bam.bai',
          normal_vcf_path = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf.gz",
          normal_vcf_idx = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf.gz.csi",
          sample_names = "NA12878_TWIST_36 NA12878_TWIST_18",
          phase_snps=True,
          upload_bucket = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/",
          sample_name = "NA12878_TWIST_36_18"
          )
