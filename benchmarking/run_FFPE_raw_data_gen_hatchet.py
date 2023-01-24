import wolf
from hatchet_wolf_task.workflow import Hatchet_Generate_Raw

chromosomes_1032 = ['chr' + str(c) for c in list(set(range(1,23)) - set([9,14]))]
chromosomes_1032_string = ' '.join(chromosomes_1032)

chromosomes_1022 = ['chr' + str(c) for c in list(set(range(1,23)) - set([4,6,7,10,13]))]
chromosomes_1022_string = ' '.join(chromosomes_1022)

with wolf.Workflow(workflow = Hatchet_Generate_Raw) as w:
    w.run(run_name = "Hatchet_FFPE_NA12878_platinum_CH1032LN_CH1032GL",
          ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
          ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
          ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
          reference_genome_version="hg38",
          tumor_bam = 'gs://jh-xfer/realignment/CH1032GL/CH1032GL.bam',
          tumor_bai = 'gs://jh-xfer/realignment/CH1032GL/CH1032GL.bam.bai', 
          second_tumor_bam = 'gs://jh-xfer/realignment/CH1032LN/CH1032LN.bam',
          second_tumor_bai = 'gs://jh-xfer/realignment/CH1032LN/CH1032LN.bam.bai', 
          normal_bam = "gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam", 
          normal_bai = "gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam.bai",
          normal_vcf_path = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf.gz",
          normal_vcf_idx = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf.gz.csi",
          sample_names = "NA12878_platinum CH1032GL CH1032LN",
          chromosomes = chromosomes_1032_string,
          phase_snps=True,
          sample_name = "NA12878_platinum_CH1032GL_CH1032LN",
          upload_bucket = "gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1032LN/"
          )
    
    w.run(run_name = "Hatchet_FFPE_NA12878_platinum_CH1022LN_GH1022GL",
          ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
          ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
          ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
          reference_genome_version="hg38",
          tumor_bam = 'gs://jh-xfer/realignment/CH1022GL/CH1022GL.bam',
          tumor_bai = 'gs://jh-xfer/realignment/CH1022GL/CH1022GL.bam.bai', 
          second_tumor_bam = 'gs://jh-xfer/realignment/CH1022LN/CH1022LN.bam',
          second_tumor_bai = 'gs://jh-xfer/realignment/CH1022LN/CH1022LN.bam.bai', 
          normal_bam = "gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam", 
          normal_bai = "gs://jh-xfer/realignment/NA12878_platinum/NA12878_platinum.bam.bai",
          normal_vcf_path = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf.gz",
          normal_vcf_idx = "gs://opriebe-tmp/HapASeg/benchmarking/NA12878_wgs/NA12878.vcf.gz.csi",
          sample_names = "NA12878_platinum CH1022GL CH1022LN",
          chromosomes = chromosomes_1022_string,
          phase_snps=True,
          sample_name = "NA12878_platinum_CH1022GL_CH1022LN",
          upload_bucket = "gs://opriebe-tmp/HapASeg/benchmarking/FFPE/CH1022LN/"
          )
