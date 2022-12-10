import pandas as pd
import wolf

alignment = wolf.ImportTask(
  "git@github.com:getzlab/alignment_pipeline_wolF.git",
  "alignment"
)

def align_and_upload(sample_name = None,
                     fastq_1 = None,
                     fastq_2 = None,
                     bwa_index_dir = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/",
                     readgroups = None):
    bam = alignment.alignment_workflow(fastq_1, bwa_index_dir, readgroups, fastq2s = fastq_2,  n_shard = 30)
    
#    reheader_task = wolf.Task(name="reset_readgroups",
#              inputs ={"input_bam":'rodisk://canine-scratch-prayvma-ypljbha-3muskudfizvhu-0/merged.bam'},
#              script = """samtools addreplacerg -r SM:NA12878 -r ID:SRR6691666 -r LB:1 -r PL:illumina -r PU:1 -w -m overwrite_all ${input_bam} -o replaced_readgroups.bam""", 
#              outputs={"new_rg_bam":"replaced_readgroups.bam"}, 
#              docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.6",
#              use_scratch_disk = True)
#   processed_bam, processed_bai = alignment.postprocess_workflow(reheader_task["new_rg_bam"], sample_name)
    processed_bam, processed_bai = alignment.postprocess_workflow(bam, sample_name)

    wolf.UploadToBucket(
      files = [processed_bam, processed_bai],
      bucket = f"gs://opriebe_tmp/{sample_name}/"
    )

with wolf.Workflow(workflow=align_and_upload, scheduler_processes = 3) as w:
    w.run(RUN_NAME = "SRR6691666_alignment",
          sample_name = "NA12878_SRR6691666",
          fastq_1 = "rodisk://sra-download-disk/SRR6691666_1.fastq",
          fastq_2 = "rodisk://sra-download-disk/SRR6691666_2.fastq",
          readgroups= r"@RG\tID:SRR6691666\tSM:NA12878\tLB:1\tPL:illumina\tPU:1")
