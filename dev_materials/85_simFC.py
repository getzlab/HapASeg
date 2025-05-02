import wolf

mutect = wolf.ImportTask("github.com:getzlab/MuTect1_TOOL.git", "M1")

def workflow(
  bam, bai, vcf,
  refFasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
  refFastaIdx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
  refFastaDict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict"
):
    localize = wolf.LocalizeToDisk(
      files = {
        "bam" : bam,
        "bai" : bai,
        "vcf" : vcf,
        "refFasta" : refFasta,
        "refFastaIdx" : refFastaIdx,
        "refFastaDict" : refFastaDict
      }
    )

    split_vcf = wolf.Task(
      name = "split_vcf",
      inputs = { "vcf" : localize["vcf"] },
      script = """
      grep '^#' ${vcf} > header
      sed '/^#/d' ${vcf} | split -l 10000 -d -a 3 --filter='cat header /dev/stdin > $FILE' - VCF_chunk
      """,
      outputs = { "shards" : "VCF_chunk*" }
    )

    m1_scatter = mutect.mutect1(
      inputs = {
        "pairName" : "platinum",
        "caseName" : "platinum",
        "t_bam" : localize["bam"],
        "t_bai" : localize["bai"],
        "force_calling" : True,
        "intervals" : split_vcf["shards"],
        "fracContam" : 0,
        "refFasta" : localize["refFasta"],
        "refFastaIdx" : localize["refFastaIdx"],
        "refFastaDict" :  localize["refFastaDict"]
      }
    )

    m1_gather = wolf.Task(
      name = "m1_gather",
      inputs = { "callstats_array" : [m1_scatter["mutect1_cs"]] },
      script = """
      head -n2 $(head -n1 ${callstats_array}) > header
      while read -r i; do
        sed '1,2d' $i
      done < ${callstats_array} | sort -k1,1V -k2,2n > cs_sorted
      cat header cs_sorted > cs_concat.tsv
      """,
      outputs = { "cs_gather" : "cs_concat.tsv" }
    )

with wolf.Workflow(workflow = workflow, namespace = "HS_sim") as w:
    w.run(
      RUN_NAME = "NA12878_WGS_platinum_hg38",
      bam = "gs://jh-xfer/NA12878_bwamem_illumina_platinum_bed.bam",
      bai = "gs://jh-xfer/NA12878_bwamem_illumina_platinum_bed.bam.bai",
      vcf = "gs://jh-xfer/NA12878.vcf"
    )
