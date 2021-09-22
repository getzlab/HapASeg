import wolf

#
# import tasks

# for genotyping het sites/getting het site coverage
het_pulldown = wolf.ImportTask(
  #task_path = 'git@github.com:getzlab/het_pulldown_from_callstats_TOOL.git',
  task_path = '/home/jhess/j/proj/cnv/20200909_hetpull',
  task_name = "het_pulldown"
)

# for phasing
phasing = wolf.ImportTask(
  task_path = "/home/jhess/j/proj/cnv/20210901_phasing_TOOL", # TODO: make remote
  task_name = "phasing"
)

# workflow scrap

# get het site coverage/genotypes from callstats
hp = het_pulldown.get_het_coverage_from_callstats(
  callstats_file = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/pair/05bd347a/05bd347a-3da7-4d1f-9bc6-4375226a0cb4_de3962db-0bd7-4126-85d9-da9ffe131088.MuTect1.call_stats.txt",
  common_snp_list = "gs://getzlab-workflows-reference_files-oa/hg38/gnomad/gnomAD_MAF10_50pct_45prob_hg38_final.txt",
 ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
 ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
 ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
 dens_cutoff = 0.58
)

hp_results = hp.run()

# shim task to convert output of het pulldown to VCF
convert = wolf.Task(
  inputs = {
    "genotype_file" : hp_results["normal_genotype"],
    "sample_name" : "test",
    "ref_fasta" : "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
    "ref_fasta_idx" : "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
    "ref_fasta_dict" : "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
  }, 
  script = """
bcftools convert --tsv2vcf ${genotype_file} -c CHROM,POS,AA -s ${sample_name} \
  -f ${ref_fasta} -Oz -o ${sample_name}.vcf.gz && \
tabix ${sample_name}.vcf.gz
""",
  outputs = {
    "vcf" : "*.vcf.gz",
    "vcf_idx" : "*.vcf.gz.tbi"
  }
)

convert_results = convert.run()

# run Eagle, per chromosome

eagle = phasing.eagle(
  genetic_map_file = "",
  vcf_in = convert_results["vcf"],
  vcf_idx_in = convert_results["vcf_idx"],
)

def workflow():

phasing.eagle
