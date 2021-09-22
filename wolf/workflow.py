import glob
import pandas as pd
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

# localize reference files to RODISK

ref_panel = pd.DataFrame({ "path" : glob.glob("/mnt/j/db/hg38/1kg/*.bcf*") })
ref_panel = ref_panel.join(ref_panel["path"].str.extract(".*(?P<chr>chr[^.]+)\.(?P<ext>bcf(?:\.csi)?)"))
ref_panel["key"] = ref_panel["chr"] + "_" + ref_panel["ext"]

localization_task = wolf.localization.BatchLocalDisk(
  files = dict(
    ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
    ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
    ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",

    genetic_map_file = "/home/jhess/Downloads/Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz",

    # reference panel
    **ref_panel.loc[:, ["key", "path"]].set_index("key")["path"].to_dict()
  ),
  run_locally = True
)

localization = localization_task.run()

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
  name = "convert_het_pulldown",
  inputs = {
    "genotype_file" : hp_results["normal_genotype"],
    "sample_name" : "test",
    "ref_fasta" : "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
    "ref_fasta_idx" : "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
    "ref_fasta_dict" : "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
  }, 
  script = r"""
set -x
bcftools convert --tsv2vcf ${genotype_file} -c CHROM,POS,AA -s ${sample_name} \
  -f ${ref_fasta} -Ou -o all_chrs.bcf && bcftools index all_chrs.bcf
for chr in $(bcftools view -h all_chrs.bcf | ssed -nR '/^##contig/s/.*ID=(.*),.*/\1/p' | head -n24); do
  bcftools view -Ou -r ${chr} -o ${chr}.chrsplit.bcf all_chrs.bcf && bcftools index ${chr}.chrsplit.bcf
done
""",
  outputs = {
    "bcf" : "*.chrsplit.bcf",
    "bcf_idx" : "*.chrsplit.bcf.csi"
  },
  docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"
)

convert_results = convert.run()

# convert.debug(**convert.conf["inputs"])

# ensure that chromosomes/indices/reference VCFs are in the same order
F = pd.DataFrame(dict(bcf_path = convert_results["bcf"]))
F = F.set_index(F["bcf_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

F2 = pd.DataFrame(dict(bcf_idx_path = convert_results["bcf_idx"]))
F2 = F2.set_index(F2["bcf_idx_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

F = F.join(F2)

# run Eagle, per chromosome
eagle = phasing.eagle(
  genetic_map_file = "/home/jhess/Downloads/Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz",
  vcf_in = convert_results["bcf"],
  vcf_idx_in = convert_results["bcf_idx"],
  vcf_ref = [],
)

def workflow():

phasing.eagle
