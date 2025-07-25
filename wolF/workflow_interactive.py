import glob
import os
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

# for Hapaseg itself
hapaseg = wolf.ImportTask(
  task_path = ".", # TODO: make remote
  task_name = "hapaseg"
)

# workflow scrap

#
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

#
# get het site coverage/genotypes from callstats
hp = het_pulldown.get_het_coverage_from_callstats(
  callstats_file = "gs://fc-secure-66f5eeb9-27c4-4e5c-b9d6-0519aca5889d/pair/05bd347a/05bd347a-3da7-4d1f-9bc6-4375226a0cb4_de3962db-0bd7-4126-85d9-da9ffe131088.MuTect1.call_stats.txt",
  common_snp_list = "gs://getzlab-workflows-reference_files-oa/hg38/gnomad/gnomAD_MAF10_50pct_45prob_hg38_final.txt",
 ref_fasta = localization["ref_fasta"],
 ref_fasta_idx = localization["ref_fasta_idx"],
 ref_fasta_dict = localization["ref_fasta_dict"],
 dens_cutoff = 0.58
)

hp_results = hp.run()

# for easy access, link these
os.symlink(hp_results["tumor_hets"], "genome/05bd347a.tumor_hets.tsv")

#
# shim task to convert output of het pulldown to VCF
convert = wolf.Task(
  name = "convert_het_pulldown",
  inputs = {
    "genotype_file" : hp_results["normal_genotype"],
    "sample_name" : "test",
    "ref_fasta" : localization["ref_fasta"],
    "ref_fasta_idx" : localization["ref_fasta_idx"],
    "ref_fasta_dict" : localization["ref_fasta_dict"],
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

#
# ensure that BCFs/indices/reference BCFs are in the same order

# BCFs
F = pd.DataFrame(dict(bcf_path = convert_results["bcf"]))
F = F.set_index(F["bcf_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

# indices
F2 = pd.DataFrame(dict(bcf_idx_path = convert_results["bcf_idx"]))
F2 = F2.set_index(F2["bcf_idx_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

F = F.join(F2)

# reference panel BCFs
R = pd.DataFrame({ "path" : localization } ).reset_index()
F = F.join(R.join(R.loc[R["index"].str.contains("^chr.*_bcf$"), "index"].str.extract(r"(?P<chr>chr[^_]+)"), how = "right").set_index("chr").drop(columns = ["index"]).rename(columns = { "path" : "ref_bcf" }), how = "inner")
F = F.join(R.join(R.loc[R["index"].str.contains("^chr.*csi$"), "index"].str.extract(r"(?P<chr>chr[^_]+)"), how = "right").set_index("chr").drop(columns = ["index"]).rename(columns = { "path" : "ref_bcf_idx" }), how = "inner")

#
# run Eagle, per chromosome
eagle = phasing.eagle(
  inputs = dict(
    genetic_map_file = localization["genetic_map_file"],
    vcf_in = F["bcf_path"],
    vcf_idx_in = F["bcf_idx_path"],
    vcf_ref = F["ref_bcf"],
    vcf_ref_idx = F["ref_bcf_idx"],
    output_file_prefix = "foo"
  )
)

eagle_results = eagle.run()

# TODO: run whatshap

#
# combine VCFs
combine = wolf.Task(
  name = "combine_vcfs",
  inputs = { "vcf_array" },
  script = "bcftools concat -O u $(cat ${vcf_array} | tr '\n' ' ') | bcftools sort -O v -o combined.vcf",
  outputs = { "combined_vcf" : "combined.vcf" },
  docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"
)

combined_results = combine.run(vcf_array = [eagle_results["phased_vcf"]])

#
# run HapASeg

# load
hapaseg_load = hapaseg.Hapaseg_load(
  inputs = {
    "phased_VCF" : combined_results["combined_vcf"],
    "tumor_allele_counts" : hp_results["tumor_hets"],
    "normal_allele_counts" : hp_results["normal_hets"],
    "cytoband_file" : "/mnt/j/db/hg38/ref/cytoBand_primary.txt"
  }
)

hapaseg_load_results = hapaseg_load.run()

# get intervals for burnin
chunks = pd.read_csv(hapaseg_load_results["scatter_chunks"], sep = "\t")

# burnin chunks
hapaseg_burnin = hapaseg.Hapaseg_burnin(
 inputs = {
   "allele_counts" : hapaseg_load_results["allele_counts"],
   "start" : chunks["start"],
   "end" : chunks["end"]
 }
)

hapaseg_burnin_results = hapaseg_burnin.run()

# concat burned in chunks, infer reference bias
hapaseg_concat = hapaseg.Hapaseg_concat(
 inputs = {
   "chunks" : [hapaseg_burnin_results["burnin_MCMC"]],
   "scatter_intervals" : hapaseg_load_results["scatter_chunks"]
 }
)

hapaseg_concat_results = hapaseg_concat.run()

# run on arms
hapaseg_arm_AMCMC = hapaseg.Hapaseg_amcmc(
 inputs = {
   "amcmc_object" : hapaseg_concat_results["arms"],
   "ref_bias" : hapaseg_concat_results["ref_bias"]
 }
)

hapaseg_arm_AMCMC_results = hapaseg_arm_AMCMC.run()

# concat arm level results
A = []
for arm_file in hapaseg_arm_AMCMC_results["arm_level_MCMC"]:
    with open(arm_file, "rb") as f:
        H = pickle.load(f)
        A.append(pd.Series({ "chr" : H.P["chr"].iloc[0], "start" : H.P["pos"].iloc[0], "end" : H.P["pos"].iloc[-1], "results" : H }))

# get into order
A = pd.concat(A, axis = 1).T.sort_values(["chr", "start", "end"]).reset_index(drop = True)

# save
# TODO: make temp file
A.to_pickle("arms.pickle")

# run DP


#
# coverage collection notes

split_intervals = wolf.ImportTask(
  task_path = "/home/jhess/Downloads/split_intervals_TOOL", # TODO: make remote
  task_name = "split_intervals"
)

tumor_bam_localization_task = wolf.localization.BatchLocalDisk(
  files = dict(
    bam = "/mnt/j/proj/cnv/20201018_hapseg2/exome/18144_6_C1D1_tissue_DNA.bam",
    bai = "/mnt/j/proj/cnv/20201018_hapseg2/exome/18144_6_C1D1_tissue_DNA.bai"
  ),
  run_locally = True
)

tumor_bam_localization = tumor_bam_localization_task.run()

# split target list

split_intervals_task = split_intervals.split_intervals(
  bam = tumor_bam_localization["bam"],
  bai = tumor_bam_localization["bai"],
  interval_type = "bed",
)

split_intervals_results = split_intervals_task.run()

# shim task to transform split_intervals files into subset parameters for covcollect task

#@prefect.task
def interval_gather(interval_files):
    ints = []
    for f in interval_files:
        ints.append(pd.read_csv(f, sep = "\t", header = None, names = ["chr", "start", "end"]))
    return pd.concat(ints).sort_values(["chr", "start", "end"])

subset_intervals = interval_gather(split_intervals_results["interval_files"])


# get coverage

cov_collect = wolf.ImportTask(
  task_path = "/mnt/j/proj/cnv/20210326_coverage_collector", # TODO: make remote
  task_name = "covcollect"
)

cov_collect_task = cov_collect.Covcollect(
  inputs = dict(
    bam = tumor_bam_localization["bam"],
    bai = tumor_bam_localization["bai"],
    intervals = "/mnt/j/proj/cnv/20201018_hapseg2/exome/broad_custom_exome_v1.Homo_sapiens_assembly19.targets.interval_list.noheader",
    subset_chr = subset_intervals["chr"],
    subset_start = subset_intervals["start"],
    subset_end = subset_intervals["end"],
  )
)

cov_collect_results = cov_collect_task.run()

# gather coverage
cov_gather = wolf.Task(
  name = "gather_coverage",
  inputs = { "coverage_beds" : [cov_collect_results["coverage"]] },
  script = """cat $(cat ${coverage_beds}) > coverage_cat.bed""",
  outputs = { "coverage" : "coverage_cat.bed" }
)

cov_gather_results = cov_gather.run()
