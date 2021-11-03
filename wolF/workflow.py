import glob
import os
import pandas as pd
import pickle
import prefect
import tempfile
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

def workflow(
  callstats_file = None,
  common_snp_list = "gs://getzlab-workflows-reference_files-oa/hg38/gnomad/gnomAD_MAF10_50pct_45prob_hg38_final.txt",
):
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

    #
    # get het site coverage/genotypes from callstats
    hp_task = het_pulldown.get_het_coverage_from_callstats(
      callstats_file = callstats_file,
      common_snp_list = common_snp_list,
      ref_fasta = localization_task["ref_fasta"],
      ref_fasta_idx = localization_task["ref_fasta_idx"],
      ref_fasta_dict = localization_task["ref_fasta_dict"],
      dens_cutoff = 0.58 # TODO: set dynamically
    )

    #
    # shim task to convert output of het pulldown to VCF
    convert_task = wolf.Task(
      name = "convert_het_pulldown",
      inputs = {
        "genotype_file" : hp_task["normal_genotype"],
        "sample_name" : "test", # TODO: allow to be specified
        "ref_fasta" : localization_task["ref_fasta"],
        "ref_fasta_idx" : localization_task["ref_fasta_idx"],
        "ref_fasta_dict" : localization_task["ref_fasta_dict"],
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

    #
    # ensure that BCFs/indices/reference BCFs are in the same order
    @prefect.task
    def order_indices(bcf_path, bcf_idx_path, localization_task):
        # BCFs
        F = pd.DataFrame(dict(bcf_path = bcf_path))
        F = F.set_index(F["bcf_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

        # indices
        F2 = pd.DataFrame(dict(bcf_idx_path = bcf_idx_path))
        F2 = F2.set_index(F2["bcf_idx_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

        F = F.join(F2)

        # reference panel BCFs
        R = pd.DataFrame({ "path" : localization_task } ).reset_index()
        F = F.join(R.join(R.loc[R["index"].str.contains("^chr.*_bcf$"), "index"].str.extract(r"(?P<chr>chr[^_]+)"), how = "right").set_index("chr").drop(columns = ["index"]).rename(columns = { "path" : "ref_bcf" }), how = "inner")
        F = F.join(R.join(R.loc[R["index"].str.contains("^chr.*csi$"), "index"].str.extract(r"(?P<chr>chr[^_]+)"), how = "right").set_index("chr").drop(columns = ["index"]).rename(columns = { "path" : "ref_bcf_idx" }), how = "inner")

        return F

    F = order_indices(convert_task["bcf"], convert_task["bcf_idx"], localization_task)

    #
    # run Eagle, per chromosome
    eagle_task = phasing.eagle(
      inputs = dict(
        genetic_map_file = localization_task["genetic_map_file"],
        vcf_in = F["bcf_path"],
        vcf_idx_in = F["bcf_idx_path"],
        vcf_ref = F["ref_bcf"],
        vcf_ref_idx = F["ref_bcf_idx"],
        output_file_prefix = "foo"
      )
    )

    # TODO: run whatshap
    # when we include this, define combine_task without inputs and call it twice,
    # once for eagle, once for whatshap

    #
    # combine VCFs
    combine_task = wolf.Task(
      name = "combine_vcfs",
      inputs = { "vcf_array" : [eagle_task["phased_vcf"]] },
      script = "bcftools concat -O u $(cat ${vcf_array} | tr '\n' ' ') | bcftools sort -O v -o combined.vcf",
      outputs = { "combined_vcf" : "combined.vcf" },
      docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"
    )

    #
    # run HapASeg

    # load
    hapaseg_load_task = hapaseg.Hapaseg_load(
      inputs = {
        "phased_VCF" : combine_task["combined_vcf"],
        "tumor_allele_counts" : hp_task["tumor_hets"],
        "normal_allele_counts" : hp_task["normal_hets"],
        "cytoband_file" : "/mnt/j/db/hg38/ref/cytoBand_primary.txt" # TODO: allow to be specified
      }
    )

    # get intervals for burnin
    @prefect.task
    def get_chunks(scatter_chunks):
        return pd.read_csv(scatter_chunks, sep = "\t")

    chunks = get_chunks(hapaseg_load_task["scatter_chunks"])

    # burnin chunks
    hapaseg_burnin_task = hapaseg.Hapaseg_burnin(
     inputs = {
       "allele_counts" : hapaseg_load_task["allele_counts"],
       "start" : chunks["start"],
       "end" : chunks["end"]
     }
    )

    # concat burned in chunks, infer reference bias
    hapaseg_concat_task = hapaseg.Hapaseg_concat(
     inputs = {
       "chunks" : [hapaseg_burnin_task["burnin_MCMC"]],
       "scatter_intervals" : hapaseg_load_task["scatter_chunks"]
     }
    )

    # run on arms
    hapaseg_arm_AMCMC_task = hapaseg.Hapaseg_amcmc(
     inputs = {
       "amcmc_object" : hapaseg_concat_task["arms"],
       "ref_bias" : hapaseg_concat_task["ref_bias"]
     }
    )

    # concat arm level results
    @prefect.task
    def concat_arm_level_results(arm_results):
        A = []
        for arm_file in arm_results:
            with open(arm_file, "rb") as f:
                H = pickle.load(f)
                A.append(pd.Series({ "chr" : H.P["chr"].iloc[0], "start" : H.P["pos"].iloc[0], "end" : H.P["pos"].iloc[-1], "results" : H }))

        # get into order
        A = pd.concat(A, axis = 1).T.sort_values(["chr", "start", "end"]).reset_index(drop = True)

        # save
        _, tmpfile = tempfile.mkstemp(  )
        A.to_pickle(tmpfile) 

        return tmpfile

    arm_concat = concat_arm_level_results(hapaseg_arm_AMCMC_task["arm_level_MCMC"])

    # run DP
    hapaseg_allelic_DP_task = hapaseg.Hapaseg_allelic_DP(
     inputs = {
       "seg_dataframe" : arm_concat,
       #"n_dp_iter" : 10,   # TODO: allow to be specified?
       #"n_seg_samps" : 10,
       "ref_fasta" : localization_task["ref_fasta"],
       "ref_fasta_idx" : localization_task["ref_fasta_idx"],  # not used; just supplied for symlink
       "ref_fasta_dict" : localization_task["ref_fasta_dict"] # not used; just supplied for symlink
     }
    )

