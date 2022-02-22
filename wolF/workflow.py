import glob
import numpy as np
import os
import pandas as pd
import pickle
import prefect
import subprocess
import tempfile
import wolf

#
# import tasks

# for genotyping het sites/getting het site coverage
het_pulldown = wolf.ImportTask(
  task_path = 'git@github.com:getzlab/het_pulldown_from_callstats_TOOL.git',
  task_name = "het_pulldown"
)

mutect1 = wolf.ImportTask(
  task_path = "git@github.com:getzlab/MuTect1_TOOL.git",
  task_name = "mutect1"
)

# for phasing
phasing = wolf.ImportTask(
  task_path = "git@github.com:getzlab/phasing_TOOL.git",
  task_name = "phasing"
)

# for Hapaseg itself
hapaseg = wolf.ImportTask(
  task_path = ".", # TODO: make remote
  task_name = "hapaseg"
)

# for coverage collection
split_intervals = wolf.ImportTask(
  task_path = "git@github.com:getzlab/split_intervals_TOOL.git",
  task_name = "split_intervals"
)

cov_collect = wolf.ImportTask(
  task_path = "git@github.com:getzlab/coverage_collector_TOOL.git", # TODO: make remote
  task_name = "covcollect"
)

####
# defining reference config generators for hg19 and hg38

#hg19
def _hg19_config_gen(coverage_dir):
    hg19_ref_panel = pd.DataFrame({ "path" : subprocess.check_output("gsutil ls gs://getzlab-workflows-reference_files-oa/hg19/1000genomes/*.bcf*", shell = True).decode().rstrip().split("\n") })
    hg19_ref_panel = ref_panel.join(ref_panel["path"].str.extract(".*(?P<chr>chr[^.]+)\.(?P<ext>bcf(?:\.csi)?)"))
    hg19_ref_panel["key"] = ref_panel["chr"] + "_" + ref_panel["ext"]
    hg19_ref_dict = ref_panel.loc[:, ["key", "path"]].set_index("key")["path"].to_dict()

    hg19_ref_config = dict(
        ref_fasta ="gs://getzlab-workflows-reference_files-oa/hg19/Homo_sapiens_assembly19.fasta",
        ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg19/Homo_sapiens_assembly19.fasta.fai",
        ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg19/Homo_sapiens_assembly19.dict",
        genetic_map_file = "gs://getzlab-workflows-reference_files-oa/hg19/eagle/genetic_map_hg19_withX.txt.gz",
        common_snp_list = "gs://getzlab-workflows-reference_files-oa/hg19/gnomad/gnomAD_MAF10_80pct_45prob.txt",
        cytoband_file = 'gs://getzlab-workflows-reference_files-oa/hg19/cytoBand.txt',
        coverage_dir = coverage_dir,
        ref_panel_1000g = hg19_ref_dict
    )
    return hg19_ref_config

#hg38
def _hg38_config_gen(coverage_dir):
    hg38_ref_panel = pd.DataFrame({ "path" : subprocess.check_output("gsutil ls gs://getzlab-workflows-reference_files-oa/hg38/1000genomes/*.bcf*", shell = True).decode().rstrip().split("\n") })
    hg38_ref_panel = ref_panel.join(ref_panel["path"].str.extract(".*(?P<chr>chr[^.]+)\.(?P<ext>bcf(?:\.csi)?)"))
    hg38_ref_panel["key"] = ref_panel["chr"] + "_" + ref_panel["ext"]
    hg38_ref_dict = ref_panel.loc[:, ["key", "path"]].set_index("key")["path"].to_dict()

    hg38_ref_config= dict(
        ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
        ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
        ref_fasta_dict = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
        genetic_map_file = "gs://getzlab-workflows-reference_files-oa/hg38/eagle/genetic_map_hg38_withX.txt.gz",
        common_snp_list = "gs://getzlab-workflows-reference_files-oa/hg38/gnomad/gnomAD_MAF10_50pct_45prob_hg38_final.txt",
        cytoband_file= 'gs://getzlab-workflows-reference_files-oa/hg19/cytoBand.txt',
        coverage_dir=coverage_dir,
        ref_panel_1000g = hg38_ref_panel
    )
    return hg38_ref_config

def workflow(
  callstats_file = None,

  tumor_bam = None,
  tumor_bai = None,
  tumor_coverage_bed = None,

  normal_bam = None,
  normal_bai = None,
  normal_coverage_bed = None,

  ref_genome_build=None, #must be hg19 or hg38
  
  target_list = None,
  coverage_dir = None
):
    
    ###
    # Select config based on ref genome choice
    if ref_genome_build is None:
        raise ValueError("Reference genome must be specified! Options are 'hg19' or hg38'")
    
    elif ref_genome_build == "hg19":
        ref_config = _hg19_config_gen(coverage_dir)
    elif ref_genome_build == "hg38":
        ref_config = _hg19_config_gen(coverage_dir)
    else:
        raise ValueError("Reference genome options are 'hg19' or hg38', got {}".format(ref_genome_build))
        
    # localize reference files to RODISK
    ref_panel = pd.DataFrame({ "path" : subprocess.check_output("gsutil ls gs://getzlab-workflows-reference_files-oa/hg38/1000genomes/*.bcf*", shell = True).decode().rstrip().split("\n") })
    ref_panel = ref_panel.join(ref_panel["path"].str.extract(".*(?P<chr>chr[^.]+)\.(?P<ext>bcf(?:\.csi)?)"))
    ref_panel["key"] = ref_panel["chr"] + "_" + ref_panel["ext"]

    localization_task = wolf.LocalizeToDisk(
      files = dict(
        ref_fasta = ref_config["ref_fasta"],
        ref_fasta_idx = ref_config["ref_fasta_idx"],
        ref_fasta_dict = ref_config["ref_fasta_dict"],

        genetic_map_file = ref_config["genetic_map_file"],
          
        cytoband_file = ref_config["cytoband_file"],
        # reference panel
        **ref_config["ref_panel_1000g"]
      )
    )

    #
    # localize BAMs to RODISK
    if tumor_bam is not None and tumor_bai is not None:
        tumor_bam_localization_task = wolf.LocalizeToDisk(
          files = {
            "bam" : tumor_bam,
            "bai" : tumor_bai,
          }
        )
        collect_tumor_coverage = True
    elif tumor_coverage_bed is not None:
        collect_tumor_coverage = False
    else:
        raise ValueError("You must supply either a tumor BAM+BAI or a tumor coverage BED file!")

    use_normal_coverage = True
    if normal_bam is not None and normal_bai is not None:
        normal_bam_localization_task = wolf.LocalizeToDisk(
          files = {
            "bam" : normal_bam,
            "bai" : normal_bai
          }
        )
        collect_normal_coverage = True
    elif normal_coverage_bed is not None:
        collect_normal_coverage = False
    else:
        print("Normal coverage will not be used as a covariate; ability to regress out germline CNVs may suffer.")
        use_normal_coverage = False

    #
    # collect or load coverage

    # tumor
    if collect_tumor_coverage:
        # create scatter intervals
        split_intervals_task = split_intervals.split_intervals(
          bam = tumor_bam_localization_task["bam"],
          bai = tumor_bam_localization_task["bai"],
          interval_type = "bed",
        )

        # shim task to transform split_intervals files into subset parameters for covcollect task
        @prefect.task
        def interval_gather(interval_files):
            ints = []
            for f in interval_files:
                ints.append(pd.read_csv(f, sep = "\t", header = None, names = ["chr", "start", "end"]))
            return pd.concat(ints).sort_values(["chr", "start", "end"])

        subset_intervals = interval_gather(split_intervals_task["interval_files"])

        # dispatch coverage scatter
        tumor_cov_collect_task = cov_collect.Covcollect(
          inputs = dict(
            bam = tumor_bam_localization_task["bam"],
            bai = tumor_bam_localization_task["bai"],
            intervals = target_list,
            subset_chr = subset_intervals["chr"],
            subset_start = subset_intervals["start"],
            subset_end = subset_intervals["end"],
          )
        )

        # gather tumor coverage
        tumor_cov_gather_task = wolf.Task(
          name = "gather_coverage",
          inputs = { "coverage_beds" : [tumor_cov_collect_task["coverage"]] },
          script = """cat $(cat ${coverage_beds}) > coverage_cat.bed""",
          outputs = { "coverage" : "coverage_cat.bed" }
        )

    # load from supplied BED file
    else:
        tumor_cov_gather_task = { "coverage" : tumor_coverage_bed }

    # normal
    #if collect_normal_coverage:

    #
    # get het site coverage/genotypes from callstats
    if callstats_file is not None:
        hp_task = het_pulldown.get_het_coverage_from_callstats(
          callstats_file = callstats_file,
          common_snp_list = common_snp_list,
          ref_fasta = localization_task["ref_fasta"],
          ref_fasta_idx = localization_task["ref_fasta_idx"],
          ref_fasta_dict = localization_task["ref_fasta_dict"],
          dens_cutoff = 0.58 # TODO: set dynamically
        )

    # otherwise, run M1 and get it from the BAM
    elif callstats_file is None and tumor_bam is not None and normal_bam is not None:
        m1_task = mutect1.mutect1(inputs = dict(
          pairName = "het_coverage",
          caseName = "tumor",
          ctrlName = "normal",

          t_bam = tumor_bam_localization_task["bam"],
          t_bai = tumor_bam_localization_task["bai"],
          n_bam = normal_bam_localization_task["bam"],
          n_bai = normal_bam_localization_task["bai"],

          fracContam = 0,

          refFasta = localization_task["ref_fasta"],
          refFastaIdx = localization_task["ref_fasta_idx"],
          refFastaDict = localization_task["ref_fasta_dict"],

          intervals = split_intervals_task["interval_files"]
        ))

        hp_scatter = het_pulldown.get_het_coverage_from_callstats(
          callstats_file = m1_task["mutect1_cs"],
          common_snp_list = common_snp_list,
          ref_fasta = localization_task["ref_fasta"],
          ref_fasta_idx = localization_task["ref_fasta_idx"],
          ref_fasta_dict = localization_task["ref_fasta_dict"],
          dens_cutoff = 0.58 # TODO: set dynamically
        )

        # gather het pulldown
        hp_task = wolf.Task(
          name = "hp_gather",
          inputs = {
            "tumor_hets" : [hp_scatter["tumor_hets"]],
            "normal_hets" : [hp_scatter["normal_hets"]],
            "normal_genotype" : [hp_scatter["normal_genotype"]],
          },
          script = """
          cat <(cat $(head -n1 ${normal_genotype}) | head -n1) \
            <(for f in $(cat ${normal_genotype}); do sed 1d $f; done | sort -k1,1V -k2,2n) > normal_genotype.txt
          cat <(cat $(head -n1 ${normal_hets}) | head -n1) \
            <(for f in $(cat ${normal_hets}); do sed 1d $f; done | sort -k1,1V -k2,2n) > normal_hets.txt
          cat <(cat $(head -n1 ${tumor_hets}) | head -n1) \
            <(for f in $(cat ${tumor_hets}); do sed 1d $f; done | sort -k1,1V -k2,2n) > tumor_hets.txt
          """,
          outputs = {
            "tumor_hets" : "tumor_hets.txt",
            "normal_hets" : "normal_hets.txt",
            "normal_genotype" : "normal_genotype.txt",
          }
        )

    else:
        raise ValueError("You must either provide a callstats file or tumor+normal BAMs to collect SNP coverage")

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

    # load SNPs
    hapaseg_load_snps_task = hapaseg.Hapaseg_load_snps(
      inputs = {
        "phased_VCF" : combine_task["combined_vcf"],
        "tumor_allele_counts" : hp_task["tumor_hets"],
        "normal_allele_counts" : hp_task["normal_hets"],
        "cytoband_file" : "/mnt/j/db/hg38/ref/cytoBand_primary.txt", # TODO: allow to be specified
      }
    )

    # get intervals for burnin
    @prefect.task
    def get_chunks(scatter_chunks):
        return pd.read_csv(scatter_chunks, sep = "\t")

    chunks = get_chunks(hapaseg_load_snps_task["scatter_chunks"])

    # burnin chunks
    hapaseg_burnin_task = hapaseg.Hapaseg_burnin(
     inputs = {
       "allele_counts" : hapaseg_load_snps_task["allele_counts"],
       "start" : chunks["start"],
       "end" : chunks["end"]
     }
    )

    # concat burned in chunks, infer reference bias
    hapaseg_concat_task = hapaseg.Hapaseg_concat(
     inputs = {
       "chunks" : [hapaseg_burnin_task["burnin_MCMC"]],
       "scatter_intervals" : hapaseg_load_snps_task["scatter_chunks"]
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
    @prefect.task(nout = 2)
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

        # get number of MCMC samples
        n_samps = int(np.minimum(np.inf, A.loc[~A["results"].isna(), "results"].apply(lambda x : len(x.breakpoint_list))).min())

        return tmpfile, list(range(0, n_samps))

    arm_concat, n_samps_range = concat_arm_level_results(hapaseg_arm_AMCMC_task["arm_level_MCMC"])

    ## run DP

    # scatter DP
    hapaseg_allelic_DP_task = hapaseg.Hapaseg_allelic_DP(
     inputs = {
       "seg_dataframe" : arm_concat,
       "n_dp_iter" : 10,   # TODO: allow to be specified?
       "seg_samp_idx" : n_samps_range,
       "cytoband_file" : "/mnt/j/db/hg38/ref/cytoBand_primary.txt", # TODO: allow to be specified
       "ref_fasta" : localization_task["ref_fasta"],
       "ref_fasta_idx" : localization_task["ref_fasta_idx"],  # not used; just supplied for symlink
       "ref_fasta_dict" : localization_task["ref_fasta_dict"] # not used; just supplied for symlink
     }
    )
    
    ### coverage tasks ####
    
    # prepare coverage MCMC
    prep_cov_mcmc_task = hapaseg.Hapaseg_prepare_coverage_mcmc(
    inputs={
        "coverage_csv":tumor_cov_gather_task["coverage"],
        "allelic_clusters_object":hapaseg_allelic_DP_task["cluster_and_phase_assignments"],
        "SNPs_pickle":hapaseg_allelic_DP_task["all_SNPs"],
        "covariate_dir":localization_task["covariate_dir"],
        "ref_file_path":localization_task["ref_fasta"]
        }
    )
    
    # coverage MCMC scatter
    num_clusters = np.load(prep_cov_mcmc_res["preprocess_data"])["Pi"].shape[1]
    cluster_idxs = [i for i in np.arange(num_clusters)]
    cov_mcmc_scatter_task = hapaseg.Hapaseg_coverage_mcmc(
        inputs={
            "preprocess_data":prep_cov_mcmc_task["preprocess_data"],
            "num_draws":10,
            "cluster_num":cluster_idxs
        }
    )
    
    # collect coverage MCMC
    cov_mcmc_gather_task = hapaseg.Hapaseg_collect_coverage_mcmc(
    inputs = {
        "cov_mcmc_files":[cov_mcmc_scatter_task["cov_segmentation_data"]],
        "cov_df_pickle":prep_cov_mcmc_task["cov_df_pickle"]
        }
    )
    
    # coverage DP
    cov_dp_task = hapaseg.Hapaseg_coverage_dp(
    inputs = {
        "f_cov_df":cov_mcmc_scatter_task["cov_df_segmentation"][0],
        "cov_mcmc_data": cov_mcmc_gather_task["cov_collected_data"],
        "num_segmentation_samples":10,
        "num_draws":10
        }   
    )
    
    # generate acdp dataframe
    adp_draw_num = int(np.load(prep_cov_mcmc_res["preprocess_data"])["adp_cluster"])
    gen_acdp_task = hapaseg.Hapaseg_acdp_generate_df(
    inputs = {
        "SNPs_pickle":hapaseg_allelic_DP_task["all_SNPs"],
        "allelic_clusters_object":hapaseg_allelic_DP_task["cluster_and_phase_assignments"],
        "coverage_dp_object":cov_dp_task["cov_dp_object"],
        "allelic_draw_index":adp_draw_num,
        "ref_file_path":localization_task["ref_fasta"]
        }
    )
    
    # run acdp
    acdp_task = hapaseg.Hapaseg_run_acdp(
    inputs = {
        "coverage_dp_object":cov_dp_task["cov_dp_object"],
        "acdp_df":gen_acdp_task["acdp_df_pickle"],
        "num_samples":10,
        "cytoband_file": localization_task["cytoband_file"]
        }
    )
    
    
