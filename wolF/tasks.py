import wolf

class Hapaseg(wolf.Task):
    inputs = {
      "phased_VCF",
      "tumor_allele_counts",
      "normal_allele_counts",
      "cytoband_file"
    }
    script = """
    hapaseg --phased_VCF ${phased_VCF} \
            --allele_counts_T ${tumor_allele_counts} \
            --allele_counts_N ${normal_allele_counts} \
            --cytoband_file ${cytoband_file} \
            --n_workers 8
    """
    resources = { "cpus-per-task" : 8 }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v455"

class Hapaseg_load_snps(wolf.Task):
    inputs = {
      "phased_VCF",
      "tumor_allele_counts",
      "normal_allele_counts"
    }
    script = """
    hapaseg load_snps --phased_VCF ${phased_VCF} \
            --allele_counts_T ${tumor_allele_counts} \
            --allele_counts_N ${normal_allele_counts} \
            --cytoband_file ${cytoband_file}
    """
    output_patterns = {
      "allele_counts" : "allele_counts.pickle",
      "scatter_chunks" : "scatter_chunks.tsv"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v488"

class Hapaseg_burnin(wolf.Task):
    inputs = {
      "allele_counts",
      "start",
      "end"
    }
    script = """
    hapaseg amcmc --snp_dataframe ${allele_counts} \
            --start ${start} \
            --end ${end} \
            --stop_after_burnin
    """
    output_patterns = {
      "burnin_MCMC" : "amcmc_results.pickle"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v458"

class Hapaseg_concat(wolf.Task):
    inputs = {
      "chunks",
      "scatter_intervals"
    }
    script = """
    CHUNKS_STR=$(cat ${chunks} | tr '\n' ' ')
    hapaseg concat --chunks $CHUNKS_STR --scatter_intervals ${scatter_intervals}
    """
    output_patterns = {
      "arms" : "AMCMC-arm*.pickle",
      "ref_bias" : ("ref_bias.txt", wolf.read_file)
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v458"

class Hapaseg_amcmc(wolf.Task):
    inputs = {
      "amcmc_object" : None,
      "ref_bias" : None,
      "n_iter" : 20000,
    }
    script = """
    hapaseg amcmc --amcmc_object ${amcmc_object} \
            --ref_bias ${ref_bias} \
            --n_iter ${n_iter}
    """
    output_patterns = {
      "arm_level_MCMC" : "amcmc_results.pickle"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v458"

class Hapaseg_allelic_DP(wolf.Task):
    inputs = {
      "seg_dataframe" : None,
      "n_dp_iter" : 10,
      "seg_samp_idx" : 0,
      "ref_fasta" : None,
      "cytoband_file" : None
    }
    script = """
    hapaseg dp --seg_dataframe ${seg_dataframe} \
            --n_dp_iter ${n_dp_iter} \
            --seg_samp_idx ${seg_samp_idx} \
            --ref_fasta ${ref_fasta} \
            --cytoband_file ${cytoband_file}
    """
    output_patterns = {
      "cluster_and_phase_assignments" : "allelic_DP_SNP_clusts_and_phase_assignments.npz",
      "all_SNPs" : "all_SNPs.pickle",
      "SNP_plot" : "figures/SNPs.png",
      "seg_plot" : "figures/allelic_imbalance_preDP.png",
      "clust_plot" : "figures/allelic_imbalance_postDP.png",
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v499"
    resources = { "mem" : "5G" }
