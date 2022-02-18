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

class Hapaseg_coverage_mcmc(wolf.Task):
    inputs = {
        "coverage_bed" : None,
        "allelic_clusters_object" : None,
        "SNPs_pickle" : None,
        "covariate_dir" : None,
        "num_draws" : 50,
        "cluster_num" : None,
        "allelic_sample" : None,
        "ref_file_path":None
    }
    script = """
    export CAPY_REF_FA=${ref_file_path}
    hapaseg coverage_mcmc --coverage_csv ${coverage_bed} \
    --allelic_clusters_object ${allelic_clusters_object} \
    --SNPs_pickle ${SNPs_pickle} \
    --covariate_dir ${covariate_dir} \
    --num_draws ${num_draws} \
    --cluster_num ${cluster_num} \
    --allelic_sample ${allelic_sample}
    """
     
    output_patterns = {
        "cov_segmentation_model":'coverage_mcmc_clusters/cov_mcmc_model_cluster_*.pickle',
        "cov_segmentation_data": 'coverage_mcmc_clusters/cov_mcmc_data_cluster_*.npz',
        "cov_df_segmentation": 'coverage_mcmc_clusters/cov_df.pickle',
        "cov_seg_figures":'coverage_mcmc_clusters/figures/cov_mcmc_cluster_*_visual.png'
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v608"
    resources = {"mem" : "5G"}

class Hapaseg_collect_coverage_mcmc(wolf.Task):
    inputs = {
        "cov_mcmc_files":None,
        "cov_df_pickle":None
     }

    script = """
    hapaseg collect_cov_mcmc --cov_mcmc_files ${cov_mcmc_files} --cov_df_pickle ${cov_df_pickle}
    """
    
    output_patterns={
        "cov_collected_data":'cov_mcmc_collected_data.npz'   
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v608"
class Hapaseg_coverage_dp(wolf.Task):
    inputs = {
        "f_cov_df": None,
        "cov_mcmc_data": None,
        "num_segmentation_samples": 50,
        "num_draws": 10
    }

    script = """
    hapaseg coverage_dp --f_cov_df ${f_cov_df} \
    --cov_mcmc_data ${cov_mcmc_data} \
    --num_segmentation_samples ${num_segmentation_samples} \
    --num_draws ${num_draws}
    """

    output_patterns = {
        "cov_dp_object" : "Cov_DP_model.pickle",
        "cov_dp_figure" : "coverage_figures/coverage_draw*"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v608"

class Hapaseg_acdp_generate_df(wolf.Task):
    inputs = {
        "SNPs_pickle": None,
        "allelic_clusters_object" : None,
        "coverage_dp_object" : None,
        "allelic_draw_index" : -1,
        "ref_file_path": None
    }

    script = """
    export CAPY_REF_FA=${ref_file_path}
    hapaseg generate_acdp_df --snp_dataframe ${SNPs_pickle} \
    --coverage_dp_object ${coverage_dp_object} \
    --allelic_clusters_object ${allelic_clusters_object} \
    --allelic_draw_index ${allelic_draw_index}
    """

    output_patterns = {
        "acdp_df_pickle": "acdp_df.pickle"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v608"

class Hapaseg_run_acdp(wolf.Task):
    inputs = {
        "coverage_dp_object" : None,
        "acdp_df" : None,
        "num_samples" : None,
        "cytoband_df" : None
    }

    script = """
    hapaseg allelic_coverage_dp  --coverage_dp_object ${coverage_dp_object} \
    --acdp_df_path ${acdp_df} \
    --num_samples ${num_samples} \
    --cytoband_dataframe ${cytoband_df}
    """

    output_patterns = {
        "acdp_model_pickle": "acdp_model.pickle"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v608"
