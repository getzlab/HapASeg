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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_v623"

class Hapaseg_load_snps(wolf.Task):
    inputs = {
      "phased_VCF" : None,
      "tumor_allele_counts" : None,
      "normal_allele_counts" : "",
      "cytoband_file" : None,
      "ref_file_path" : None    
    }
    def script(self):
        script = """
        export CAPY_REF_FA=${ref_file_path}
        hapaseg load_snps --phased_VCF ${phased_VCF} \
                --allele_counts_T ${tumor_allele_counts} \
                --cytoband_file ${cytoband_file}"""
        if self.conf["inputs"]["normal_allele_counts"] != "":
            script += " --allele_counts_N ${normal_allele_counts}"
        return script
    
    output_patterns = {
      "allele_counts" : "allele_counts.pickle",
      "scatter_chunks" : "scatter_chunks.tsv"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = { "cpus-per-task" : 2, "mem":"4G"}

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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"

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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"

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
      "arm_level_MCMC" : "amcmc_results.pickle",
      "segmentation_plot" : "figures/MLE_segmentation.png",
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"

class Hapaseg_concat_arms(wolf.Task):
    inputs = {
        "arm_results":None,
        "ref_fasta":None,
    }

    script = """
    export CAPY_REF_FA=${ref_fasta}
    hapaseg concat_arms --arm_results ${arm_results} 
    """
    output_patterns = {
    "arm_cat_results_pickle" : "arm_results.pickle",
    "num_samples_obj" : "num_arm_samples.np*"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"

class Hapaseg_allelic_DP(wolf.Task):
    inputs = {
      "seg_dataframe" : None,
      "ref_fasta" : None,
      "cytoband_file" : None
    }
    script = """
    export CAPY_REF_FA=${ref_fasta}
    hapaseg dp --seg_dataframe ${seg_dataframe} \
            --ref_fasta ${ref_fasta} \
            --cytoband_file ${cytoband_file}
    """
    output_patterns = {
      "cluster_and_phase_assignments" : "allelic_DP_SNP_clusts_and_phase_assignments.npz",
      "all_SNPs" : "all_SNPs.pickle",
      "segmentation_breakpoints" : "segmentations.pickle",
      "likelihood_trace_plot" : "figures/likelihood_trace.png",
      "SNP_plot" : "figures/SNPs.png",
      "seg_plot" : "figures/segs_only.png",
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = { "mem" : "4G" }

class Hapaseg_prepare_coverage_mcmc(wolf.Task):
    inputs = {
        "coverage_csv": None,
        "allelic_clusters_object": None,
        "SNPs_pickle": None,
        "segmentations_pickle": None,
        "repl_pickle": None,
        "faire_pickle": "", # TODO: make remote
        "normal_coverage_csv": "",
        "gc_pickle":"",
        "allelic_sample":"",
        "ref_fasta": None,
        "bin_width": 1,
        "wgs": True
    }
    def script(self):
        script = """
        hapaseg coverage_mcmc_preprocess --coverage_csv ${coverage_csv} \
        --ref_fasta ${ref_fasta} \
        --allelic_clusters_object ${allelic_clusters_object} \
        --SNPs_pickle ${SNPs_pickle} \
        --segmentations_pickle ${segmentations_pickle} \
        --repl_pickle ${repl_pickle} \
        --bin_width ${bin_width}"""

        if self.conf["inputs"]["wgs"] == True:
            script += " --wgs"
        if self.conf["inputs"]["faire_pickle"] != "":
            script += " --faire_pickle ${faire_pickle}"
        if self.conf["inputs"]["gc_pickle"] != "":
            script += " --gc_pickle ${gc_pickle}"
        if self.conf["inputs"]["normal_coverage_csv"] != "":
            script += " --normal_coverage_csv ${normal_coverage_csv}"
        if self.conf["inputs"]["allelic_sample"] != "":
            script += " --allelic_sample ${allelic_sample}"

        return script

    output_patterns = {
        "preprocess_data": "preprocess_data.npz",
        "cov_df_pickle": "cov_df.pickle",
        "allelic_seg_groups": "allelic_seg_groups.pickle",
        "allelic_seg_idxs": "allelic_seg_idxs.txt"
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = { "mem" : "4G" }

#scatter by allelic segment
class Hapaseg_coverage_mcmc_by_Aseg(wolf.Task):
    inputs = {
        "preprocess_data": None,
        "num_draws": 50,
        "allelic_seg_scatter_idx": None,
        "allelic_seg_indices":None,
        "bin_width":None,
        "range":""
    }
    script = """
    hapaseg coverage_mcmc_shard --preprocess_data ${preprocess_data} \
    --num_draws ${num_draws} \
    --allelic_seg_indices ${allelic_seg_indices} \
    --allelic_seg_idx ${allelic_seg_scatter_idx} \
    --bin_width ${bin_width}"""
     
    def prolog(self):
        if self.conf["inputs"]["range"] != "":
            self.conf["script"][-1] += " --range ${range}"
    
    output_patterns = {
        "cov_segmentation_model": 'cov_mcmc_model*.pickle',
        "cov_segmentation_data": 'cov_mcmc_data*.npz',
        "cov_seg_figure": 'cov_mcmc_*_visual.png'
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = {"cpus-per-task": 4, "mem" : "5G"}

class Hapaseg_coverage_mcmc(wolf.Task):
    inputs = {
        "preprocess_data": None,      # npz of covariate matrix (C), global beta, ADP cluster mu's, covbin ADP cluster assignments (all_mu), covbin raw coverage values (r)
        "allelic_seg_indices": None,  # dataframe containing indicies into C/r/all_mu for each allelic segment
        "allelic_seg_scatter_idx": None,      # allelic segment to operate on (for scatter)
        "num_draws": 50,
        "bin_width":None,
        "burnin_files":""
    }
    script = """
    hapaseg coverage_mcmc_shard --preprocess_data ${preprocess_data} \
    --allelic_seg_indices ${allelic_seg_idx} \
    --allelic_seg_idx ${allelic_seg_scatter_idx} \
    --num_draws ${num_draws} \
    --bin_width ${bin_width}"""
     
    def prolog(self):
        if self.conf["inputs"]["burnin_files"] != "":
            self.conf["script"][-1] += " --burnin_files ${burnin_files}"
    
    output_patterns = {
        "cov_segmentation_model": 'cov_mcmc_model_cluster_*.pickle',
        "cov_segmentation_data": 'cov_mcmc_data_cluster_*.npz',
        "cov_seg_figure": 'cov_mcmc_cluster_*_visual.png'
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = {"cpus-per-task": 4, "mem" : "5G"}

class Hapaseg_collect_coverage_mcmc(wolf.Task):
    inputs = {
        "cov_mcmc_files":None,
        "cov_df_pickle":None,
        "seg_indices_pickle":None,
        "bin_width":1,
        "cytoband_file":None
     }

    script = """
    hapaseg collect_cov_mcmc --seg_indices_pickle ${seg_indices_pickle} --cov_mcmc_files ${cov_mcmc_files} --cov_df_pickle ${cov_df_pickle} --bin_width ${bin_width} --cytoband_file ${cytoband_file}
    """
    
    output_patterns={
        "cov_collected_data":'cov_mcmc_collected_data.npz',
        "seg_plot":'figures/segs.png',
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = {"cpus-per-task": 4, "mem" : "12G"} # need high mem for poisson regression on massive Pi matrix

class Hapaseg_coverage_dp(wolf.Task):
    inputs = {
        "f_cov_df": None,
        "cov_mcmc_data": None,
        "num_segmentation_samples": 10,
        "num_dp_samples":10,
        "bin_width":"",
        "sample_idx":"",
    }

    script = """
    hapaseg coverage_dp --f_cov_df ${f_cov_df} \
    --cov_mcmc_data ${cov_mcmc_data} \
    --num_segmentation_samples ${num_segmentation_samples}\
    --num_dp_samples ${num_dp_samples}"""
    
    def prolog(self):
        if self.conf["inputs"]["sample_idx"] != "":
            self.conf["script"][-1] += " --sample_idx ${sample_idx}"
        if self.conf["inputs"]["bin_width"] != "":
            self.conf["script"][-1] += " --bin_width ${bin_width}"

    output_patterns = {
        "cov_dp_object" : "Cov_DP_model*",
        "cov_dp_figure" : "cov_dp_visual_draw*"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = {"cpus-per-task": 8, "mem" : "10G"} #potentially overkill and wont be necessary if cache table implemented

class Hapaseg_acdp_generate_df(wolf.Task):
    inputs = {
        "SNPs_pickle": None,
        "allelic_clusters_object" : None,
        "cdp_object" : "",
        "cov_df_pickle" : "",
        "cov_seg_data":"",
        "cdp_filepaths" : "",
        "allelic_draw_index" : -1,
        "ref_file_path": None,
        "bin_width":"",
        "wgs":""
    }

    script = """
    export CAPY_REF_FA=${ref_file_path}
    hapaseg generate_acdp_df --snp_dataframe ${SNPs_pickle} \
    --allelic_clusters_object ${allelic_clusters_object} \
    --allelic_draw_index ${allelic_draw_index}"""

    def prolog(self):
        if self.conf["inputs"]["bin_width"] != "":
            self.conf["script"][-1] += " --bin_width ${bin_width}"
        if self.conf["inputs"]["cdp_object"] != "":
            self.conf["script"][-1] += " --cdp_object ${cdp_object}"
        if self.conf["inputs"]["cdp_filepaths"] != "":
            self.conf["script"][-1] += " --cdp_filepaths ${cdp_filepaths}"
        if self.conf["inputs"]["cov_df_pickle"] != "":
            self.conf["script"][-1] += " --cov_df_pickle ${cov_df_pickle}"
        if self.conf["inputs"]["cov_seg_data"] != "":
            self.conf["script"][-1] += " --cov_seg_data ${cov_seg_data}"
        if self.conf["inputs"]["wgs"] != "":
            self.conf["script"][-1] += " --wgs"
    
    output_patterns = {
        "acdp_df_pickle": "acdp_df.pickle",
        "opt_cdp_idx" : ("opt_cdp_draw.txt", wolf.read_file),
        "lnp_data_pickle": "lnp_data.pickle"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = {"cpus-per-task": 2, "mem" : "3G"}

class Hapaseg_run_acdp(wolf.Task):
    inputs = {
        "cov_seg_data" : None,
        "acdp_df" : None,
        "num_samples" : None,
        "cytoband_file" : None,
        "opt_cdp_idx": None,
        "lnp_data_pickle":None,
        "wgs": False,
        "use_single_draw":""
    }
    
    def script(self):
        script = """
        hapaseg allelic_coverage_dp  --cov_seg_data ${cov_seg_data} \
        --acdp_df_path ${acdp_df} \
        --lnp_data_pickle ${lnp_data_pickle} \
        --num_samples ${num_samples} \
        --cytoband_file ${cytoband_file} \
        --opt_cdp_idx ${opt_cdp_idx}"""
        
        if self.conf["inputs"]["wgs"] == True:
            script += " --wgs"
        if self.conf["inputs"]["use_single_draw"] == True:
            script += " --use_single_draw"
        return script

    output_patterns = {
        "acdp_model_pickle": "acdp_model.pickle",
        "acdp_clusters_plot": "acdp_clusters_plot.png",
        "acdp_tuples_plot": "acdp_tuples_plot.png",
        "acdp_genome_plots": 'acdp*draws.png',
        "acdp_segfile" : "acdp_segfile.txt",
        "unclustered_segs": "unclustered_segs.txt",
        "opt_fit_params": "optimal_fit_params.txt"
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1024"
    resources = { "cpus-per-task":4, "mem" : "6G"}
