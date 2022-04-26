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
      "normal_allele_counts",
      "cytoband_file",
      "ref_file_path"    
    }
    script = """
    export CAPY_REF_FA=${ref_file_path}
    hapaseg load_snps --phased_VCF ${phased_VCF} \
            --allele_counts_T ${tumor_allele_counts} \
            --allele_counts_N ${normal_allele_counts} \
            --cytoband_file ${cytoband_file}
    """
    output_patterns = {
      "allele_counts" : "allele_counts.pickle",
      "scatter_chunks" : "scatter_chunks.tsv"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"

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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:all_SNPs_v617"

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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:all_SNPs_v617"

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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:all_SNPs_v617"

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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"

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
      "likelihood_trace_plot" : "figures/likelihood_trace.png",
      "SNP_plot" : "figures/SNPs.png",
      "seg_plot" : "figures/segs_only.png",
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:all_SNPs_v623"
    resources = { "mem" : "5G" }

class Hapaseg_prepare_coverage_mcmc(wolf.Task):
    inputs = {
        "coverage_csv": None,
        "allelic_clusters_object": None,
        "SNPs_pickle": None,
        "repl_pickle": None,
        "gc_pickle":"",
        "allelic_sample":"",
        "ref_file_path": None
    }
    script = """
    export CAPY_REF_FA=${ref_file_path}
    hapaseg coverage_mcmc_preprocess --coverage_csv ${coverage_csv} \
    --allelic_clusters_object ${allelic_clusters_object} \
    --SNPs_pickle ${SNPs_pickle} \
    --repl_pickle ${repl_pickle}"""
    
    def prolog(self):
        if self.conf["inputs"]["gc_pickle"] != "":
            self.conf["script"][-1] += " --gc_pickle ${gc_pickle}"
        if self.conf["inputs"]["allelic_sample"] != "":
            self.conf["script"][-1] += " --allelic_sample ${allelic_sample}"

    output_patterns = {
        "preprocess_data": "preprocess_data.npz",
        "cov_df_pickle": "cov_df.pickle"
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"
    resources = { "mem" : "15G" }


class Hapaseg_coverage_mcmc_burnin(wolf.Task):
    inputs = {
        "preprocess_data": None,
        "num_draws": 50,
        "cluster_num": None,
        "bin_width":None,
        "range":""
    }
    script = """
    hapaseg coverage_mcmc_shard --preprocess_data ${preprocess_data} \
    --num_draws ${num_draws} \
    --cluster_num ${cluster_num} \
    --bin_width ${bin_width}"""
     
    def prolog(self):
        if self.conf["inputs"]["range"] != "":
            self.conf["script"][-1] += " --range ${range}"
    
    output_patterns = {
        "burnin_model": 'cov_mcmc_model_cluster_*.pickle',
        "burnin_data": 'cov_mcmc_data_cluster_*.npz',
        "burnin_figure": 'cov_mcmc_cluster_*_visual.png'
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"
    resources = {"mem" : "5G"}

class Hapaseg_coverage_mcmc(wolf.Task):
    inputs = {
        "preprocess_data": None,
        "num_draws": 50,
        "cluster_num": None,
        "bin_width":None,
        "burnin_files":""
    }
    script = """
    hapaseg coverage_mcmc_shard --preprocess_data ${preprocess_data} \
    --num_draws ${num_draws} \
    --cluster_num ${cluster_num} \
    --bin_width ${bin_width}"""
     
    def prolog(self):
        if self.conf["inputs"]["burnin_files"] != "":
            self.conf["script"][-1] += " --burnin_files ${burnin_files}"
    
    output_patterns = {
        "cov_segmentation_model": 'cov_mcmc_model_cluster_*.pickle',
        "cov_segmentation_data": 'cov_mcmc_data_cluster_*.npz',
        "cov_seg_figure": 'cov_mcmc_cluster_*_visual.png'
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"
    resources = {"mem" : "5G"}

class Hapaseg_collect_coverage_mcmc(wolf.Task):
    inputs = {
        "cov_mcmc_files":None,
        "cov_df_pickle":None,
        "bin_width":1
     }

    script = """
    hapaseg collect_cov_mcmc --cov_mcmc_files ${cov_mcmc_files} --cov_df_pickle ${cov_df_pickle} --bin_width ${bin_width}
    """
    
    output_patterns={
        "cov_collected_data":'cov_mcmc_collected_data.npz'   
    }

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"


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
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"
    resources = {"mem" : "10G"} #potentially overkill and wont be necessary if cache table implemented

class Hapaseg_acdp_generate_df(wolf.Task):
    inputs = {
        "SNPs_pickle": None,
        "allelic_clusters_object" : None,
        "cdp_object" : "",
        "cdp_filepaths" : "",
        "allelic_draw_index" : -1,
        "ref_file_path": None,
        "bin_width":""
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
    
    output_patterns = {
        "acdp_df_pickle": "acdp_df.pickle"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"
    resources = {"mem" : "15G"}

class Hapaseg_run_acdp(wolf.Task):
    inputs = {
        "coverage_dp_object" : None,
        "acdp_df" : None,
        "num_samples" : None,
        "cytoband_file" : None
    }

    script = """
    hapaseg allelic_coverage_dp  --coverage_dp_object ${coverage_dp_object} \
    --acdp_df_path ${acdp_df} \
    --num_samples ${num_samples} \
    --cytoband_file ${cytoband_file}
    """

    output_patterns = {
        "acdp_model_pickle": "acdp_model.pickle",
        "acdp_clusters_plot": "acdp_clusters_plot.png",
        "acdp_genome_plot": "acdp_genome_plot.png",
        "acdp_tuples_plot": "acdp_tuples_plot.png"
    }
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_v623"
    resources = {"mem" : "15G"}
