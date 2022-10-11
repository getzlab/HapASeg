import wolf

# HapASeg
class Downstream_HapASeg_Analysis(wolf.Task):
    inputs = {"hapaseg_seg_file": None,
              "ground_truth_seg_file":None,
              "sample_name":None,
              "ref_fasta":None,
              "cytoband_file":None,
             }
    
    script = """
    compare_outputs.py --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . hapaseg --hapaseg_seg_file ${hapaseg_seg_file}
    """

    output_patterns = {"comparison_plot": "*_hapaseg_comparison_plot.png",
                       "comparison_results" : "*_hapaseg_comparison_results.txt",
                       "comparison_segfile": "*_hapaseg_comparison_segfile.tsv"
                      }

    resources = {"mem":"2G"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# GATK
class Downstream_GATK_Analysis(wolf.Task):
    inputs = {"gatk_sim_cov_input": None,
              "gatk_sim_acounts": None,
              "gatk_seg_file": None,
              "ground_truth_seg_file": None,
              "sample_name": None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
    
    script = """
    compare_outputs.py --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . gatk --gatk_sim_cov_input ${gatk_sim_cov_input} --gatk_sim_acounts ${gatk_sim_acounts} --gatk_seg_file ${gatk_seg_file}
    """

    output_patterns = {"comparison_plot": "*_gatk_comparison_plot.png",
                       "comparison_results" : "*_gatk_comparison_results.txt",
                       "comparison_segfile": "*_gatk_comparison_segfile.tsv",
                       "input_plot": "*_gatk_input_plot.png"
                      }

    resources = {"mem":"2G"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# ASCAT
class Downstream_ASCAT_Analysis(wolf.Task):
    inputs = {"ascat_t_logr": None,
              "ascat_t_baf": None,
              "ascat_seg_file": None,
              "ground_truth_seg_file": None,
              "sample_name": None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
    
    script = """
    compare_outputs.py --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . ascat --ascat_t_logr ${ascat_t_logr} --ascat_t_baf ${ascat_t_baf} --ascat_seg_file ${ascat_seg_file}
    """

    output_patterns = {"comparison_plot": "*_ascat_comparison_plot.png",
                       "comparison_results" : "*_ascat_comparison_results.txt",
                       "comparison_segfile": "*_ascat_comparison_segfile.tsv",
                       "input_plot": "*_ascat_input_plot.png"
                      }

    resources = {"mem":"2G"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# Facets
class Downstream_Facets_Analysis(wolf.Task):
    inputs = {"facets_input_counts": None,
              "facets_seg_file": None,
              "ground_truth_seg_file": None,
              "sample_name": None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
    
    script = """
    compare_outputs.py --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . facets --facets_input_counts ${facets_input_counts} --facets_seg_file ${facets_seg_file} 
    """

    output_patterns = {"comparison_plot": "*_facets_comparison_plot.png",
                       "comparison_results" : "*_facets_comparison_results.txt",
                       "comparison_segfile": "*_facets_comparison_segfile.tsv",
                       "input_plot": "*_facets_input_plot.png"
                      }

    resources = {"mem":"2G"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

