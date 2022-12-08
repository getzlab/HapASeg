import wolf

# generate ground truth seg file
class Generate_Groundtruth_Segfile(wolf.Task):
    inputs = {"sample_label" :None,
              "purity":None,
              "sim_profile":None,
              "normal_vcf_path": None,
              "hapaseg_hetsite_depth_path":None,
              "hapaseg_coverage_tsv":None,
              }

    script = """
    python3 -c "import pandas as pd; cnv_profile=pd.read_pickle('${sim_profile}');\
                cnv_profile.generate_profile_seg_file('./${sample_label}_${purity}_gt_seg_file.tsv',
                '${normal_vcf_path}', '${hapaseg_hetsite_depth_path}', '${hapaseg_coverage_tsv}', float(${purity}),
                do_parallel=False)"
            """
    output_patterns = {"ground_truth_seg_file":"*_gt_seg_file.tsv"}
   
    resources = {"mem":"2G"}
 
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# HapASeg
class Downstream_HapASeg_Analysis(wolf.Task):
    inputs = {"hapaseg_seg_file": None,
              "ground_truth_seg_file":None,
              "sample_name":None,
              "sim_profile":None,
              "ref_fasta":None,
              "cytoband_file":None,
             }
    
    script = """
    compare_outputs.py --sim_profile ${sim_profile} --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . hapaseg --hapaseg_seg_file ${hapaseg_seg_file}
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
              "sim_profile":None,
              "gatk_seg_file": None,
              "ground_truth_seg_file": None,
              "sample_name": None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
    
    script = """
    compare_outputs.py --sim_profile ${sim_profile} --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . gatk --gatk_sim_cov_input ${gatk_sim_cov_input} --gatk_sim_acounts ${gatk_sim_acounts} --gatk_seg_file ${gatk_seg_file}
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
              "sim_profile":None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
    
    script = """
    compare_outputs.py --sim_profile ${sim_profile} --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . ascat --ascat_t_logr ${ascat_t_logr} --ascat_t_baf ${ascat_t_baf} --ascat_seg_file ${ascat_seg_file}
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
              "sim_profile":None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
    
    script = """
    compare_outputs.py --sim_profile ${sim_profile} --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . facets --facets_input_counts ${facets_input_counts} --facets_seg_file ${facets_seg_file} 
    """

    output_patterns = {"comparison_plot": "*_facets_comparison_plot.png",
                       "comparison_results" : "*_facets_comparison_results.txt",
                       "comparison_segfile": "*_facets_comparison_segfile.tsv",
                       "input_plot": "*_facets_input_plot.png"
                      }

    resources = {"cpus-per-task": 4, "mem":"14G"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

class Downstream_Hatchet_Analysis(wolf.Task):
    inputs = {"hatchet_seg_file": None,
              "hatchet_bin_file": None,
              "ground_truth_seg_file": None,
              "sample_name": None,
              "sim_profile":None,
              "ref_fasta": None,
              "cytoband_file": None,
             }
              
    script = """
    compare_outputs.py --sim_profile ${sim_profile} --ref_fasta ${ref_fasta} --cytoband_file ${cytoband_file} --sample_name ${sample_name} --ground_truth_segfile ${ground_truth_seg_file} --outdir . hatchet --hatchet_seg_file ${hatchet_seg_file} --hatchet_bin_file ${hatchet_bin_file} 
    """

    output_patterns = {"comparison_plot": "*_hatchet_comparison_plot.png",
                       "comparison_results" : "*_hatchet_comparison_results.txt",
                       "comparison_segfile": "*_hatchet_comparison_segfile.tsv",
                      }

    resources = {"mem":"2G"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"
