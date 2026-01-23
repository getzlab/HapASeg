#!/usr/bin/env python
import glob
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
import tempfile
import click
import re
import yaml
from pathlib import Path

# Fix the relative import to work when run as a script
try:
    # When run as a module
    from .ref_file_config import create_ref_file_dict
    from .run_external_methods import *
    from .hapaseg_wrappers import *
except ImportError:
    # When run as a script
    from hapaseg_local.ref_file_config import create_ref_file_dict
    from hapaseg_local.run_external_methods import *
    from hapaseg_local.hapaseg_wrappers import *

def results_exist(results_dict):
    """
    Check if all files in the results dictionary exist.
    
    Args:
        results_dict: Dictionary with result names as keys and file paths as values.
                     Values can be either a single file path or a list of file paths.
    
    Returns:
        True if all files exist, False otherwise.
    """
    for value in results_dict.values():
        if isinstance(value, list):
            # If value is a list, check if all files in the list exist
            if not all(Path(file).exists() for file in value):
                return False
        else:
            # If value is a single file path
            if not Path(value).exists():
                return False
    return True

def job_resources(job_cpus, job_memory, max_cpus, max_memory):
    return int(min(max_cpus / job_cpus,  max_memory / job_memory))

def hapaseg_local_main(
    sample_name = None,
    ref_root_path = None, 
    out_dir = None,

    tumor_bam = None,
    tumor_bai = None,

    normal_bam = None,
    normal_bai = None,

    genotyping_method = "mixture_model",

    single_ended = False, # Flag for designating if sequencing was done with single ended reads. Coverage collection differs depending on whether BAM is paired end
    ref_genome_build=None, # reference genome build string. Must be hg19 or hg38

    WES_target_list = None, # WES exome bait set list
    WGS_bin_size = 2000, # WGS bin size default: 2000 base pairs
    common_snp_list = None, # for adding a custom SNP list
    betahyp = 4, # hyperparameter for smoothing initial allelic segmentation. only applicable for whole exomes.

    num_cov_seg_samples=5, # number of draws from the MCMC chain to use during coverage segmentation. Default: 5 is recommended.
    run_cdp=False, # option to run coverage DP on WES data

    phased_vcf=None, # if running for benchmarking, can skip phasing by passsing vcf

    is_ffpe = False, # flag to use FAIRE as covariate for coverage model
    is_cfdna = False,  # flag to use FAIRE (w/ cfDNA samples) as covariate for coverage model

    extra_covariate_beds = None, # list of filepaths to BED files containing additional covariates to use for coverage model
    ref_file_override_dict = {}, # dictionary of reference file variable name keys and file path values to override the default ref file config

    max_cpus = 8,
    max_memory = 16,
):

     # integer target list implies wgs
    wgs = True if WES_target_list is None else False

    ref_config = create_ref_file_dict(ref_root_path, ref_genome_build, ref_file_override_dict)

    # hack to account for "chr" in hg38 but not in hg19
    if ref_genome_build == "hg38":
        primary_contigs = ['chr{}'.format(i) for i in range(1,23)]
        primary_contigs.extend(['chrX','chrY','chrM'])
    else:
        primary_contigs = [str(x) for x in range(1, 23)] + ["X", "Y", "M"]

    out_path = Path(out_dir)
    out_path = out_path.joinpath(sample_name)
    out_path.mkdir(exist_ok=True, parents=True)

    # RUN COVCOLLECT ON TUMOR, NORMAL and Mutect on T/N
    covcollct_normal_out_path = out_path.joinpath('CovCollect_N_Results')
    covcollct_normal_out_path.mkdir(exist_ok=True)
    covcollct_tumor_out_path = out_path.joinpath('CovCollect_T_Results')
    covcollct_tumor_out_path.mkdir(exist_ok=True)
    mutect_out_path = out_path.joinpath('Mutect_Results')
    mutect_out_path.mkdir(exist_ok=True)
    
    covcollect_n_script, covcollect_n_results_dict = make_covcollect_script(covcollct_normal_out_path,
                                                                            normal_bam,
                                                                            normal_bai,
                                                                            WGS_bin_size if wgs else WES_target_list,
                                                                            single_ended)
    covcollect_t_script, covcollect_t_results_dict = make_covcollect_script(covcollct_tumor_out_path,
                                                                            tumor_bam,
                                                                            tumor_bai,
                                                                            WGS_bin_size if wgs else WES_target_list,
                                                                            single_ended)

    mutect_script, mutect_results_dict = make_mutect_script(out_dir= mutect_out_path,
                                                            n_bam = normal_bam, 
                                                            n_bai = normal_bai,
                                                            t_bam = tumor_bam,
                                                            t_bai = tumor_bai,
                                                            
                                                            # references
                                                            refFasta = ref_config['ref_fasta'],
                                                            refFastaIdx = ref_config['ref_fasta_idx'],
                                                            refFastaDict = ref_config['ref_fasta_dict'],
                                                            intervals = ref_config['common_snp_list'])

    # Check which tasks need to be run
    scripts_to_run = {}
    if not results_exist(covcollect_n_results_dict):
        scripts_to_run['covcollect_n_script'] = covcollect_n_script
        
    if not results_exist(covcollect_t_results_dict):
        scripts_to_run['covcollect_t_script'] = covcollect_t_script
        
    if not results_exist(mutect_results_dict):
        scripts_to_run['mutect_script'] = mutect_script
    
    # Run only the tasks that need to be executed
    if scripts_to_run:
        print(f"Running {len(scripts_to_run)} tasks (skipping {3 - len(scripts_to_run)} completed tasks)...")
        run_gnu_parallel_tasks_backend(scripts_to_run, out_path.joinpath('cov_mutect_runlog'), max_jobs=job_resources(4, 4, max_cpus, max_memory))
    else:
        print("All CovCollect and Mutect results already exist. Skipping execution.")

    # RUN HET COVERAGE
    hetpull_results_dict = run_single_task(
        out_path,
        'Het_pulldown',
        make_het_pulldown_script,
        dict(
            callstats_file = mutect_results_dict['mutect1_cs'],
            ref_fasta = ref_config['ref_fasta'],
            ref_fasta_idx = ref_config['ref_fasta_idx'],
            ref_fasta_dict = ref_config['ref_fasta_dict'],
            method = genotyping_method,
            common_snp_list = ref_config['common_snp_list'],
            pod_min_depth = 10 if wgs else 4, # normal min genotyping depth; set lower for exomes due to bait falloff (normal coverage in flanking regions will be proportionally much lower than tumor coverage)
            min_tumor_depth = 1 if wgs else 10 # tumor min coverage; set higher for exomes due to off-target signal being noisier
        )
    )
    
    # # convert hetupull to vcfs
    convert_hp_results_dict = run_single_task(out_path,
                                        'Het_2_vcfs',
                                        convert_hetpull_2_vcf_script,
                                        dict(genotype_file=hetpull_results_dict['normal_genotype'],
                                        ref_fasta=ref_config['ref_fasta'],
                                        ref_fasta_idx=ref_config['ref_fasta_idx'],
                                        ref_fasta_dict=ref_config['ref_fasta_dict']))
    
    # RUN EAGLE
    ## first we need to organize the bcfs and their corresponding indices
    # ensure that BCFs/indices/reference BCFs are in the same order
    def order_indices(ref_config, convert_hp_results_dict):
        # reference panel BCFs
        ref_bcfs = Path(ref_config['ref_1kG']).glob('ALL*.bcf')
        R = pd.DataFrame(dict(ref_bcf_path = ref_bcfs))
        R = R.set_index(R["ref_bcf_path"].apply(lambda x: re.search(r"^ALL.((?:chr)?(?:[^.]+)).*", os.path.basename(x)).groups()[0]))

        # reference panel indices
        ref_bcfs_idx = Path(ref_config['ref_1kG']).glob('ALL*.csi')
        R2 = pd.DataFrame(dict(ref_bcf_idx_path = ref_bcfs_idx))
        R2 = R2.set_index(R2["ref_bcf_idx_path"].apply(lambda x: re.search(r"^ALL.((?:chr)?(?:[^.]+)).*", os.path.basename(x)).groups()[0]))

        R = R.join(R2)
        
        # BCFs
        F = pd.DataFrame(dict(bcf_path = convert_hp_results_dict['bcfs']))
        F = F.set_index(F["bcf_path"].apply(lambda x: re.search(r"^((?:chr)?(?:[^.]+)).*", os.path.basename(x)).groups()[0]))

        # indices
        F2 = pd.DataFrame(dict(bcf_idx_path = convert_hp_results_dict['bcf_idxs']))
        F2 = F2.set_index(F2["bcf_idx_path"].apply(lambda x: re.search(r"^((?:chr)?(?:[^.]+)).*", os.path.basename(x)).groups()[0]))

        F = F.join(F2)

        # prepend "chr" to F's index if it's missing
        idx = ~F.index.str.contains("^chr")
        if idx.any():
            new_index = F.index.values
            new_index[idx] = "chr" + F.index[idx]
            F = F.set_index(new_index)

        # join with reference panel
        F = F.join(R, how = "left")
        return F

    F = order_indices(ref_config, convert_hp_results_dict)

    # run Eagle, per chromosome using run_parallel_tasks
    # Filter out rows with missing values
    valid_chroms = F.dropna(subset=["ref_bcf_path", "bcf_path"])
    
    # Prepare arguments for run_parallel_tasks
    eagle_script_args = {
        # Lists for chromosome-specific arguments
        'vcf_in': valid_chroms["bcf_path"].tolist(),
        'vcf_idx_in': valid_chroms["bcf_idx_path"].tolist(),
        'vcf_ref': valid_chroms["ref_bcf_path"].tolist(),
        'vcf_ref_idx': valid_chroms["ref_bcf_idx_path"].tolist(),
        'output_file_prefix': [f"eagle_{chrom}" for chrom in valid_chroms.index],
        
        # Consistent arguments across all chromosomes
        'genetic_map_file': ref_config["genetic_map_file"],
        'num_threads': 2
    }
    
    # Run Eagle in parallel if there are chromosomes to process
    if eagle_script_args['vcf_in']:
        eagle_outputs = run_parallel_tasks(
            out_path=out_path,
            task_name='Eagle',
            script_generator=make_eagle_script,
            script_args=eagle_script_args,
            max_jobs=job_resources(2, 3, max_cpus, max_memory)
        )
    else:
        print("No chromosomes available for Eagle phasing. Skipping execution.")
        eagle_outputs = {"phased_vcf": []}

    # Use run_single_task function to combine Eagle VCFs
    comb_vcfs_results_dict = run_single_task(
        out_path=out_path,
        task_name='Comb_phased_vcfs',
        script_generator=combine_eagle_vcfs_script,
        script_args={
            'vcf_paths': eagle_outputs['phased_vcf']
        },
        check_for_existing=True
    )

    # run HapASeg

    # Use run_single_task function to run HapASeg load SNPs
    hapaseg_load_snps_results_dict = run_single_task(
        out_path=out_path,
        task_name='Hapaseg_load_snps',
        script_generator=make_hapaseg_load_snps_script,
        script_args={
            'phased_VCF': comb_vcfs_results_dict['combined_vcf'],
            'tumor_allele_counts': hetpull_results_dict['tumor_hets'],
            'cytoband_file': ref_config['cytoband_file'],
            'ref_file_path': ref_config['ref_fasta'],
            'normal_allele_counts': hetpull_results_dict['normal_hets']
        },
        check_for_existing=True
    )

    scatter_chunks_df = pd.read_csv(hapaseg_load_snps_results_dict["scatter_chunks"], sep = "\t")

    hapaseg_burnin_results_dict = run_parallel_tasks(
            out_path=out_path,
            task_name='Hapaseg_burnin',
            script_generator=make_hapaseg_burnin_script,
            script_args={
                'allele_counts': hapaseg_load_snps_results_dict["allele_counts"],
                'start': scatter_chunks_df["start"].tolist(),
                'end': scatter_chunks_df["end"].tolist(),
                'betahyp': -1 if wgs else 0
            },
            max_jobs=job_resources(1, 2, max_cpus, max_memory)
    )

    # concat burned in chunks, infer reference bias
    hapaseg_concat_results_dict = run_single_task(
        out_path=out_path,
        task_name='Hapaseg_concat',
        script_generator=make_hapaseg_concat_script,
        script_args={
            'chunks': hapaseg_burnin_results_dict["burnin_MCMC"],
            'scatter_intervals': hapaseg_load_snps_results_dict["scatter_chunks"]
        },
        check_for_existing=True
    )

    ref_bias = float(open(hapaseg_concat_results_dict["ref_bias"], 'r').read())

    # run on arms
    hapaseg_amcmc_results_dict = run_parallel_tasks(
        out_path=out_path,
        task_name='Hapaseg_amcmc',
        script_generator=make_hapaseg_amcmc_script,
        script_args={
            'amcmc_object': hapaseg_concat_results_dict["arms"],
            'ref_bias': ref_bias,
            'n_iter': 20000,
            'betahyp': -1 if wgs else betahyp
        },
        max_jobs=job_resources(1, 2, max_cpus, max_memory)
    )

    # concat arm level results
    hapseg_concat_arm_results_path = out_path.joinpath("Hapaseg_concat_arms_Results")
    hapseg_concat_arm_results_path.mkdir(exist_ok=True, parents=True)
    A = []
    for arm_file in hapaseg_amcmc_results_dict["arm_level_MCMC"]:
        with open(arm_file, 'rb') as f:
            H = pickle.load(f)
            A.append(pd.Series({ 'chr' : H.P['chr'].iloc[0], 'start' : H.P['pos'].iloc[0], 'end' : H.P['pos'].iloc[-1], 'results' : H }))

    # get into order
    A = pd.concat(A, axis = 1).T.sort_values(['chr', 'start', 'end']).reset_index(drop = True)

    # save
    hapaseg_concat_arm_results_output_path = hapseg_concat_arm_results_path.joinpath("concat_arms.pickle")
    A.to_pickle(hapaseg_concat_arm_results_output_path)
    hapaseg_concat_arm_results_dict = {"concat_arms" : hapaseg_concat_arm_results_output_path}

    ## run DP

    # run allelic dynamic programming
    hapaseg_allelic_dp_results_dict = run_single_task(
        out_path=out_path,
        task_name='Hapaseg_allelic_DP',
        script_generator=make_hapaseg_allelic_dp_script,
        script_args={
            'seg_dataframe': hapaseg_concat_arm_results_dict["concat_arms"],
            'ref_fasta': ref_config["ref_fasta"],
            'cytoband_file': ref_config["cytoband_file"],
            'wgs': wgs
        }
    )

    #
    # coverage tasks
    #

    # bin width sets an exposure term for the coverage MCMC. In general this exposure term is non needed, and hence should be set to 1 = exp(0)
    bin_width = 1 

    # prepare coverage MCMC
    hapaseg_prepare_coverage_mcmc_results_dict = run_single_task(
        out_path=out_path,
        task_name='Hapaseg_prepare_coverage_mcmc',
        script_generator=make_hapaseg_prepare_coverage_mcmc_script,
        script_args={
            'coverage_csv': covcollect_t_results_dict["coverage"],
            'allelic_clusters_object': hapaseg_allelic_dp_results_dict["cluster_and_phase_assignments"],
            'SNPs_pickle': hapaseg_allelic_dp_results_dict['all_SNPs'],
            'segmentations_pickle': hapaseg_allelic_dp_results_dict['segmentation_breakpoints'],
            'repl_pickle': ref_config["repl_file"],
            'ref_fasta': ref_config["ref_fasta"],
            'bin_width': bin_width,
            'wgs': wgs,
            'faire_pickle': "" if (not is_ffpe and not is_cfdna) else (ref_config["cfdna_wes_faire_file"] if (is_cfdna and not wgs) else ref_config["faire_file"]),
            'gc_pickle': "",
            'normal_coverage_csv': covcollect_n_results_dict["coverage"],
            'extra_covariates': extra_covariate_beds if extra_covariate_beds is not None else None
        }
    )

    
    # shim task to get number of allelic segments
    # coverage MCMC will be scattered over each allelic segment
    def get_N_seg_groups(idx_file):
        indices = np.r_[np.genfromtxt(idx_file, delimiter='\n', dtype=int)]
        return list(indices)

    cov_mcmc_shards_list = get_N_seg_groups(hapaseg_prepare_coverage_mcmc_results_dict["allelic_seg_idxs"])

    # Run coverage MCMC in parallel for each allelic segment
    cov_mcmc_results_dict = run_parallel_tasks(
        out_path=out_path,
        task_name='Hapaseg_coverage_mcmc',
        script_generator=make_hapaseg_coverage_mcmc_by_Aseg_script,
        script_args={
            'preprocess_data': hapaseg_prepare_coverage_mcmc_results_dict["preprocess_data"],
            'allelic_seg_indices': hapaseg_prepare_coverage_mcmc_results_dict["allelic_seg_groups"],
            'allelic_seg_scatter_idx': cov_mcmc_shards_list,
            'num_draws': num_cov_seg_samples,
            'bin_width': bin_width
        },
        max_jobs=job_resources(4, 5, max_cpus, max_memory)
    )

    # collect coverage MCMC
    hapaseg_collect_coverage_mcmc_results_dict = run_single_task(
        out_path=out_path,
        task_name='Hapaseg_collect_coverage_mcmc',
        script_generator=make_hapaseg_collect_coverage_mcmc_script,
        script_args={
            'cov_mcmc_files': cov_mcmc_results_dict["cov_segmentation_data"],
            'cov_df_pickle': hapaseg_prepare_coverage_mcmc_results_dict["cov_df_pickle"],
            'seg_indices_pickle': hapaseg_prepare_coverage_mcmc_results_dict["allelic_seg_groups"],
            'bin_width': bin_width,
            'cytoband_file': ref_config["cytoband_file"]
        }
    )
    
    

    #get the adp draw number from the preprocess data object
    def _get_ADP_draw_num(preprocess_data_obj):
        return int(np.load(preprocess_data_obj)["adp_cluster"])
    
    adp_draw_num = _get_ADP_draw_num(hapaseg_prepare_coverage_mcmc_results_dict["preprocess_data"])
    

    # only run cov DP if using exomes. genomes should have enough bins in each segment
    if not wgs and run_cdp:
        # coverage DP
        # Run coverage DP in parallel for each sample index
        hapaseg_coverage_dp_results_dict = run_parallel_tasks(
            out_path=out_path,
            task_name='Hapaseg_coverage_dp',
            script_generator=make_hapaseg_coverage_dp_script,
            script_args={
                'f_cov_df': hapaseg_prepare_coverage_mcmc_results_dict["cov_df_pickle"],
                'cov_mcmc_data': hapaseg_collect_coverage_mcmc_results_dict["cov_collected_data"],
                'num_segmentation_samples': num_cov_seg_samples,
                'num_dp_samples': 5,
                'bin_width': bin_width,
                'sample_idx': list(range(num_cov_seg_samples))
            },
            max_jobs=job_resources(8, 10, max_cpus, max_memory)
        )

        # generate acdp dataframe 
        hapaseg_gen_acdp_results_dict = run_single_task(
            out_path=out_path,
            task_name="Hapaseg_acdp_generate_df",
            script_generator=make_hapaseg_acdp_generate_df_script,
            script_args={
                "SNPs_pickle": hapaseg_allelic_dp_results_dict['all_SNPs'],  # each scatter result is the same
                "allelic_clusters_object": hapaseg_allelic_dp_results_dict["cluster_and_phase_assignments"],
                "ref_file_path": ref_config["ref_fasta"],
                "allelic_draw_index": adp_draw_num,
                "cdp_filepaths": [hapaseg_coverage_dp_results_dict["cov_dp_object"]],
                "bin_width": bin_width
            }
        )
        # run acdp
        hapaseg_run_acdp_results_dict = run_single_task(
            out_path=out_path,
            task_name="Hapaseg_run_acdp",
            script_generator=make_hapaseg_run_acdp_script,
            script_args={
                "cov_seg_data": hapaseg_collect_coverage_mcmc_results_dict["cov_collected_data"],
                "acdp_df": hapaseg_gen_acdp_results_dict["acdp_df_pickle"],
                "num_samples": num_cov_seg_samples,
                "cytoband_file": ref_config["cytoband_file"],
                "opt_cdp_idx": int(open(hapaseg_gen_acdp_results_dict["opt_cdp_idx"], 'r').read()),
                "lnp_data_pickle": hapaseg_gen_acdp_results_dict["lnp_data_pickle"],
                "wgs": wgs
            }
        )
    else:
        # otherwise generate acdp dataframe directly from cov_mcmc results
        # Generate ACDP dataframe directly from cov_mcmc results
        hapaseg_gen_acdp_results_dict = run_single_task(
            out_path=out_path,
            task_name="Hapaseg_acdp_generate_df",
            script_generator=make_hapaseg_acdp_generate_df_script,
            script_args={
                "SNPs_pickle": hapaseg_allelic_dp_results_dict['all_SNPs'],
                "allelic_clusters_object": hapaseg_allelic_dp_results_dict["cluster_and_phase_assignments"],
                "ref_file_path": ref_config["ref_fasta"],
                "allelic_draw_index": adp_draw_num,
                "cov_df_pickle": hapaseg_prepare_coverage_mcmc_results_dict["cov_df_pickle"],
                "cov_seg_data": hapaseg_collect_coverage_mcmc_results_dict["cov_collected_data"],
                "bin_width": bin_width,
                "wgs": wgs
            }
        )

        # run acdp
        hapaseg_run_acdp_results_dict = run_single_task(
            out_path=out_path,
            task_name="Hapaseg_run_acdp",
            script_generator=make_hapaseg_run_acdp_script,
            script_args={
                "cov_seg_data": hapaseg_collect_coverage_mcmc_results_dict["cov_collected_data"],
                "acdp_df": hapaseg_gen_acdp_results_dict["acdp_df_pickle"],
                "num_samples": num_cov_seg_samples,
                "cytoband_file": ref_config["cytoband_file"],
                "opt_cdp_idx": int(open(hapaseg_gen_acdp_results_dict["opt_cdp_idx"], 'r').read()),
                "lnp_data_pickle": hapaseg_gen_acdp_results_dict["lnp_data_pickle"],
                "wgs": wgs,
                "use_single_draw": True  # for now only use single best draw for wgs
            }
        )

    # create final summary plot
    hapaseg_summary_plot_results_dict = run_single_task(
        out_path=out_path,
        task_name="Hapaseg_summary_plot",
        script_generator=make_hapaseg_summary_plot_script,
        script_args={
            "snps_pickle": hapaseg_allelic_dp_results_dict['all_SNPs'],
            "adp_results": hapaseg_allelic_dp_results_dict["cluster_and_phase_assignments"],
            "segmentations_pickle": hapaseg_allelic_dp_results_dict['segmentation_breakpoints'],
            "acdp_model": hapaseg_run_acdp_results_dict["acdp_model_pickle"],
            "ref_fasta": ref_config["ref_fasta"],
            "cytoband_file": ref_config["cytoband_file"],
            "hapaseg_segfile": hapaseg_run_acdp_results_dict["hapaseg_segfile"]
        }
    )
    print(f"Finished running HapASeg on {sample_name}")
    output_dict = {'covcollect_n_results_dict' : covcollect_n_results_dict,
            'covcollect_t_results_dict' : covcollect_t_results_dict,
            'mutect_results_dict' : mutect_results_dict,
            'hetpull_results_dict' : hetpull_results_dict,
            'convert_hp_results_dict' : convert_hp_results_dict,
            'eagle_outputs': eagle_outputs,
            'comb_vcf_outputs': comb_vcfs_results_dict,
            'hapaseg_load_snps_results_dict': hapaseg_load_snps_results_dict,
            'hapaseg_burnin_results_dict': hapaseg_burnin_results_dict,
            'hapaseg_concat_results_dict': hapaseg_concat_results_dict,
            'hapaseg_amcmc_results_dict': hapaseg_amcmc_results_dict,
            'hapaseg_concat_arm_results_dict': hapaseg_concat_arm_results_dict,
            'hapaseg_allelic_dp_results_dict': hapaseg_allelic_dp_results_dict,
            'hapaseg_prepare_coverage_mcmc_results_dict': hapaseg_prepare_coverage_mcmc_results_dict,
            'hapaseg_cov_mcmc_results_dict': cov_mcmc_results_dict,
            'hapaseg_collect_coverage_mcmc_results_dict': hapaseg_collect_coverage_mcmc_results_dict,
            'hapaseg_coverage_dp_results_dict': hapaseg_coverage_dp_results_dict if run_cdp and not wgs else None,
            'hapaseg_gen_acdp_results_dict': hapaseg_gen_acdp_results_dict,
            'hapaseg_run_acdp_results_dict': hapaseg_run_acdp_results_dict,
            'hapaseg_summary_plot_results_dict': hapaseg_summary_plot_results_dict
            }

    return output_dict

def save_final_results(output_dict, results_path):
    """
    Save final HapaSeg results by creating symlinks to important output files.
    
    Args:
        output_dict: Dictionary containing paths to output files
        results_dir: Directory where symlinks should be created
    """
    # Create a function to handle creating symlinks
    def create_symlink(source_file, target_path):
        print(source_file, target_path)
        if source_file is None:
            return False
        source_path = Path(source_file)
        if not source_path.exists():
            return False
        link_path = target_path.joinpath(source_path.name)
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(source_path)
        return True

    # Softlink the hapaseg summary plot to the results directory
    if 'hapaseg_summary_plot_results_dict' in output_dict and 'hapaseg_summary_plot' in output_dict['hapaseg_summary_plot_results_dict']:
        create_symlink(output_dict['hapaseg_summary_plot_results_dict']['hapaseg_summary_plot'], results_path)

    # Softlink relevant ACDP outputs
    acdp_outputs = [
        "acdp_model_pickle", 
        "acdp_clusters_plot", 
        "acdp_tuples_plot",
        "hapaseg_segfile", 
        "absolute_segfile", 
        "hapaseg_skip_acdp_segfile",
        "acdp_optimal_fit_params"
    ]
    
    # Add optional outputs that are only generated when not using single draw
    if "acdp_sd_plot" in output_dict:
        acdp_outputs.extend(["acdp_sd_plot", "acdp_sd_best_plot"])
    
    # Create symlinks for all available ACDP outputs
    for output_key in acdp_outputs:
        create_symlink(output_dict['hapaseg_run_acdp_results_dict'][output_key], results_path)

@click.command()
@click.argument('out_dir', type=click.Path())
@click.argument('sample_name', type=str)
@click.option('--tumor-bam',
              required=True,
              type=click.Path(exists=True),
              help='Tumor sample BAM file')
@click.option('--tumor-bai',
              required=True,
              type=click.Path(exists=True),
              help='Tumor sample BAM index file (BAI)')
@click.option('--normal-bam',
              required=True,
              type=click.Path(exists=True),
              help='Normal sample BAM file')
@click.option('--normal-bai', 
              help='Normal sample BAM index file (BAI)')
@click.option('--ref-genome-build',
              required=True,
              help='Reference genome build string. Must be hg19 or hg38')
@click.option('--ref-root-path',
              required=True,
              type=click.Path(exists=True),
              help='Reference root path')
@click.option('--genotyping-method',
              default="mixture_model",
              help='Genotyping method to use. Default: mixture_model is recommended. See het_pulldown_from_callstats_TOOL for more options and method details.')
@click.option('--single-ended',
              is_flag=True,
              default=False,
              help='Flag for designating if sequencing was done with single ended reads. Coverage collection differs depending on whether BAM is paired end')
@click.option('--WES-target-list', 'WES_target_list',
              required=False,
              type=click.Path(exists=True),
              help='WES exome target set BED file to use as bin intervals. Designate running on a WES sample by passing this file. Required for running on WES samples and should correspond to the sequencing bait set used for the sample (e.g. TWIST)')
@click.option('--WGS-bin-size', 'WGS_bin_size',
              default=2000,
              type=int,
              help='coverage bin size for running on WGS samples. Default: 2000 base pairs')
@click.option('--betahyp',
              default=4,
              type=int,
              help='Hyperparameter for smoothing initial allelic segmentation. Only applicable for whole exomes. Default: 4')
@click.option('--num-cov-seg-samples',
              default=5,
              type=int,
              help='Number of draws from the MCMC chain to use during coverage segmentation. Default: 5 is recommended.')
@click.option('--run-cdp',
              is_flag=True,
              default=False,
              help='Option to run coverage DP on WES data.')
@click.option('--is-ffpe',
              is_flag=True,
              default=False,
              help='Flag to designate tumor sample was FFPE preserved. Will cause HapASeg to use FAIRE as covariate for coverage model')
@click.option('--is-cfdna',
              is_flag=True,
              default=False,
              help='Flag to designate tumor sample is cfDNA. Will cause HapASeg to use cfDNA FAIRE as covariate for coverage model')
@click.option('--extra-covariate-beds', 
              required=False,
              help='(Optional) List of filepaths to BED files containing additional covariates to use for coverage model (comma-separated)')
@click.option('--ref-file-override-dict',
              required=False,
              type=click.Path(exists=True),
              help='(Optional) Path to a YAML file of reference file variable name keys and file path values to override the default ref file config')
@click.option('--max-cpus',
              default=8,
              type=int,
              help='Maximum number of CPUs to use. Default: 8')
@click.option('--max-memory',
              default=16,
              type=int,
              help='Maximum memory to use in GBs. Default: 16GB')
def run_hapaseg_local(out_dir,
                      sample_name,
                      **kwargs):
    """
    Command line interface for running HapaSeg locally.
    """
    # Process any special inputs that need conversion
    if kwargs.get('extra_covariate_beds'):
        kwargs['extra_covariate_beds'] = kwargs['extra_covariate_beds'].split(',')
    
    if kwargs.get('ref_file_override_dict'):
        try:
            kwargs['ref_file_override_dict'] = yaml.safe_load(kwargs['ref_file_override_dict'])
        except yaml.YAMLError:
            click.echo("Error: ref-file-override-dict must be valid YAML", err=True)
            return
    else:
        kwargs['ref_file_override_dict'] = {}
    
    # Call the actual function
    kwargs['out_dir'] = out_dir
    kwargs['sample_name'] = sample_name
    output_dict = hapaseg_local_main(**kwargs)
    
    # save output dict results to a results directory
    out_path = Path(out_dir)
    results_path = out_path.joinpath(f"{sample_name}/Hapaseg_final_results")
    results_path.mkdir(exist_ok=True, parents=True)
    # Save final results to the results directory
    save_final_results(output_dict, results_path)
    
    click.echo(f"Final results saved to {results_path}")

if __name__ == '__main__':
    run_hapaseg_local()

