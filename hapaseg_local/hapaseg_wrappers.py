import subprocess
from pathlib import Path
import time
from typing import List, Optional
import os
import tempfile
import glob
import pandas as pd

def make_hapaseg_load_snps_script(
    out_path: Path,
    phased_VCF,
    tumor_allele_counts,
    cytoband_file,
    ref_file_path,
    normal_allele_counts="",
):
    """
    Generate a script to run hapaseg load_snps command.
    
    Parameters:
    -----------
    out_path : Path
        Directory for output files
    phased_VCF : str
        Path to phased VCF file
    tumor_allele_counts : str
        Path to tumor allele counts file
    cytoband_file : str
        Path to cytoband file
    ref_file_path : str
        Path to reference file
    normal_allele_counts : str, optional
        Path to normal allele counts file, defaults to ""
    
    Returns:
    --------
    tuple
        (script, output_files_dict)
    """
    # Define output files
    out_path.mkdir(exist_ok=True, parents=True)
    allele_counts_path = out_path.joinpath('allele_counts.pickle')
    scatter_chunks_path = out_path.joinpath('scatter_chunks.tsv')
    
    # Construct the base script
    script = f"export CAPY_REF_FA={ref_file_path}\n"
    script += f"hapaseg --output_dir {out_path} load_snps --phased_VCF {phased_VCF} --allele_counts_T {tumor_allele_counts} --cytoband_file {cytoband_file}"
    
    # Add normal allele counts if provided
    if normal_allele_counts != "":
        script += f" --allele_counts_N {normal_allele_counts}"
    
    outputs = {"allele_counts": allele_counts_path,
               "scatter_chunks": scatter_chunks_path
              }
    # Return the script and output file paths
    return script, outputs

def make_hapaseg_burnin_script(
    out_path: Path,
    allele_counts,
    start,
    end,
    betahyp=-1
):
    """
    Generate a script to run hapaseg amcmc burnin for a specific chunk.
    
    Parameters:
    -----------
    out_path : Path
        Directory for output files
    allele_counts : str
        Path to allele counts pickle file
    start : int or str
        Start position for this chunk
    end : int or str
        End position for this chunk
    betahyp : int or float, optional
        Beta hyperparameter, defaults to -1
    
    Returns:
    --------
    tuple
        (script, output_files_dict)
    """
    # Define output file path
    out_path.mkdir(exist_ok=True, parents=True)
    burnin_mcmc_path = out_path.joinpath("amcmc_results.pickle")
    
    # Construct the script
    script = f"hapaseg --output_dir {out_path} amcmc --snp_dataframe {allele_counts} --start {start} --end {end} --stop_after_burnin --betahyp {betahyp}"
    
    # Return the script and output file paths
    outputs = {"burnin_MCMC": burnin_mcmc_path}

    return script, outputs

def make_hapaseg_concat_script(
    out_path: Path,
    chunks,
    scatter_intervals
):
    """
    Generate a script to run hapaseg concat to combine chunks and infer reference bias.
    
    Parameters:
    -----------
    out_path : Path
        Directory for output files
    chunks : list
        List of paths to burnin MCMC pickle files
    scatter_intervals : str
        Path to scatter intervals file
    
    Returns:
    --------
    tuple
        (script, output_files_dict)
    """
    # Define output file paths
    out_path.mkdir(exist_ok=True, parents=True)

    arms_list = list(pd.read_csv(scatter_intervals, sep="\t").arm.unique())
    
    # Create a temporary file with the list of chunks
    chunks_file = out_path.joinpath("chunks_list.txt")
    with open(chunks_file, 'w') as f:
        for chunk in chunks:
            f.write(f"{chunk}\n")
    
    # Construct the script
    script = f"""CHUNKS_STR=$(cat {chunks_file} | tr '\\n' ' ') && hapaseg --output_dir {out_path} concat --chunks $CHUNKS_STR --scatter_intervals {scatter_intervals}"""
    
    # Define output patterns
    arm_paths = [out_path.joinpath(f"AMCMC-arm{arm}.pickle") for arm in arms_list]
    ref_bias_path = out_path.joinpath("ref_bias.txt")
    
    # Return the script and output file paths
    outputs = {
        "arms": arm_paths,
        "ref_bias": ref_bias_path
    }
    
    return script, outputs

def make_hapaseg_amcmc_script(
    out_path: Path,
    amcmc_object,
    ref_bias,
    n_iter=20000,
    betahyp=-1
):
    """
    Generate a script to run hapaseg amcmc on arm-level data.
    
    Parameters:
    -----------
    out_path : Path
        Directory for output files
    amcmc_object : str
        Path to the AMCMC object pickle file
    ref_bias : str
        Path to the reference bias file
    n_iter : int, optional
        Number of iterations (default: 20000)
    betahyp : float, optional
        Beta hyperparameter value (default: -1)
    
    Returns:
    --------
    tuple
        (script, output_files_dict)
    """
    # Define output file paths
    out_path.mkdir(exist_ok=True, parents=True)
    figures_dir = out_path.joinpath("figures")
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Construct the script
    script = f"""hapaseg --output_dir {out_path} amcmc --amcmc_object {amcmc_object} --ref_bias {ref_bias} --n_iter {n_iter} --betahyp {betahyp}"""
    
    # Define output patterns
    arm_level_mcmc_path = out_path.joinpath("amcmc_results.pickle")
    segmentation_plot_path = out_path.joinpath("figures/MLE_segmentation.png")
    
    # Return the script and output file paths
    outputs = {
        "arm_level_MCMC": arm_level_mcmc_path,
        "segmentation_plot": segmentation_plot_path
    }
    
    return script, outputs

def make_hapaseg_allelic_dp_script(
    out_path: Path,
    seg_dataframe,
    ref_fasta,
    cytoband_file,
    wgs=False
):
    """
    Generate a script to run hapaseg dp for allelic dynamic programming.
    
    Parameters:
    -----------
    out_path : Path
        Directory for output files
    seg_dataframe : str
        Path to the segmentation dataframe pickle file
    ref_fasta : str
        Path to the reference FASTA file
    cytoband_file : str
        Path to the cytoband file
    wgs : bool, optional
        Whether the data is whole genome sequencing (default: False)
    
    Returns:
    --------
    tuple
        (script, output_files_dict)
    """
    # Define output file paths
    out_path.mkdir(exist_ok=True, parents=True)
    figures_dir = out_path.joinpath("figures")
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Construct the script
    script = f"export CAPY_REF_FA={ref_fasta}\n"
    script += f"hapaseg --output_dir {out_path} dp --seg_dataframe {seg_dataframe} --ref_fasta {ref_fasta} --cytoband_file {cytoband_file}"
    
    # Add WGS flag if needed
    if wgs:
        script += " --wgs"
    
    # Define output patterns
    cluster_assignments_path = out_path.joinpath("allelic_DP_SNP_clusts_and_phase_assignments.npz")
    all_snps_path = out_path.joinpath("all_SNPs.pickle")
    segmentation_breakpoints_path = out_path.joinpath("segmentations.pickle")
    likelihood_trace_plot_path = out_path.joinpath("figures/likelihood_trace.png")
    snp_plot_path = out_path.joinpath("figures/SNPs.png")
    seg_plot_path = out_path.joinpath("figures/segs_only.png")
    
    # Return the script and output file paths
    outputs = {
        "cluster_and_phase_assignments": cluster_assignments_path,
        "all_SNPs": all_snps_path,
        "segmentation_breakpoints": segmentation_breakpoints_path,
        "likelihood_trace_plot": likelihood_trace_plot_path,
        "SNP_plot": snp_plot_path,
        "seg_plot": seg_plot_path
    }
    
    return script, outputs

def make_hapaseg_prepare_coverage_mcmc_script(
    out_path: Path,
    coverage_csv: str,
    allelic_clusters_object: str,
    SNPs_pickle: str,
    segmentations_pickle: str,
    repl_pickle: str,
    ref_fasta: str,
    bin_width: int = 1,
    wgs: bool = True,
    faire_pickle: str = "",
    gc_pickle: str = "",
    normal_coverage_csv: str = "",
    allelic_sample: str = "",
    extra_covariates: Optional[List[str]] = None
):
    """
    Generate a script to prepare data for HapASeg coverage MCMC.
    
    Parameters:
    -----------
    out_path : Path
        Directory for output files
    coverage_csv : str
        Path to the coverage CSV file
    allelic_clusters_object : str
        Path to the allelic clusters object
    SNPs_pickle : str
        Path to the SNPs pickle file
    segmentations_pickle : str
        Path to the segmentations pickle file
    repl_pickle : str
        Path to the replication timing pickle file
    ref_fasta : str
        Path to the reference FASTA file
    bin_width : int, optional
        Bin width for coverage analysis (default: 1)
    wgs : bool, optional
        Whether the data is whole genome sequencing (default: True)
    faire_pickle : str, optional
        Path to the FAIRE pickle file (default: "")
    gc_pickle : str, optional
        Path to the GC content pickle file (default: "")
    normal_coverage_csv : str, optional
        Path to the normal coverage CSV file (default: "")
    allelic_sample : str, optional
        Allelic sample identifier (default: "")
    extra_covariates : str, optional
        Path to extra covariates bed files (default: "")
    
    Returns:
    --------
    tuple
        (script, output_files_dict)
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)

    # Handle extra_covariates if it's a list of paths
    if extra_covariates and isinstance(extra_covariates, list):
        # Create a temporary file with one path per line
        extra_covariates_file = out_path.joinpath("extra_covariates_paths.txt")
        with open(extra_covariates_file, 'w') as f:
            for path in extra_covariates:
                f.write(f"{path}\n")
    
    # Construct the base script
    script = f"""hapaseg --output_dir {out_path} coverage_mcmc_preprocess --coverage_csv {coverage_csv} \\
    --ref_fasta {ref_fasta} \\
    --allelic_clusters_object {allelic_clusters_object} \\
    --SNPs_pickle {SNPs_pickle} \\
    --segmentations_pickle {segmentations_pickle} \\
    --repl_pickle {repl_pickle} \\
    --bin_width {bin_width}"""
    
    # Add optional arguments
    if wgs:
        script += " --wgs"
    if faire_pickle:
        script += f" --faire_pickle {faire_pickle}"
    if gc_pickle:
        script += f" --gc_pickle {gc_pickle}"
    if normal_coverage_csv:
        script += f" --normal_coverage_csv {normal_coverage_csv}"
    if allelic_sample:
        script += f" --allelic_sample {allelic_sample}"
    if extra_covariates:
        script += f" --extra_covariates_bed_paths {extra_covariates_file}"
    
    # Define output file paths
    preprocess_data_path = out_path.joinpath("preprocess_data.npz")
    cov_df_pickle_path = out_path.joinpath("cov_df.pickle")
    allelic_seg_groups_path = out_path.joinpath("allelic_seg_groups.pickle")
    allelic_seg_idxs_path = out_path.joinpath("allelic_seg_idxs.txt")
    
    # Return the script and output file paths
    outputs = {
        "preprocess_data": preprocess_data_path,
        "cov_df_pickle": cov_df_pickle_path,
        "allelic_seg_groups": allelic_seg_groups_path,
        "allelic_seg_idxs": allelic_seg_idxs_path
    }
    
    return script, outputs

def make_hapaseg_coverage_mcmc_by_Aseg_script(
    out_path,
    preprocess_data,
    allelic_seg_indices,
    allelic_seg_scatter_idx,
    num_draws=50,
    bin_width=None,
    range=None
):
    """
    Generate a script for running hapaseg coverage_mcmc_shard on a specific allelic segment.
    
    Args:
        out_path: Path to output directory
        preprocess_data: Path to preprocess data npz file
        allelic_seg_indices: Path to allelic segment indices pickle file
        allelic_seg_scatter_idx: Allelic segment index to operate on
        num_draws: Number of MCMC draws to perform
        bin_width: Bin width for coverage MCMC
        range: Optional range parameter
        
    Returns:
        tuple: (script, outputs) where script is the command to run and outputs is a dict of output files
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Construct the base script
    script = f"""hapaseg --output_dir {out_path} coverage_mcmc_shard \\
    --preprocess_data {preprocess_data} \\
    --num_draws {num_draws} \\
    --allelic_seg_indices {allelic_seg_indices} \\
    --allelic_seg_idx {allelic_seg_scatter_idx} \\
    --bin_width {bin_width}"""
    
    # Add optional arguments
    if range:
        script += f" --range {range}"
    
    # Define output file patterns
    # Using wildcards to match the output patterns from the Task class
    cov_segmentation_model_path = out_path.joinpath(f"cov_mcmc_model_allelic_seg_{allelic_seg_scatter_idx}.pickle")
    cov_segmentation_data_path = out_path.joinpath(f"cov_mcmc_data_allelic_seg_{allelic_seg_scatter_idx}.npz")
    cov_seg_figure_path = out_path.joinpath(f"cov_mcmc_allelic_seg_{allelic_seg_scatter_idx}_visual.png")
    
    # Return the script and output file paths
    outputs = {
        "cov_segmentation_model": cov_segmentation_model_path,
        "cov_segmentation_data": cov_segmentation_data_path,
        "cov_seg_figure": cov_seg_figure_path
    }
    
    return script, outputs

def make_hapaseg_collect_coverage_mcmc_script(
    out_path,
    cov_mcmc_files,
    cov_df_pickle,
    seg_indices_pickle,
    cytoband_file,
    bin_width=1
):
    """
    Generate a script for running hapaseg collect_cov_mcmc to collect coverage MCMC results.
    
    Args:
        out_path: Path to output directory
        cov_mcmc_files: List of paths to coverage MCMC data files
        cov_df_pickle: Path to coverage dataframe pickle file
        seg_indices_pickle: Path to segment indices pickle file
        cytoband_file: Path to cytoband file (required)
        bin_width: Bin width for coverage MCMC (default: 1)
        
    Returns:
        tuple: (script, outputs) where script is the command to run and outputs is a dict of output files
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)
    figures_dir = out_path.joinpath("figures")
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a temporary file with the list of cov_mcmc_files
    cov_mcmc_files_list = out_path.joinpath("cov_mcmc_files.txt")
    with open(cov_mcmc_files_list, "w") as f:
        for file_path in cov_mcmc_files:
            f.write(f"{file_path}\n")
    
    # Construct the script
    script = f"""hapaseg --output_dir {out_path} collect_cov_mcmc \\
    --seg_indices_pickle {seg_indices_pickle} \\
    --cov_mcmc_files {cov_mcmc_files_list} \\
    --cov_df_pickle {cov_df_pickle} \\
    --bin_width {bin_width} \\
    --cytoband_file {cytoband_file}"""
    
    # Define output file paths
    cov_collected_data_path = out_path.joinpath("cov_mcmc_collected_data.npz")
    seg_plot_path = out_path.joinpath("figures/segs.png")
    
    # Return the script and output file paths
    outputs = {
        "cov_collected_data": cov_collected_data_path,
        "seg_plot": seg_plot_path
    }
    
    return script, outputs

def make_hapaseg_coverage_dp_script(
    out_path,
    f_cov_df,
    cov_mcmc_data,
    num_segmentation_samples=10,
    num_dp_samples=10,
    bin_width=None,
    sample_idx=None
):
    """
    Generate a script for running hapaseg coverage_dp to perform coverage Dirichlet Process.
    
    Args:
        out_path: Path to output directory
        f_cov_df: Path to coverage dataframe pickle file
        cov_mcmc_data: Path to coverage MCMC data
        num_segmentation_samples: Number of segmentation samples (default: 10)
        num_dp_samples: Number of DP samples (default: 10)
        bin_width: Bin width for coverage (optional)
        sample_idx: Sample index as a string (optional)
        
    Returns:
        tuple: (script, outputs) where script is the command to run and outputs is a dict of output files
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Construct the base script
    script = f"""hapaseg --output_dir {out_path} coverage_dp \\
    --f_cov_df {f_cov_df} \\
    --cov_mcmc_data {cov_mcmc_data} \\
    --num_segmentation_samples {num_segmentation_samples} \\
    --num_dp_samples {num_dp_samples}"""
    
    # Add optional arguments
    if sample_idx is not None:
        script += f" --sample_idx {sample_idx}"
    
    if bin_width is not None:
        script += f" --bin_width {bin_width}"
    
    # Define output file paths
    cov_dp_object_path = out_path.joinpath("Cov_DP_model.pickle")
    cov_dp_figure_path = out_path.joinpath("cov_dp_visual_draw.png")
    
    # Return the script and output file paths
    outputs = {
        "cov_dp_object": cov_dp_object_path,
        "cov_dp_figure": cov_dp_figure_path
    }
    
    return script, outputs


def make_hapaseg_acdp_generate_df_script(
    out_path,
    SNPs_pickle,
    allelic_clusters_object,
    ref_file_path,
    allelic_draw_index=-1,
    cdp_object=None,
    cov_df_pickle=None,
    cov_seg_data=None,
    cdp_filepaths=None,
    bin_width=None,
    wgs=False
):
    """
    Generate a script for running hapaseg generate_acdp_df to create ACDP dataframe.
    
    Args:
        out_path: Path to output directory
        SNPs_pickle: Path to SNPs dataframe pickle file
        allelic_clusters_object: Path to allelic clusters object
        ref_file_path: Path to reference FASTA file
        allelic_draw_index: Index of allelic draw to use (default: -1)
        cdp_object: Path to CDP object (optional)
        cov_df_pickle: Path to coverage dataframe pickle (optional)
        cov_seg_data: Path to coverage segmentation data (optional)
        cdp_filepaths: List of paths to CDP files (optional)
        bin_width: Bin width for coverage (optional)
        wgs: Whether data is whole genome sequencing (default: False)
        
    Returns:
        tuple: (script, outputs) where script is the command to run and outputs is a dict of output files
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Start building the script
    script = f"""export CAPY_REF_FA={ref_file_path}
    hapaseg --output_dir {out_path} generate_acdp_df \\
    --snp_dataframe {SNPs_pickle} \\
    --allelic_clusters_object {allelic_clusters_object} \\
    --allelic_draw_index {allelic_draw_index}"""
    
    # Add optional arguments
    if bin_width is not None:
        script += f" --bin_width {bin_width}"
    if cdp_object is not None:
        script += f" --cdp_object {cdp_object}"
    if cdp_filepaths is not None:
        # Create a temporary file with the list of CDP files
        cdp_files_list = out_path.joinpath("cdp_filepaths.txt")
        with open(cdp_files_list, "w") as f:
            for file_path in cdp_filepaths:
                f.write(f"{file_path}\n")
        script += f" --cdp_filepaths {cdp_files_list}"
    if cov_df_pickle is not None:
        script += f" --cov_df_pickle {cov_df_pickle}"
    if cov_seg_data is not None:
        script += f" --cov_seg_data {cov_seg_data}"
    if wgs:
        script += " --wgs"
    
    # Define output file paths
    acdp_df_pickle_path = out_path.joinpath("acdp_df.pickle")
    opt_cdp_idx_path = out_path.joinpath("opt_cdp_draw.txt")
    lnp_data_pickle_path = out_path.joinpath("lnp_data.pickle")
    
    # Return the script and output file paths
    outputs = {
        "acdp_df_pickle": acdp_df_pickle_path,
        "opt_cdp_idx": opt_cdp_idx_path,
        "lnp_data_pickle": lnp_data_pickle_path
    }
    
    return script, outputs


def make_hapaseg_run_acdp_script(
    out_path,
    cov_seg_data,
    acdp_df,
    num_samples,
    cytoband_file,
    opt_cdp_idx,
    lnp_data_pickle,
    wgs=False,
    use_single_draw=False
):
    """Generate a script to run the allelic coverage DP algorithm.
    
    Args:
        out_path: Path to output directory
        cov_seg_data: Path to coverage segmentation data
        acdp_df: Path to ACDP dataframe
        num_samples: Number of samples to use
        cytoband_file: Path to cytoband file
        opt_cdp_idx: Path to optimal CDP index file
        lnp_data_pickle: Path to log-likelihood data pickle
        wgs: Whether data is whole genome sequencing (default: False)
        use_single_draw: Whether to use a single draw (default: False)
        
    Returns:
        tuple: (script, outputs) where script is the command to run and outputs is a dict of output files
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Start building the script
    script = f"""hapaseg --output_dir {out_path} allelic_coverage_dp \\
    --cov_seg_data {cov_seg_data} \\
    --acdp_df_path {acdp_df} \\
    --lnp_data_pickle {lnp_data_pickle} \\
    --num_samples {num_samples} \\
    --cytoband_file {cytoband_file} \\
    --opt_cdp_idx {opt_cdp_idx}"""
    
    # Add optional arguments
    if wgs:
        script += " --wgs"
    if use_single_draw:
        script += " --use_single_draw"
    
    # Define output file paths
    acdp_model_pickle_path = out_path.joinpath("acdp_model.pickle")
    acdp_clusters_plot_path = out_path.joinpath("acdp_clusters_plot.png")
    acdp_tuples_plot_path = out_path.joinpath("acdp_tuples_plot.png")
   
    hapaseg_segfile_path = out_path.joinpath("hapaseg_segfile.txt")
    absolute_segfile_path = out_path.joinpath("absolute_segfile.txt")
    hapaseg_skip_acdp_segfile_path = out_path.joinpath("hapaseg_skip_acdp_segfile.txt")
    acdp_optimal_fit_params_path = out_path.joinpath("acdp_optimal_fit_params.txt")
    
    # these are only genrated if not using single draw
    acdp_sd_plot_path = out_path.joinpath("acdp_draws.png") 
    acdp_sd_best_plot_path = out_path.joinpath("acdp_best_cdp_draw.png")
    acdp_dict ={"acdp_sd_plot": acdp_sd_plot_path, "acdp_sd_best_plot": acdp_sd_best_plot_path} if not use_single_draw else {}
    # Return the script and output file paths
    outputs = {**{
        "acdp_model_pickle": acdp_model_pickle_path,
        "acdp_clusters_plot": acdp_clusters_plot_path,
        "acdp_tuples_plot": acdp_tuples_plot_path,
        "hapaseg_segfile": hapaseg_segfile_path,
        "absolute_segfile": absolute_segfile_path,
        "hapaseg_skip_acdp_segfile": hapaseg_skip_acdp_segfile_path,
        "acdp_optimal_fit_params": acdp_optimal_fit_params_path
    }, **acdp_dict}
    
    return script, outputs

def make_hapaseg_summary_plot_script(
    out_path: Path,
    snps_pickle: Path,
    adp_results: Path,
    segmentations_pickle: Path,
    acdp_model: Path,
    ref_fasta: Path,
    cytoband_file: Path,
    hapaseg_segfile: Path
):
    """
    Generate a script to create a summary plot for HapaSeg results.
    
    Args:
        out_path: Output directory path
        snps_pickle: Path to SNPs pickle file
        adp_results: Path to ADP results file
        segmentations_pickle: Path to segmentations pickle file
        acdp_model: Path to ACDP model file
        ref_fasta: Path to reference FASTA file
        cytoband_file: Path to cytoband file
        hapaseg_segfile: Path to HapaSeg segmentation file
        
    Returns:
        tuple: (script, outputs) where script is the command to run and outputs is a dict of output files
    """
    # Create output directory
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Start building the script
    script = f"""hapaseg --output_dir {out_path} summary_plot \\
    --snps_pickle {snps_pickle} \\
    --adp_results {adp_results} \\
    --segmentations_pickle {segmentations_pickle} \\
    --acdp_model {acdp_model} \\
    --ref_fasta {ref_fasta} \\
    --cytoband_file {cytoband_file} \\
    --hapaseg_segfile {hapaseg_segfile} \\
    --outdir {out_path}"""
    
    # Define output file paths
    hapaseg_summary_plot_path = out_path.joinpath("hapaseg_summary_plot.png")
    
    # Return the script and output file paths
    outputs = {
        "hapaseg_summary_plot": hapaseg_summary_plot_path
    }
    
    return script, outputs
