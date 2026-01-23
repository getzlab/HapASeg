import subprocess
from pathlib import Path
import time
from typing import List, Optional
import os
import tempfile
import glob

from typing import Dict, List, Callable, Tuple, Any, Optional, Union

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

class ProcessFailedError(Exception):
    """Exception raised when a subprocess exits with non-zero status."""
    def __init__(self, process_id, return_code, stderr_file=None):
        self.process_id = process_id
        self.return_code = return_code
        self.stderr_file = stderr_file
        message = f"Process {process_id} failed with return code {return_code}"
        if stderr_file:
            message += f". Check logs at {stderr_file}"
        super().__init__(message)


def wait_for_complete(
    processes: List[subprocess.Popen],
    check_interval: int = 5,
    timeout: Optional[int] = None,
    stderr_files: Optional[List[str]] = None
) -> None:
    """
    Wait for all processes to complete successfully.
    
    Args:
        processes: List of subprocess.Popen objects to monitor
        check_interval: Time in seconds between status checks
        timeout: Maximum time in seconds to wait (None for no timeout)
        stderr_files: Optional list of stderr log files corresponding to processes
    
    Raises:
        ProcessFailedError: If any process exits with non-zero status
        TimeoutError: If the timeout is reached before all processes complete
    """
    if stderr_files and len(stderr_files) != len(processes):
        raise ValueError("If provided, stderr_files must have same length as processes")
    
    start_time = time.time()
    pending_processes = list(range(len(processes)))
    
    while pending_processes:
        # Check each pending process
        for idx in pending_processes[:]:  # Create a copy to safely modify during iteration
            process = processes[idx]
            return_code = process.poll()
            
            # Process has completed
            if return_code is not None:
                pending_processes.remove(idx)
                
                # Check if it failed
                if return_code != 0:
                    stderr_file = stderr_files[idx] if stderr_files else None
                    raise ProcessFailedError(
                        process_id=process.pid,
                        return_code=return_code,
                        stderr_file=stderr_file
                    )
        
        # All processes completed successfully
        if not pending_processes:
            return
        
        # Check timeout
        if timeout and (time.time() - start_time > timeout):
            still_running = [processes[idx].pid for idx in pending_processes]
            raise TimeoutError(f"Timed out after {timeout} seconds. Processes still running: {still_running}")
        
        # Wait before checking again
        time.sleep(check_interval)


def run_gnu_parallel_tasks_backend(script_dict, log_dir: Path, max_jobs=8):
    """
    Run a collection of bash scripts in parallel using GNU parallel with concurrency control.
    
    Args:
        script_dict: Dictionary with script names as keys and bash script strings as values.
        log_dir: Directory to store log files and parallel output results.
        max_jobs: Maximum number of concurrent jobs (default: 8)
    
    Returns:
        True if all tasks completed successfully, False otherwise.
    """
    # Ensure the log directory exists.
    log_dir.mkdir(exist_ok=True)
    
    # Create directories for GNU parallel output and for storing individual scripts.
    results_dir = log_dir.joinpath("results")
    results_dir.mkdir(exist_ok=True)
    
    scripts_dir = log_dir.joinpath("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Write each script to a file named after its key.
    script_files = []
    for script_name, script_content in script_dict.items():
        # Use the script name as the file name.
        # (Assumes script_name is a valid filename; adjust as needed.)
        script_file = scripts_dir.joinpath(script_name)
        with open(script_file, 'w') as f:
            f.write(script_content)
        script_files.append(str(script_file.name))
    
    # Build the GNU parallel command.
    # For each script file, GNU parallel will run: bash <script_file>
    # The --results option will create a subdirectory for each file (labeled by the file name)
    # where the stdout, stderr, and exit codes are stored.
    cmd = [
        "parallel",
        "--joblog", str(log_dir / "parallel.log"),
        "-j", str(max_jobs),
        "--tag",
        "--results", str(results_dir),
        "bash", str(scripts_dir / "{}"),
        ":::"
    ] + script_files

    try:
        # Run GNU parallel and wait for it to complete.
        process = subprocess.run(
            cmd, 
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # If GNU parallel returned a non-zero exit code, print the error and exit.
        if process.returncode != 0:
            print(f"Error running GNU parallel: {process.stderr}")
            return False
        
        # Read the joblog to see if any individual job failed.
        with open(log_dir / "parallel.log", 'r') as joblog:
            # Skip header line.
            next(joblog, None)
            for line in joblog:
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    exit_code = int(parts[6])
                    if exit_code != 0:
                        task_id = parts[0]
                        print(f"Task {task_id} failed with exit code {exit_code}")
                        return False
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def popen_command_pipe_outputs(script: str, out_dir: Path):
    # Open files for writing logs
    with open(out_dir.joinpath("stdout.log"), "w") as stdout_file, open(out_dir.joinpath("stderr.log"), "w") as stderr_file:
        # Run command and redirect output to files
        process = subprocess.Popen(
            script, 
            stdout=stdout_file,
            stderr=stderr_file,
            shell=True
        )
    return process

def run_command_pipe_outputs(script: str, out_dir: Path):
    # Open files for writing logs
    with open(out_dir.joinpath("stdout.log"), "w") as stdout_file, open(out_dir.joinpath("stderr.log"), "w") as stderr_file:
        # Run command and redirect output to files
        results = subprocess.run(
            script, 
            stdout=stdout_file,
            stderr=stderr_file,
            shell=True
        )
    return results


def run_parallel_tasks(
    out_path: Path,
    task_name: str,
    script_generator: Callable,
    script_args: Dict[str, Any],
    max_jobs: int = 8
):
    """
    Generic function to handle parallel execution of chunked tasks with simplified argument handling.
    
    Parameters:
    -----------
    out_path : Path
        Base output directory
    task_name : str
        Name of the task (used for directory naming)
    script_generator : Callable
        Function that generates scripts for each chunk. Must return (script, output_dict) tuple.
    script_args : Dict[str, Any]
        Arguments to pass to the script generator function. Values can be:
        - Single values (used for all chunks)
        - Lists of values (one per chunk, all lists must have the same length)
    max_jobs : int, optional
        Maximum number of parallel jobs, defaults to 8
        
    Returns:
    --------
    Dict[str, Path]
        Combined dictionary of all output file paths
    """
    # Create output directory
    parent_out_path = out_path.joinpath(f'{task_name}_Results')
    parent_out_path.mkdir(exist_ok=True, parents=True)
    
    # Determine how many chunks we have by finding any list arguments
    list_args = {k: v for k, v in script_args.items() if isinstance(v, list)}
    
    if not list_args:
        raise ValueError("No list arguments found in script_args. At least one argument must be a list for chunked execution.")
    
    # Verify all lists have the same length
    list_lengths = [len(v) for v in list_args.values()]
    if len(set(list_lengths)) > 1:
        raise ValueError(f"All list arguments must have the same length. Found lengths: {list_lengths}")
    
    num_chunks = list_lengths[0]
    
    # Dictionary to store scripts, results, and execution status
    all_scripts = {}
    all_results = {}
    chunks_to_run = []
    
    # Generate scripts for each chunk
    for idx in range(num_chunks):
        # Create chunk-specific arguments
        chunk_args = {}
        for key, value in script_args.items():
            if isinstance(value, list):
                chunk_args[key] = value[idx]
            else:
                chunk_args[key] = value
        
        # Set output directory
        chunk_args['out_path'] = parent_out_path.joinpath(f'{task_name}_chunk_{idx}')
        
        # Generate script for this chunk
        chunk_name = f"{task_name}_chunk_{idx}"
        chunk_script, chunk_results = script_generator(**chunk_args)
        
        all_scripts[chunk_name] = chunk_script
        
        # Store results with unique keys for each chunk to avoid overwriting
        for key, value in chunk_results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(value)
        
        # Check if output files for this chunk already exist
        chunk_outputs_exist = results_exist(chunk_results)
        
        # Add to execution list if outputs don't exist
        if not chunk_outputs_exist:
            chunks_to_run.append(chunk_name)
    
    # Run only the chunks that need to be executed
    if chunks_to_run:
        print(f"Running {len(chunks_to_run)} of {num_chunks} {task_name} tasks in parallel (skipping {num_chunks - len(chunks_to_run)} completed tasks)...")
        scripts_to_run = {name: all_scripts[name] for name in chunks_to_run}
        run_gnu_parallel_tasks_backend(
            scripts_to_run, 
            parent_out_path.joinpath(f'{task_name}_runlog'),
            max_jobs=max_jobs
        )
    else:
        print(f"All {num_chunks} {task_name} chunks already completed. Skipping execution.")
    
    # Return all result file paths
    return all_results


def run_single_task(
    out_path: Path,
    task_name: str,
    script_generator: Callable,
    script_args: Dict[str, Any],
    check_for_existing: bool = True
):
    """
    Run a single task with option to check if outputs already exist.
    
    Parameters:
    -----------
    out_path : Path
        Base output directory
    task_name : str
        Name of the task (used for directory naming)
    script_generator : Callable
        Function that generates the script. Must return (script, output_dict) tuple.
    script_args : Dict[str, Any]
        Arguments to pass to the script generator function
    check_for_existing : bool, optional
        Whether to check if output files already exist before running, defaults to True
        
    Returns:
    --------
    Dict[str, Path]
        Dictionary of output file paths
    """
    # Create output directory
    task_out_path = out_path.joinpath(f'{task_name}_Results')
    task_out_path.mkdir(exist_ok=True, parents=True)
    
    # Set output directory
    script_args['out_path'] = task_out_path
    
    # Generate script
    script, results_dict = script_generator(**script_args)
    
    # Check if output files already exist
    if check_for_existing:
        all_exist = results_exist(results_dict)
    else:
        all_exist = False
    
    # Run the task if outputs don't exist
    if not all_exist:
        run_command_pipe_outputs(script, task_out_path)
    else:
        print(f"{task_name} results already exist. Skipping execution.")
    
    # Return result file paths
    return results_dict

    
# covcollect
def make_covcollect_script(
    out_dir: Path,
    bam,
    bai,
    intervals,
    single_ended=False,
    ):

    result_path = out_dir.joinpath('coverage.bed')
    script = f"covcollect -b {bam} -i {intervals} -o {result_path}" + (" -S" if single_ended else "")
    return script, { "coverage" : result_path}

def run_covcollect(out_dir: Path,
                   bam,
                   bai,
                   intervals,
                   single_ended=False,
                   ):

    script = make_covcollect_script(out_dir, bam, bai, intervals, single_ended=False)

    try:
        process = run_command_pipe_outputs(script, out_dir)
    except Exception as e:
        print(f"Run covcollect failed with error {e}\n attempted script:  {script}")
    
    return process, { "coverage" : result_path , 'stderr':out_dir.joinpath('stderr.log')}

# Mutect1
def make_mutect_script(
    out_dir: Path,
    n_bam = None, 
    n_bai = None,
    t_bam = None,
    t_bai = None,
    
    # references
    refFasta = None,
    refFastaIdx = None,
    refFastaDict = None,
    intervals = "",

    pairName = 'het_coverage',
    caseName = 'tumor',
    ctrlName = 'normal', 
    downsample = 99999,
    max_mismatch_baseq_sum = 1000,
    fracContam = 0,
    force_calling = True,
    exclude_chimeric = True,
    ):
        cs_path = out_dir.joinpath('het_coverage.MuTect1.call_stats.txt')
        vcf_path = out_dir.joinpath('het_coverage.Mutect1.vcf')

        script = f"""/usr/local/java/jdk1.7.0_80/bin/java -Xmx3g -jar /app/mutect.jar \
            --analysis_type MuTect \
            --tumor_sample_name {caseName} \
            -I:tumor {t_bam} \
            --normal_sample_name {ctrlName} \
            -I:normal {n_bam} \
            --reference_sequence {refFasta} \
            --fraction_contamination {fracContam} \
            --out {cs_path} \
            --downsample_to_coverage {downsample} \
            --max_read_mismatch_quality_score_sum {max_mismatch_baseq_sum} \
            --vcf {vcf_path}"""

        if force_calling:
            script += " --force_output"
        if exclude_chimeric:
            script += " --exclude_chimeric_reads"
        if intervals != "":
            ln_intv_path = out_dir.joinpath(Path(intervals).stem + '.picard')
            script = f"ln -s {intervals} {ln_intv_path} && " + script
            script += f" -L {ln_intv_path}"

        output_dicts = {
        "mutect1_cs" : cs_path,
        "mutect1_vcf": vcf_path
        }

        return script, output_dicts

# Het pulldown
def make_het_pulldown_script(out_path: Path,
                             callstats_file = None,
                             ref_fasta = None,
                             ref_fasta_idx = None,
                             ref_fasta_dict = None,
                             method = None,
                             common_snp_list = "",
                             beta_dens_cutoff = 0.7,
                             log_pod_threshold = 2.5,
                             max_frac_mapq0 = 0.05,
                             max_frac_prefiltered = 0.1,
                             tumor_only = False,
                             pod_min_depth = 10,
                             min_tumor_depth = 1):
    out_path.mkdir(exist_ok=True, parents=True)
    out_stem = out_path.joinpath('het_coverage')
    script =  f"hetpull.py -g -c {callstats_file} -r {ref_fasta} -o {out_stem} " + \
            f"--dens {beta_dens_cutoff} --max_frac_mapq0 {max_frac_mapq0} -m {method} " + \
            f"--log_pod_threshold {log_pod_threshold} --pod_min_depth {pod_min_depth} " + \
            ("--use_tonly_genotyper " if tumor_only else "") + \
            (f" --min_tumor_depth {min_tumor_depth}" if min_tumor_depth != "" else "") + \
            (f" -s {common_snp_list}" if common_snp_list != "" else "")
    outputs = {
        "tumor_hets" : str(out_stem) + ".tumor.tsv",
        "normal_hets" : str(out_stem) + ".normal.tsv",
        "normal_genotype" : str(out_stem) + ".genotype.tsv",
        "all_sites" : str(out_stem) + ".all_sites.tsv",
    }
    return script, outputs

def convert_hetpull_2_vcf_script(out_path:Path,
                                 genotype_file=None,
                                 ref_fasta=None,
                                 ref_fasta_idx=None,
                                 ref_fasta_dict=None,
                                 sample_name='test'):
    out_path.mkdir(exist_ok=True, parents=True)
    script = f"""
    set -x
    bcftools convert --tsv2vcf {genotype_file} -c CHROM,POS,AA -s {sample_name} \
        -f {ref_fasta} -Ou -o {out_path.joinpath('all_chrs.bcf')} && bcftools index {out_path.joinpath('all_chrs.bcf')}
    for chr in $(bcftools view -h {out_path.joinpath('all_chrs.bcf')} | ssed -nR '/^##contig/s/.*ID=(.*),.*/\\1/p' | head -n24); do
        bcftools view -Ou -r ${{chr}} -o {out_path}/${{chr}}.chrsplit.bcf {out_path.joinpath('all_chrs.bcf')} && bcftools index {out_path}/${{chr}}.chrsplit.bcf
    done
    """
    # TODO: figure out output naming
    outputs = {
        "bcfs" : [out_path.joinpath(f'chr{c}.chrsplit.bcf') for c in range(1,23)] + [out_path.joinpath('chrX.chrsplit.bcf')],
        "bcf_idxs" : [out_path.joinpath(f'chr{c}.chrsplit.bcf.csi') for c in range(1,23)] + [out_path.joinpath('chrX.chrsplit.bcf.csi')]
        }
    return script, outputs

# Eagle
def make_eagle_script(out_path:Path,
                      genetic_map_file = None,
                      vcf_in = None,
                      vcf_idx_in = None,
                      vcf_ref = None,
                      vcf_ref_idx = None,
                      output_file_prefix = None,
                      num_threads = 1
                      ):
    #TODO: Figure out output naming
    out_path.mkdir(exist_ok=True, parents=True)
    out_file_prefix = out_path.joinpath(output_file_prefix)
    script = f"""eagle --geneticMapFile {genetic_map_file} --outPrefix {out_file_prefix} --numThreads {num_threads} --vcfRef {vcf_ref} --vcfTarget {vcf_in} --vcfOutFormat v"""
    output_patterns = { "phased_vcf" : f"{out_file_prefix}.vcf" }
    return script, output_patterns
    
def combine_eagle_vcfs_script(out_path:Path,
                              vcf_paths):
    out_path.mkdir(exist_ok=True, parents=True)
    outfile_path = out_path.joinpath('combined.vcf')
    # Create a temporary file with one VCF path per line
    tmp_file_path = out_path.joinpath('vcf_paths.txt')

    # Write each VCF path to the temporary file
    with open(tmp_file_path, 'w') as f:
        for vcf_path in vcf_paths:
            f.write(f"{vcf_path}\n")
    
    # Set the vcf_array variable to point to the temporary file
    vcf_array = tmp_file_path

    script = f"""bcftools concat -O u $(cat {vcf_array} | tr '\n' ' ') | bcftools sort -O v -o {outfile_path}"""
    
    outputs= {'combined_vcf': outfile_path}
    return script, outputs