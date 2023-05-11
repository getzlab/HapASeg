import wolf

class subsample_bam(wolf.Task):
    inputs = {"bam":None}
    script = """
    samtools view -b -s 0.01 ${bam} > subsampled_bam.bam
    samtools index subsampled_bam.bam
    """
    
    output_patterns = {"subsampled_bam": "subsampled_bam.bam",
                       "subsampled_bai": "subsampled_bam.bam.bai"
                      }

    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"
    resources = {"cpus-per-task": 4, "mem": "10G"}


class HATCHET_genotype_snps(wolf.Task):
    inputs = {"ref_fasta": None,  # reference genome FASTA file
              "normal_bam": None,
              "normal_bai": None,
              "vcf_file": "",  # Path to vcf file containing snps to consider
              "vcf_idx":"", # index file for vcf
              "min_coverage": 8,  # Use 8 for WGS with >30x and 20 for WES with ~100x
              "max_coverage": 300,  # Use 300 for WGS with >30x and 1000 for WES with ~100x; twice the values of expected average coverage to avoid aligning artifacts
              "read_quality": 0,
              "base_quality": 11,
              "num_processes": 2,
              "chromosomes": ""
    }
    
    def script(self):
        script = """
        hatchet genotype-snps --normal ${normal_bam} --reference ${ref_fasta}\
        --mincov ${min_coverage} --maxcov ${max_coverage} --readquality ${read_quality}\
        --basequality ${base_quality} --processes ${num_processes}"""

        if self.conf["inputs"]["vcf_file"] is not None:
            script += " --snps ${vcf_file}"
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
            
        return script
    
    output_patterns = {
    "germline_snp_vcfs": "*.vcf.gz",
    }
    
    resources = {"cpus-per-task": 2, "mem": "6G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"

    
class HATCHET_count_alleles(wolf.Task):
    inputs = {"ref_fasta": None,  # reference FASTA file
              "normal_bam": None,
              "normal_bai": None,
              "tumor_bam": None,  # can be list of multiple bams
              "tumor_bai": None,
              "second_tumor_bam":"",
              "second_tumor_bai":"",
              "vcf_file": None,  # One or more files listing heterozygous SNP positions, format [CHR POS]
              "vcf_file_idx": None, # vcf indices expected for each
              "sample_names": "normal_id tumor_id",  # should be [normal_id, tumor_id]
              "min_coverage": 8,  # Use 8 for WGS with >30x and 20 for WES with ~100x
              "max_coverage": 300,  # Use 300 for WGS with >30x and Use 1000 for WES with ~100x
              "read_quality": 0,
              "base_quality": 11,
              "snp_quality": 11,
              "snp_gamma": 0.05,
              "num_processes": 1,
              "chromosomes": ""
    }
    
    def script(self):
        script = """
        ln -s ${normal_bam} ./normal.bam; ln -s ${normal_bai} ./normal.bam.bai
        ln -s ${tumor_bam} ./tumor.bam; ln -s ${tumor_bai} ./tumor.bam.bai
        hatchet count-alleles --normal ./normal.bam --reference ${ref_fasta}\
        --snps ${vcf_file} --mincov ${min_coverage} --maxcov ${max_coverage}\
        --readquality ${read_quality} --basequality ${base_quality} --snpquality ${snp_quality}\
        --gamma ${snp_gamma} --processes ${num_processes} --outputnormal ./normal.1bed\
        --outputtumor ./tumor.1bed --outputsnps ./"""
        if self.conf["inputs"]["second_tumor_bam"]:
            script = """ln -s ${second_tumor_bam} ./second_tumor.bam; ln -s ${second_tumor_bai} ./second_tumor.bam.bai""" + script
            script += " --tumors ./tumor.bam ./second_tumor.bam"
        else:
            script += " --tumors ./tumor.bam"
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
        if self.conf["inputs"]["sample_names"]:
            script += " --samples ${sample_names}"
            
        return script
    
    output_patterns = {
                    "normal_snp_depths": "./normal.1bed",
                    "tumor_snp_depths": "./tumor.1bed",
                    }
    
    resources = { "mem": "2G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"

    
class HATCHET_count_reads(wolf.Task):
    inputs = {"reference_genome_version": "hg38",  # hg38 or hg19
              "normal_bam": None,
              "normal_bai":None,
              "tumor_bam": None,  # can be list of multiple bams
              "tumor_bai":None,
              "second_tumor_bam": "", #optional additional sample
              "second_tumor_bai": "",
              "tumor_baf": None,  # Generated from count-alleles
              "sample_names": "normal_id tumor_id",  # should be [normal_id, tumor_id]
              "read_quality": 11,
              "num_processes": 6,
              "chromosomes": ""
    }
    
    def script(self):
        script = """
        ln -s ${normal_bam} ./normal.bam; ln -s ${normal_bai} ./normal.bam.bai
        ln -s ${tumor_bam} ./tumor.bam; ln -s ${tumor_bai} ./tumor.bam.bai
        hatchet count-reads --normal ./normal.bam\
         --refversion ${reference_genome_version}\
        --baffile ${tumor_baf} --outdir .\
        --readquality ${read_quality} --processes ${num_processes}"""

        if self.conf["inputs"]["second_tumor_bam"]:
            script = """ln -s ${second_tumor_bam} ./second_tumor.bam; ln -s ${second_tumor_bai} ./second_tumor.bam.bai""" + script
            script += " --tumor ./tumor.bam ./second_tumor.bam"
        else:
            script += " --tumor ./tumor.bam"
        
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
        
        if self.conf["inputs"]["sample_names"]:
            script += " --samples " + self.conf["inputs"]["sample_names"]
            
        return script
    
    output_patterns = {
    "snp_thresholds": "*.thresholds.gz",
    "coverage_totals": "*.total.gz",
    "total_counts_file": "total.tsv",
    "sample_names_file": "samples.txt"
    }
    
    resources = {"cpus-per-task": 10, "mem": "90G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"
    
class HATCHET_download_phasing_panel(wolf.Task):
    inputs = {"reference_panel_dir": "ref_panel"}  # can we just save to this local directory?
    
    def script(self):
        return "hatchet download-panel --refpaneldir ${reference_panel_dir}"  # how to save this to correct (accessible) directory?
   
    # output patterns?
    output_patterns = {"ref_panel_dir" : "ref_panel/*"}
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"
    
class HATCHET_phase_snps(wolf.Task):
    inputs = {"reference_panel_dir": None,
              "ref_fasta": None,
              "ref_fasta_dict": None,
              "snps_vcf": None,  # taken from genotype-snps
              "reference_genome_version": "hg38",
              "chr_notation": False,  # True if bam contains "chr1" instead of "1"
              "num_processes": 24
              # outdir
    }
    
    def script(self):
        script = """
        hatchet phase-snps --snps ../inputs/chr*.vcf.gz --refpaneldir ../inputs/\
        --refgenome ${ref_fasta} --refversion ${reference_genome_version}\
        --outdir .  --processes ${num_processes}"""

        if self.conf["inputs"]["chr_notation"]:
            script += " --chrnotation"
            
        return script
    
    output_patterns = {
    "phased_vcf": "phased.vcf.gz",
    }
    
    resources = {"cpus-per-task": 32, "mem": "180G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"


class reformat_hatchet_depth_outputs(wolf.Task):  
    """
    combine counts requires a total reads (from all chromosomes) file along with a
    single sample name per line file. so we gather those results here
    """
    inputs = {"all_depths_paths": None,
              "sample_path": None,
              "reverse_samples": False
             }
    
    script = """
python -c "import pandas as pd    
total_reads = {}
for path in open('${all_depths_paths}', 'r').read().split():
    df = pd.read_csv(path, sep='\\t', header=None, names=['sample', 'reads'])
    for s, r in df.set_index('sample').to_dict()['reads'].items():
        if s in total_reads:
            total_reads[s] += r
        else:
            total_reads[s] = r
if len(total_reads) > 2:
    raise ValueError('not intended for use with more than 2 samples')
if ${reverse_samples}:
    # read in samples in swapped order
    samples = list(reversed(open('${sample_path}').read().split()))
    total_df = pd.DataFrame([[samples[0], total_reads[samples[1]]], [samples[1], total_reads[samples[0]]]])
else:
    samples = list(open('${sample_path}').read().split())
    total_df = pd.DataFrame([[samples[0], total_reads[samples[0]]], [samples[1], total_reads[samples[1]]]])

total_df.to_csv('./total.tsv', sep='\\t', header=None, index=False)
with open('samples.txt', 'w') as f:
    for samp in samples:
        f.write('{}\\n'.format(samp))
"
"""

    output_patterns = {"totals": "total.tsv",
                       "sample": "sample.txt"
                      }
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"

class HATCHET_preprocess(wolf.Task):
    inputs = {"totals_paths":None,
              "thresholds_paths": None,
              "tumor_baf_path": None,
              "sample_name":None,
              "dummy_normal":False #pass to ignore normal and use first two tumor samples
             }

    def script(self):
        script = """
        preprocess_raw_data.py --sample_name ${sample_name} --outdir . \
        hatchet --totals_file_paths ${totals_paths} --thresholds_file_paths ${thresholds_paths} \
        --tumor_baf_path ${tumor_baf_path}"""

        if self.conf["inputs"]["dummy_normal"]:
            script += " --dummy_normal"
        return script

    output_patterns = {"interval_counts": "*_interval_counts.for_simulation_input.txt",
                       "pos_counts" : "*_position_counts.for_simulation_input.txt",
                       "snp_counts": "*_snp_counts.for_simulation_input.txt",
                       "read_combined" : "*_read_combined_df.txt"}

    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

    resources = {"mem": "8G"}

class HATCHET_combine_counts(wolf.Task):
    inputs = {"tumor_baf": None,  # typically produced by count-alleles
              "count_reads_dir": "",  # typically populated by count-reads,
              "count_reads_array": "", # txt file with gathered count reads threholds outputs, should be used with additional totals_files
              "total_counts_file": None,  # typically populated by count-reads
              "reference_genome_version": "hg38",
              "min_snp_covering_reads": 5000,  # per bin
              "min_total_reads": 5000,  # per bin
              "phased_vcf_file": None,
              "max_phase_block_size": "",
              "max_snps_phased_block": "",
              "snp_alpha": "",
              "num_processes": 8,
              "samples_file":"", 
              "additional_totals_files":""
     }
    
    def script(self):
        script = """hatchet combine-counts --baffile ${tumor_baf} --totalcounts ${total_counts_file}\
        --refversion ${reference_genome_version} --msr ${min_snp_covering_reads} --mtr ${min_total_reads}\
        --processes ${num_processes} -o ./combined_counts.tsv"""
        
        # if our totals files were generated seperately, copy counts dir then softlink them to the new dir
        if self.conf["inputs"]["additional_totals_files"] != "":
            # samples file is also saved seperately in sim but must be in the same dir as counts
            sample_copy_script = ""
            if self.conf["inputs"]["samples_file"] != "":
                sample_copy_script = "cp ${samples_file} ./tmp_reads_dir && "
            
            thresh_transfer_script = ""
            if self.conf["inputs"]["count_reads_dir"] != "":
                thresh_transfer_script = " cp ${count_reads_dir}/* ./tmp_reads_dir "
            elif self.conf["inputs"]["count_reads_array"] != "":
                thresh_transfer_script = " for f in $(cat ${count_reads_array}); do ln -s $f ./tmp_reads_dir/; done "
            else:
                raise ValueError("need to pass count_reads_dir or count_reads_array")

            script = "set -eo pipefail && mkdir ./tmp_reads_dir &&" + thresh_transfer_script + "&& for f in $(cat ${additional_totals_files}); do ln -s $f ./tmp_reads_dir/; done && " + sample_copy_script + script
            script += " --array ./tmp_reads_dir"

        else:
            script += " --array ${count_reads_dir}"
        if self.conf["inputs"]["phased_vcf_file"] is not None:
            script += " --phase ${phased_vcf_file}"
        if self.conf["inputs"]["max_phase_block_size"] != "":
            script += " --blocksize ${max_phase_block_size}"
        if self.conf["inputs"]["max_snps_phased_block"] != "":
            script += " --max_spb ${max_snps_phased_block}"
        if self.conf["inputs"]["snp_alpha"] != "":
            script += " --alpha ${snp_alpha}"
        
        return script
    
    output_patterns = {
    "binned_counts_file": "./combined_counts.tsv",
    }
    
    resources = {"cpus-per-task": 8, "mem": "14G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"
    
class HATCHET_cluster_bins(wolf.Task):
    inputs = {"bb_file": None,  # typically produced by combine-counts
              "max_diploid_baf_shift": "", 
              "seed": "",
              "min_k": "",
              "max_k": "",
              "exact_k": "",
              "transition_matrix": "",  # fixed, diag or full
              "tau": "",
              "cov_form": "", # spherical, diag, full, tied
              "decoding_alg": "" # map or viterbi
             }
    
    def script(self):
        script = """
        hatchet cluster-bins ${bb_file} --outbins ./clustered_bins.bbc\
        --outsegment ./clustered_bins.seg"""
        
        if self.conf["inputs"]["exact_k"] != "":
            script += " --exactK ${exact_k}"
        if self.conf["inputs"]["max_diploid_baf_shift"] != "":
            script += " --diploidbaf ${max_diploid_baf_shift}"
        if self.conf["inputs"]["seed"] != "":
            script += " --seed ${seed}"
        if self.conf["inputs"]["min_k"] != "":
            script += " --minK ${min_k}"
        if self.conf["inputs"]["max_k"] != "":
            script += " --maxK ${max_k}"
        if self.conf["inputs"]["transition_matrix"] != "":
            script += " --transmat ${transition_matrix}"
        if self.conf["inputs"]["tau"] != "":
            script += " --tau ${tau}"
        if self.conf["inputs"]["cov_form"] != "":
            script += " --covar ${cov_form}"
        if self.conf["inputs"]["decoding_alg"] != "":
            script += " --decoding ${decoding_alg}"
        
        return script
    
    output_patterns = {
    "clustered_bins": "clustered_bins.bbc",
    "clustered_segments": "clustered_bins.seg",
    }
    
    resources = {"cpus-per-task": 4, "mem": "15G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"
    
class HATCHET_compute_cn(wolf.Task):
    inputs = {"cluster_bins_bbc": None,
              "cluster_bins_seg": None
             }
    
    script = """hatchet compute-cn -j 4 -i ${cluster_bins_bbc: 0:-4}"""
    
    output_patterns = {
    "best_bins_file": "best.bbc.ucn",
    "best_segments_file": "best.seg.ucn",
    }
    
    resources = {"cpus-per-task": 8, "mem": "20G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"
    
    
class HATCHET_plot_cn(wolf.Task):
    inputs = {"opt_bbc": None}

    script = """hatchet plot-cn ${opt_bbc}"""

    output_patterns = {
        "output_plots" : "*.pdf"
    }
    resources = {"cpus-per-task": 2, "mem": "7G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"


class HATCHET_plot_bins(wolf.Task):
    inputs = {"clustered_bins" : None,
              "clustered_segs" : None
             }

    script = """
    hatchet plot-bins ${clustered_bins} -s ${clustered_segs} -c CRD
    hatchet plot-bins ${clustered_bins} -s ${clustered_segs} -c CBAF
    """

    output_patterns = {
        "RD_plot" : "readdepthratio_clustered.pdf",
        "BAF_plot": "ballelefrequency_clustered.pdf"
    }
    resources = {"cpus-per-task": 2, "mem": "6G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v2"

