import wolf

class HATCHET_genotype_snps(wolf.Task):
    inputs = {"reference_genome_path": None,  # FASTA file
              "normal_bam": None,
              "vcf_snps": None,  # Path to file of SNPs in the format [CHR POS]
              "min_coverage": 8,  # Use 8 for WGS with >30x and 20 for WES with ~100x
              "max_coverage": 300,  # Use 300 for WGS with >30x and 1000 for WES with ~100x; twice the values of expected average coverage to avoid aligning artifacts
              "read_quality": 0,
              "base_quality": 11,
              "num_processes": 6,
              "chromosomes": []
    }
    
    def script(self):
        script = """
        hatchet genotype-snps --normal ${normal_bam} --reference ${reference_genome_path}\
        --mincov ${min_coverage} --maxcov ${max_coverage} --readquality ${read_quality}\
        --basequality ${base_quality} --processes ${num_processes}
        """
        if self.conf["inputs"]["vcf_snps"] is not None:
            script += " --snps ${vcf_snps}"
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
            
        return script
    
    output_patterns = {
    "germline_hets": "germline_het_sites.txt",
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
class HATCHET_count_alleles(wolf.Task):
    inputs = {"reference_genome_path": None,  # FASTA file
              "normal_bam": None,
              "tumor_bam": None,  # can be list of multiple bams
              "vcf_snps": None,  # One or more files listing heterozygous SNP positions, format [CHR POS]
              "sample_names": [],  # should be [normal_id, tumor_id]
              "min_coverage": 8,  # Use 8 for WGS with >30x and 20 for WES with ~100x
              "max_coverage": 300,  # Use 300 for WGS with >30x and Use 1000 for WES with ~100x
              "read_quality": 0,
              "base_quality": 11,
              "snp_quality": 11,
              "snp_gamma": 0.05,
              "num_processes": 22,
              "chromosomes": []
    }
    
    def script(self):
        script = """
        hatchet count-alleles --normal ${normal_bam} --tumors ${tumor_bam} --reference ${reference_genome_path}\
        --snps ${vcf_snps} --mincov ${min_coverage} --maxcov ${max_coverage}\
        --readquality ${read_quality} --basequality ${base_quality} --snpquality ${snp_quality}\
        --gamma {snp_gamma} --processes ${num_processes}
        """
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
        if self.conf["inputs"]["samples"]:
            script += " --samples ${sample_names}"
            
        return script
    
    output_patterns = {
    "normal_snp_depths": "./baf/normal.1bed",
    "tumor_snp_depths": "./baf/tumor.1bed"
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
class HATCHET_count_reads(wolf.Task):
    inputs = {"reference_genome_version": "hg38",  # hg38 or hg19
              "normal_bam": None,
              "tumor_bam": None,  # can be list of multiple bams
              "normal_baf": None,  # Generated from count-alleles
              "sample_names": [],  # should be [normal_id, tumor_id]
              "read_quality": 11,
              "num_processes": 22,
              "chromosomes": []
    }
    
    def script(self):
        script = """
        hatchet count-reads --normal ${normal_bam} --tumors ${tumor_bam} --refversion ${reference_genome_version}\
        --baffile ${normal_baf} \
        --readquality ${read_quality} --processes ${num_processes}
        """
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
        if self.conf["inputs"]["samples"]:
            script += " --samples ${sample_names}"
            
        return script
    
    output_patterns = {
    "snp_thresholds": "./rdr/*.thresholds.gz",
    "coverage_totals": "./rdr/*.total.gz",
    "total_counts_file": "./rdr/total.tsv",
    "sample_names_file": "./rdr/samples.txt"
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
    
class HATCHET_download_phasing_panel(wolf.Task):
    inputs = {"reference_panel_dir": None}  # can we just save to this local directory?
    
    def script(self):
        return "hatchet download-panel --refpaneldir ${reference_panel_dir}"  # how to save this to correct (accessible) directory?
    
    # output patterns?
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
class HATCHET_phase_snps(wolf.Task):
    inputs = {"reference_panel_dir": None,
              "reference_genome_path": None,
              "snps_vcf": None,  # taken from genotype-snps
              "reference_genome_version": "hg38",
              "chr_notation": False,  # True if bam contains "chr1" instead of "1"
              "num_processes": 6
              # outdir?
    }
    
    def script(self):
        script = """
        hatchet phase-snps --snps ${snps_vcf} --refpaneldir ${reference_panel_dir}\
        --refgenome ${reference_genome_path} --refversion ${reference_genome_version}\
        --outdir .  --processes ${num_processes}
        """
        if self.conf["inputs"]["chr_notation"]:
            script += "--chrnotation"
            
        return script
    
    output_patterns = {
    "phased_vcf": "phased.vcf.gz",
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
class HATCHET_combine_counts(wolf.Task):
    inputs = {"tumor_baf": None,  # typically produced by count-alleles
              "count_reads_dir": None,  # typically populated by count-reads
              "total_counts_file": None,  # typically populated by count-reads
              "reference_genome_version": "hg38",
              "min_snp_covering_reads": 5000,  # per bin
              "min_total_reads": 5000,  # per bin
              "phased_vcf_file": None,
              "max_phase_block_size": 25000,
              "max_snps_phased_block": 10,
              "snp_alpha": 0.1,
              "num_processes": 6
    }
    
    def script(self):
        script = """
        hatchet combine-counts --baffile ${tumor_baf} --array ${count_reads_dir} --totalcounts ${total_counts_file}\
        --refversion ${reference_genome_version} --msr ${min_snp_covering_reads} --mtr ${min_total_reads}\
        --blocksize ${max_phase_block_size} --max_spb ${max_snps_phased_block}\
        --alpha ${snp_alpha} --processes ${num_processes}
        """
        if self.conf["inputs"]["chromosomes"]:
            script += " --chromosomes ${chromosomes}"
        if self.conf["inputs"]["phased_vcf_file"] is not None:
            script += "--phase ${phased_vcf_file}"
            
        return script
    
    output_patterns = {
    "binned_counts_file": "./bb/bulk.bb",
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
class HATCHET_cluster_bins(wolf.Task):
    inputs = {"bb_file": None,  # typically produced by combine-counts
                  "max_diploid_baf_shift": None, 
                  "seed": 0,
                  "min_k": 2,
                  "max_k": 30,
                  "exact_k": None,
                  "transition_matrix": "diag",  # fixed, diag or full
                  "tau": 1e-6,
                  "cov_form": "diag", # spherical, diag, full, tied
                  "decoding_alg": "map" # map or viterbi
    }
    
    def script(self):
        script = """
        hatchet cluster-bins ${bb_file} --diploidbaf ${max_diploid_baf_shift} --seed ${seed}\
        --minK ${min_k} --maxK ${max_k} --transmat ${transition_matrix}\
        --tau ${tau} --covar ${cov_form} --decoding ${decoding_alg} 
        """
        
        if self.conf["inputs"]["exact_k"] is not None:
            script += " --exactK ${exact_k}"
        if self.conf["inputs"]["max_diploid_baf_shift"] is not None:
            script += "--diploidbaf ${max_diploid_baf_shift}"
            
        return script
    
    output_patterns = {
    "clustered_bins_file": "./bbc/bulk.bbc",
    "clustered_segments_file": "./bbc/bulk.seg",
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
class HATCHET_compute_cn(wolf.Task):
    inputs = {"cluster_bins_prefix": None,  # typically produced by cluster-bins
              # many other specifications
    }
    
    def script(self):
        script = """
        hatchet compute-cn --input ${cluster_bins_prefix}
        """
        return script
    
    output_patterns = {
    "best_bins_file": "./results/best.bbc.ucn",
    "best_segments_file": "./results/best.seg.ucn",
    }
    
    resources = {"cpus-per-task": 4, "mem": "10G"}
    docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
    
    
class HATCHET_plotting(wolf.Task):
    pass
