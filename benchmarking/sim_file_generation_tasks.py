import wolf

# HapASeg
class Generate_HapASeg_Sim_Data(wolf.Task):
    inputs = {"sim_profile": None,
              "purity": None,
              "sample_label": None,
              "normal_vcf_path": None,
              "hetsite_depth_path": None,
              "covcollect_path":None}
    
    script = """
    generate_sim_data.py --sim_profile ${sim_profile} --purity ${purity} \
    --output_dir . --out_label ${sample_label} hapaseg --normal_vcf_path ${normal_vcf_path}\
    --hetsite_depth_path ${hetsite_depth_path} --covcollect_path ${covcollect_path}\
    """

    output_patterns = {
    "hapaseg_hets": "*_hapaseg_hets.bed",
    "hapaseg_coverage_bed": "*_hapaseg_coverage.bed"
    }
    
    resources = {"cpus-per-task":2, "mem":"6G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# GATK
class Generate_GATK_Sim_Data(wolf.Task):
    inputs = {"sim_profile": None,
              "purity": None,
              "sample_label": None,
              "normal_vcf_path": None,
              "variant_depth_path": None,
              "coverage_tsv_path": None,
              "sim_normal_allelecounts_path": None,
              "raw_gatk_allelecounts_path":None,
              "raw_gatk_coverage_path":None}
    
    script = """
    generate_sim_data.py --sim_profile ${sim_profile} --purity ${purity} \
    --output_dir . --out_label ${sample_label} gatk --normal_vcf_path ${normal_vcf_path}\
    --variant_depth_path ${variant_depth_path} --coverage_tsv_path ${coverage_tsv_path}\
    --sim_normal_allelecounts_path ${sim_normal_allelecounts_path}\
    --raw_gatk_allelecounts_path ${raw_gatk_allelecounts_path}\
    --raw_gatk_coverage_path ${raw_gatk_coverage_path}
    """

    output_patterns = {
    "tumor_coverage_tsv": "*_gatk_sim_tumor_cov.tsv",
    "tumor_allele_counts": "*_gatk_allele.counts.tsv",
    "normal_allele_counts": "*_gatk_sim_normal_allele.counts.tsv",
    "tumor_frag_counts": "*_gatk_sim_tumor.frag.counts.hdf5"
    }
    
    resources = {"cpus-per-task":2, "mem":"6G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# Facets

class Generate_Facets_Sim_Data(wolf.Task):
    inputs = {"sim_profile": None,
              "purity": None,
              "sample_label": None,
              "normal_vcf_path": None,
              "variant_depth_path": "",
              "facets_allelecounts_path":"",
              "filtered_variants_path": "",
              "normal_callstats_path": ""}
    
    def script(self):
        script = """
        generate_sim_data.py --sim_profile ${sim_profile} --purity ${purity} \
        --output_dir . --out_label ${sample_label} facets --normal_vcf_path ${normal_vcf_path}"""
        
        if self.conf["inputs"]["filtered_variants_path"] != "":
            script += " --filtered_variants_path ${filtered_variants_path}"
        if self.conf["inputs"]["normal_callstats_path"] != "":
            script += " --normal_callstats_path ${normal_callstats_path}"
        if self.conf["inputs"]["facets_allelecounts_path"] != "":
            script += " --facets_allelecounts_path ${facets_allelecounts_path}"
        if self.conf["inputs"]["variant_depth_path"] != "":
            script += " --variant_depth_path ${variant_depth_path}"

        return script

    output_patterns = {
    "facets_input_counts": "*_facets_input_counts.csv*"
    }
    
    resources = {"cpus-per-task":2, "mem":"6G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

# ASCAT
class Generate_ASCAT_Sim_Data(wolf.Task):
    inputs = {"sim_profile": None,
              "purity": None,
              "sample_label": None,
              "normal_vcf_path": None,
              "variant_depth_path": None,
              "filtered_variants_path": "",
              "normal_callstats_path" : ""}
   
    def script(self): 
        script = """
        generate_sim_data.py --sim_profile ${sim_profile} --purity ${purity} \
        --output_dir . --out_label ${sample_label} ascat --normal_vcf_path ${normal_vcf_path}\
        --variant_depth_path ${variant_depth_path}"""
        
        if self.conf["inputs"]["filtered_variants_path"] != "":
            script += " --filtered_variants_path ${filtered_variants_path}"
        if self.conf["inputs"]["normal_callstats_path"] != "":
            script += " --normal_callstats_path ${normal_callstats_path}"

        return script

    output_patterns = {
    "ascat_tumor_logR" : "*_ascat_tumor_LogR.txt",
    "ascat_normal_logR" : "*_ascat_normal_LogR.txt",
    "ascat_tumor_BAF" : "*_ascat_tumor_BAF.txt",
    "ascat_normal_BAF" : "*_ascat_normal_BAF.txt"
    }
    
    resources = {"cpus-per-task":2, "mem":"6G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"


# HATCHet
class Generate_HATCHet_Sim_Data(wolf.Task):
    inputs = {"sim_profile": None,
              "purity": None,
              "sample_label": None,
              "normal_vcf_path": None,
              "tumor_baf_path": None,
              "int_counts_file": None,
              "pos_counts_file": None,
              "snp_counts_file": None,
              "read_combined_file": None
             }
    
    script = """
    generate_sim_data.py --sim_profile ${sim_profile} --purity ${purity} \
    --output_dir . --out_label ${sample_label} hatchet --normal_vcf_path ${normal_vcf_path}\
    --tumor_baf_path ${tumor_baf_path} --int_counts_file ${int_counts_file}\
    --pos_counts_file ${pos_counts_file} --snp_counts_file ${snp_counts_file}\
    --read_combined_file ${read_combined_file}
    """

    output_patterns = {
                "samples_file" : "./hatchet_data/samples.txt",
                "transformed_int_counts": "*_interval_counts.transformed.txt",
                "transformed_pos_counts": "*_position_counts.transformed.txt",
                "transformed_snp_counts": "*_snp_counts.transformed.txt",
                "total_count": "./hatchet_data/total.tsv",
                "tumor_snp_counts": "./hatchet_data/tumor.1bed",
                "totals_files": "./hatchet_data/*.total.gz",
                }
    
    resources = {"cpus-per-task":2, "mem":"6G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"
