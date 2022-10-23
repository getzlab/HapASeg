import wolf

hatchet = wolf.ImportTask(
  task_path = "/home/cmesser/HapASeg/benchmarking/hatchet_wolf_task/",
  task_name = "hatchet"
)

def HATCHET_Depths(reference_genome_path,  
                        reference_genome_version,
                     normal_bam,
                     tumor_bam,
                     normal_vcf_path = None,  # Path to VCF file
                     sample_names = None,
                     chromosomes = [],
                     chr_notation = False,  # True if contigs have "chr" prefix
                     phase_snps = True
                    ):
    
#     # task to generate [CHR POS] vcf file from standard SNP file
#     vcf_format_shim_task = Task(name = "format_vcf",
#                                 inputs = {"normal_vcf_path": normal_vcf_path},
#                                 script = "cut -f 1,2  > vcf_formatted.txt",
#                                 outputs = {"vcf_snps": "vcf_formatted.txt"})
    
    genotype_snps_task = hatchet.HATCHET_genotype_snps(inputs = {"reference_genome_path": reference_genome_path,
                                                         "normal_bam": normal_bam,
                                                         "vcf_snps": normal_vcf_path,
                                                         "chromosomes": chromosomes})
    count_alleles_task = hatchet.HATCHET_count_alleles(inputs = {"reference_genome_path": reference_genome_path,
                                                         "normal_bam": normal_bam,
                                                         "tumor_bam": tumor_bam,
                                                         "vcf_snps": genotype_snps_task["germline_hets"],
                                                         "sample_names": sample_names,
                                                         "chromosomes": chromosomes})
    count_reads_task = hatchet.HATCHET_count_reads(inputs = {"reference_genome_version": reference_genome_version,
                                                             "normal_bam": normal_bam,
                                                             "tumor_bam": tumor_bam,
                                                             "normal_baf": count_alleles_task["normal_snp_depths"],
                                                             "sample_names": sample_names,
                                                             "chromosomes": chromosomes})
    output_dict = {'genotype_snps_task': genotype_snps_task,
                   'count_alleles_task': count_alleles_task,
                   'count_reads_task': count_reads_task}
    if phase_snps:
        download_phasing_panel_task = hatchet.HATCHET_download_phasing_panel()
        phase_snps_task = hatchet.HATCHET_phase_snps(inputs = {"reference_genome_path": reference_genome_path,
                                                       "snps_vcf": genotype_snps_task["germline_hets"],
                                                       "reference_genome_version": reference_genome_version,
                                                       "chr_notation": chr_notation
        })   
        output_dict['phase_snps_task'] = phase_snps_task
        
    return output_dict
        

def HATCHET_Main(tumor_snp_depths=None,
                 count_reads_dir=None,
                 total_counts_file=None,
                 reference_genome_version="hg38",
                 phase_snps = True,
                 phased_vcf = None):
    
    combine_counts_task = hatchet.HATCHET_combine_counts(inputs = {"tumor_baf": tumor_snp_depths,
                                                           "count_reads_dir": count_reads_dir,
                                                           "total_counts_file": total_counts_file,
                                                           "reference_genome_version": reference_genome_version,
                                                           "phased_vcf_file": None if not phase_snps else phased_vcf})
    
    cluster_bins_task = hatchet.HATCHET_cluster_bins(inputs = {"bb_file": combine_counts_task["binned_counts_file"]})
    
    compute_cn_task = hatchet.HATCHET_compute_cn(inputs = {"cluster_bins_prefix": "./bbc/bulk."})  #todo change?
    
    return {'combine_counts_task': combine_counts_task,
            'cluster_bins_task': cluster_bins_task,
            'compute_cn_task': compute_cn_task}
