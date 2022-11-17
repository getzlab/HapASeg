import wolf
from wolf.localization import LocalizeToDisk
import prefect

from tasks import *

def HATCHET_Normal_Depths(ref_fasta=None, # ref genome fasta file
                   ref_fasta_idx=None, # ref genome fai file  
                   ref_fasta_dict=None, # ref genome fai file  
                   reference_genome_version=None,
                   normal_bam = None,
                   normal_bai = None,
                   common_snp_vcf="", # vcf file containing common snp sites
                   common_snp_vcf_idx="", # common snp vcf index
                   normal_vcf_path ="",  # Path to normal genotype VCF file, must be gzipped
                   normal_vcf_idx= "", #index for genotype VCF file
                   sample_names = "normal_id tumor_id",
                   chromosomes = "",
                   chr_notation = False,  # True if contigs have "chr" prefix
                   phase_snps = True
                   ):
    
#     # task to generate [CHR POS] vcf file from standard SNP file
#     vcf_format_shim_task = Task(name = "format_vcf",
#                                 inputs = {"normal_vcf_path": normal_vcf_path},
#                                 script = "cut -f 1,2  > vcf_formatted.txt",
#                                 outputs = {"vcf_file": "vcf_formatted.txt"})
    
    # bams may be shared with other workflows, so we seperate their localization
    # to allow for reuse
    bam_localization_task = LocalizeToDisk(name="localize_bams",
                                           files = {
                                                    "n_bam": normal_bam,
                                                    "n_bai": normal_bai
                                                    }
                                           )
    # localize other common files
    localization_task = LocalizeToDisk(name="localize_ref_files", 
                                       files = {"vcf_file": normal_vcf_path,
                                                "vcf_idx": normal_vcf_idx,
                                                "common_snp_vcf": common_snp_vcf,
                                                "common_snp_vcf_idx": common_snp_vcf_idx,
                                                "ref_fasta": ref_fasta,
                                                "ref_fasta_idx": ref_fasta_idx,
                                                "ref_fasta_dict": ref_fasta_dict
                                                }
                                       )
    
    # since hatchet needs a tumor to run on, we'll just subsample the normal to .01x
    subsample_task = subsample_bam(inputs = {"bam": bam_localization_task["n_bam"]})

    # if running on one sample, make normal into tumor (processing is done seperately on each bam)
    # hatchet only runs on autosome
    chromosomes = [f'chr{i}' for i in range(1,23)]
   
     
    if normal_vcf_path == "":
        # no genotype passed, must genotype normal    
        genotype_snps_task = HATCHET_genotype_snps(inputs = {"ref_fasta": localization_task["ref_fasta"],
                                                         "normal_bam": bam_localization_task["n_bam"], 
                                                         "normal_bai": bam_localization_task["n_bai"],
                                                         "vcf_file": localization_task["common_snp_vcf"],
                                                         "vcf_idx": localization_task["common_snp_vcf_idx"],
                                                         "chromosomes": chromosomes})
    else:
        # need to split up single vcf by chromosome for sharding
        chrom_vcf_task = wolf.Task(name = "split_vcf_task",
                                   inputs = {"vcf_file": localization_task["vcf_file"],
                                             "chromosome": chromosomes},
                                   script = """
                                   bcftools view ${vcf_file} -t ${chromosome} -o ${chromosome}.vcf.gz
                                   bcftools index ${chromosome}.vcf.gz
                                   """,
                                   
                                   outputs = {"chrom_vcf" : "*.vcf.gz",
                                              "chrom_vcf_idx": "*.vcf.gz.csi"
                                             },
                                   docker = "gcr.io/broad-getzlab-workflows/hatchet:v0"
                                )

    
    count_alleles_task = HATCHET_count_alleles(inputs = {"ref_fasta": localization_task["ref_fasta"],
                                                         "normal_bam": bam_localization_task["n_bam"],
                                                         "normal_bai": bam_localization_task["n_bai"],
                                                         "tumor_bam": subsample_task["subsampled_bam"],
                                                         "tumor_bai": subsample_task["subsampled_bai"],
                                                         "vcf_file": genotype_snps_task["germline_vcfs"] if normal_vcf_path == "" else chrom_vcf_task["chrom_vcf"],
                                                         "vcf_file_idx": genotype_snps_task["germline_vcfs_idx"] if normal_vcf_path == "" else chrom_vcf_task["chrom_vcf_idx"],
                                                         "sample_names": sample_names,
                                                         "chromosomes": chromosomes})
  
    
    count_reads_task = HATCHET_count_reads(inputs = {"reference_genome_version": reference_genome_version,
                                                             "normal_bam": bam_localization_task["n_bam"],
                                                             "normal_bai": bam_localization_task["n_bai"],
                                                             "tumor_bam": subsample_task["subsampled_bam"],
                                                             "tumor_bai": subsample_task["subsampled_bai"],
                                                             "normal_baf": count_alleles_task["normal_snp_depths"],
                                                             "sample_names": sample_names,
                                                             "chromosomes": chromosomes})


    output_dict = {'count_alleles_task': count_alleles_task,
                   'count_reads_task': count_reads_task}
    
    if phase_snps:
        download_phasing_panel_task = HATCHET_download_phasing_panel()
        phase_snps_task = HATCHET_phase_snps(inputs = {"ref_fasta": localization_task["ref_fasta"],
                                                       "ref_fasta_idx": localization_task["ref_fasta_idx"],
                                                       "ref_fasta_dict": localization_task["ref_fasta_dict"],
                                                       "reference_panel_dir": [download_phasing_panel_task["ref_panel_dir"]],
                                                       "snps_vcf": genotype_snps_task["germline_hets"] if normal_vcf_path=="" else [chrom_vcf_task["chrom_vcf"]],
                                                       "reference_genome_version": reference_genome_version,
                                                       "chr_notation": chr_notation
        })   
        output_dict['phase_snps_task'] = phase_snps_task

    post_process_allelecounts = wolf.Task(
        name = "postprocess_allelecounts",
        inputs = {"snp_counts_array": [count_alleles_task["normal_snp_depths"]]},
        script = """
        touch ./all_snps.txt;
        for f in $(cat ${snp_counts_array}); do cat $f  >> ./all_snps.txt; done
        """,
        
        outputs = {"all_snps": "all_snps.txt"}
     )

    post_process_totals_task = reformat_hatchet_depth_outputs(inputs = {
                                                            "all_depths_paths": [count_reads_task["total_counts_file"]],
                                                            "sample_path": count_reads_task["sample_names_file"][0]
                                                            }
                                                        )
    
    return output_dict
        

def HATCHET_Main(tumor_snp_depths=None,
                 count_reads_dir=None,
                 total_counts_file=None,
                 reference_genome_version="hg38",
                 phased_vcf = None):
    
    combine_counts_task = HATCHET_combine_counts(inputs = {"tumor_baf": tumor_snp_depths,
                                                           "count_reads_dir": count_reads_dir,
                                                           "total_counts_file": total_counts_file,
                                                           "reference_genome_version": reference_genome_version,
                                                           "phased_vcf_file": phased_vcf})
    
    cluster_bins_task = HATCHET_cluster_bins(inputs = {"bb_file": combine_counts_task["binned_counts_file"]})
    
    plot_bins_task = HATCHET_plot_bins(inputs = {
                                                           "clustered_bins": cluster_bins_task["clustered_bins"],
                                                           "clustered_segs": cluster_bins_task["clustered_segments"]
                                                        }
                                              )

    return {'combine_counts_task': combine_counts_task,
            'cluster_bins_task': cluster_bins_task,
            'plot_bins_task': plot_bins_task}
