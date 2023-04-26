import wolf
from wolf import LocalizeToDisk
from .gatk_generate_raw_task import GATK_Generate_Raw_Data
from .gatk_wolf_task import GATK_CNV_Workflow
import sys
sys.path.append('../')
from output_comparison_tasks import *

# workflow for running gatk pipeline from normal/tumor bams

def GATK_standard_pipeline(
        tumor_bam=None,
        tumor_bai=None,
        normal_bam=None,
        normal_bai=None,
        vcf_file=None, # for calling het sites
        ref_fasta=None,
        ref_fasta_idx=None,
        ref_fasta_dict=None,
        cytoband_file=None,
        interval_list=None,
        interval_name=None,
        sample_name=None,
        bin_length=1000,
        padding=250, #should be 0 for WGS, 250 for WES
        exclude_sex=True,
        upload_bucket=None,
        exclude_chroms="",
        count_panel=None,
        localization_token=None,
        persistent_disk_dry_run=False
    ):
    print("localization flag: {}".format(persistent_dry_run))   
    # localize_bams
    tumor_loc_task = LocalizeToDisk(
          files = {
            "t_bam" : tumor_bam,
            "t_bai" : tumor_bai,
          },
        token=localization_token,
        persistent_disk_dry_run = persistent_disk_dry_run
        )

    normal_loc_task = LocalizeToDisk(
          files = {
            "n_bam" : normal_bam,
            "n_bai" : normal_bai
          },
        token=localization_token,
        persistent_disk_dry_run = persistent_disk_dry_run
        )
    
    t_bam = tumor_loc_task['t_bam']
    t_bai = tumor_loc_task['t_bai']
    n_bam = normal_loc_task['n_bam']
    n_bai = normal_loc_task['n_bai']
    
    # localize ref files
    localization_task = LocalizeToDisk(
      files = dict(
        ref_fasta = ref_fasta,
        ref_fasta_idx = ref_fasta_idx,
        ref_fasta_dict = ref_fasta_dict,
        vcf_file = vcf_file,
        interval_list = interval_list,
        count_panel = count_panel if count_panel is not None else ""
    ))  
 
    raw_tumor_data = GATK_Generate_Raw_Data(
                           input_bam=t_bam,
                           input_bai=t_bai,
                           vcf_file=localization_task["vcf_file"],
                           ref_fasta=localization_task["ref_fasta"],
                           ref_fasta_idx=localization_task["ref_fasta_idx"],
                           ref_fasta_dict=localization_task["ref_fasta_dict"],
                           interval_list=localization_task["interval_list"],
                           interval_name=interval_name,
                           sample_name=sample_name + '_tumor',
                           bin_length=bin_length,
                           padding=padding,
                           exclude_sex=exclude_sex,
                           upload_bucket=upload_bucket,
                           persistent_dry_run = False, 
                           preprocess_tumor_bam = False,
                           exclude_chroms = "")
    
    raw_normal_data = GATK_Generate_Raw_Data(
                           input_bam=n_bam,
                           input_bai=n_bai,
                           vcf_file=localization_task["vcf_file"],
                           ref_fasta=localization_task["ref_fasta"],
                           ref_fasta_idx=localization_task["ref_fasta_idx"],
                           ref_fasta_dict=localization_task["ref_fasta_dict"],
                           interval_list=localization_task["interval_list"],
                           interval_name=interval_name,
                           sample_name=sample_name + '_normal',
                           bin_length=bin_length,
                           padding=padding,
                           exclude_sex=exclude_sex,
                           upload_bucket=upload_bucket,
                           persistent_dry_run = False, 
                           preprocess_tumor_bam = False,
                           exclude_chroms = "")

    gatk_pipeline = GATK_CNV_Workflow(sample_name = sample_name,
                                      tumor_frag_counts_hdf5 = raw_tumor_data['gatk_frag_counts'],
                                      count_panel = localization_task["count_panel"] if count_panel is not None else raw_normal_data['gatk_single_sample_pon'],
                                      tumor_allele_counts =  raw_tumor_data['gatk_allele_counts'],
                                      normal_allele_counts =  raw_normal_data['gatk_allele_counts'],
                                      sequence_dictionary = localization_task["ref_fasta_dict"])
    
    Standard_GATK_Plotting(inputs={'gatk_seg_file': gatk_pipeline,
                                   'sample_name': sample_name,
                                   'ref_fasta': localization_task["ref_fasta"],
                                   'cytoband_file': cytoband_file
                                   }
                          )
