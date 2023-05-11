import wolf
from wolf.localization import LocalizeToDisk, UploadToBucket
import prefect
import sys
sys.path.append('../')
from output_comparison_tasks import *
from .tasks import *

def Hatchet_Generate_Raw(sample_name = None,
                   ref_fasta=None, # ref genome fasta file
                   ref_fasta_idx=None, # ref genome fai file  
                   ref_fasta_dict=None, # ref genome fai file  
                   reference_genome_version=None,
                   tumor_bam = None,
                   tumor_bai = None,
                   normal_bam = None,
                   normal_bai = None,
                   common_snp_vcf="", # vcf file containing common snp sites
                   common_snp_vcf_idx="", # common snp vcf index
                   normal_vcf_path ="",  # Path to normal genotype VCF file, must be gzipped
                   normal_vcf_idx= "", #index for genotype VCF file
                   sample_names = "normal_id tumor_id",
                   chromosomes = "",
                   phase_snps = True,
                   upload_bucket = None, # upload to gcloud if not none
                   second_tumor_bam = None, # optional third sample
                   second_tumor_bai = None, # optional third sample
                   ):
    
    chr_notation = True if reference_genome_version == "hg38" else False
    
    # if a second tumor bam is passed to allow us to get allelecounts at non-het sites
    # we set dummy normal to true to use the real normal and tumor counts in preprocessing
    dummy_normal = True if second_tumor_bam is not None else False
    
    t_bam_localization_task = LocalizeToDisk(name="localize_tbam",
                                           files = {
                                                    "bam": tumor_bam,
                                                    "bai": tumor_bai
                                                    }
                                           )
    
    n_bam_localization_task = LocalizeToDisk(name="localize_nbam",
                                           files = {
                                                    "bam": normal_bam,
                                                    "bai": normal_bai
                                                    }
                                           )
    if second_tumor_bam is not None:
        assert second_tumor_bai is not None
        t2_bam_localization_task = LocalizeToDisk(name="localize_t2bam",
                                               files = {
                                                        "bam": second_tumor_bam,
                                                        "bai": second_tumor_bai
                                                        }
                                               )
    # localize other common files
    localization_task = LocalizeToDisk(name="localize_ref_files", 
                                       files = {"normal_vcf_file": normal_vcf_path,
                                                "normal_vcf_idx": normal_vcf_idx,
                                                "common_snp_vcf": common_snp_vcf,
                                                "common_snp_vcf_idx": common_snp_vcf_idx,
                                                "ref_fasta": ref_fasta,
                                                "ref_fasta_idx": ref_fasta_idx,
                                                "ref_fasta_dict": ref_fasta_dict
                                                }
                                       )
    
    
    # if running on one sample, make normal into tumor (processing is done seperately on each bam)
    # hatchet only runs on autosome
    if chromosomes != "":
        chrs_string = chromosomes
        chromosomes = chromosomes.split(" ")
    else:
        chromosomes = [f'chr{i}' for i in range(1,23)]
        chrs_string = None
     
    genotype_snps_task = HATCHET_genotype_snps(inputs = {"ref_fasta": localization_task["ref_fasta"],
                                                     "normal_bam": n_bam_localization_task["bam"], 
                                                     "normal_bai": n_bam_localization_task["bai"],
                                                     "vcf_file": localization_task["normal_vcf_file"] if normal_vcf_path != "" else localization_task["common_snp_vcf"],
                                                     "vcf_idx": localization_task["normal_vcf_idx"] if normal_vcf_idx != "" else localization_task["common_snp_vcf_idx"],
                                                     "chromosomes": chromosomes},
                                                preemptible=False                                   
                                                )

    index_genotypes_task = wolf.Task(name = "index_vcfs",
                                     inputs = {"vcf": genotype_snps_task["germline_snp_vcfs"]},
                                     script = """bcftools index -o ./$(basename ${vcf}).csi ${vcf}""",
                                     outputs = {"vcf_idx":"*.vcf.gz.csi"},
                                     docker = "gcr.io/broad-getzlab-workflows/hatchet:v2")

    
    count_alleles_task = HATCHET_count_alleles(inputs = {"ref_fasta": localization_task["ref_fasta"],
                                                         "normal_bam": n_bam_localization_task["bam"],
                                                         "normal_bai": n_bam_localization_task["bai"],
                                                         "tumor_bam": t_bam_localization_task["bam"],
                                                         "tumor_bai": t_bam_localization_task["bai"],
                                                         "second_tumor_bam": t2_bam_localization_task["bam"] if second_tumor_bam is not None else "",
                                                         "second_tumor_bai": t2_bam_localization_task["bai"] if second_tumor_bam is not None else "",

                                                         "vcf_file": genotype_snps_task["germline_snp_vcfs"],
                                                         "vcf_file_idx": index_genotypes_task["vcf_idx"],
                                                         "sample_names": sample_names,
                                                         "chromosomes": chromosomes})
  
    
    gather_allelecounts = wolf.Task(
        name = "postprocess_allelecounts",
        inputs = {"normal_snp_counts_array": [count_alleles_task["normal_snp_depths"]],
                  "tumor_snp_counts_array": [count_alleles_task["tumor_snp_depths"]]},
        script = """
        touch ./normal_snps.txt;
        touch ./tumor_snps.txt;
        for f in $(cat ${normal_snp_counts_array}); do cat $f  >> ./normal_snps.txt; done
        for f in $(cat ${tumor_snp_counts_array}); do cat $f  >> ./tumor_snps.txt; done
        """,
        
        outputs = {"tumor_snps": "tumor_snps.txt",
                   "normal_snps": "normal_snps.txt"}
     )
 
    count_reads_task = HATCHET_count_reads(inputs = {"reference_genome_version": reference_genome_version,
                                                             "normal_bam": n_bam_localization_task["bam"], 
                                                             "normal_bai": n_bam_localization_task["bai"],
                                                             "tumor_bam": t_bam_localization_task["bam"],
                                                             "tumor_bai": t_bam_localization_task["bai"],
                                                             "tumor_baf": gather_allelecounts["tumor_snps"],
                                                             "sample_names": sample_names,
                                                             "second_tumor_bam":t2_bam_localization_task["bam"] if second_tumor_bam is not None else "",
                                                             "second_tumor_bai": t2_bam_localization_task["bai"] if second_tumor_bam is not None else "", 
                                                             "chromosomes": chrs_string if chrs_string is not None else ""},
                                           preemptible = False,
                                          )

    if dummy_normal:
        # tumor bam contains two samples, need to filter out our real normal for downstream simulations
        filter_normal_sample = wolf.Task(name = "filter_normal_snp_sample",
                                         inputs = {"normal_name": sample_names.split(' ')[1],
                                                   "tumor_name" : sample_names.split(' ')[2], 
                                                   "tumor_snps": gather_allelecounts["tumor_snps"]
                                                  },
                                         script = """python -c "import pandas as pd; df = pd.read_csv('${tumor_snps}', sep='\t', names=['CHR', 'POS', 'SAMPLE', 'REF', 'ALT']); normal_df = df.loc[df.SAMPLE =='${normal_name}']; tumor_df = df.loc[df.SAMPLE == '${tumor_name}']; tumor_df = tumor_df.merge(normal_df[['CHR', 'POS']], on = ['CHR', 'POS'], how='inner'); tumor_df.to_csv('./tumor_snps.txt', index=False, header=False, sep='\t'); normal_df.to_csv('./normal_snps.txt', index=False, header=False, sep='\t')"
                                                  """,
                                         
                                         outputs = {"tumor_snps": "tumor_snps.txt",
                                                    "normal_snps": "normal_snps.txt"}
                                        )   
 

    output_dict = {'count_alleles_task': count_alleles_task,
                   'gather_allelecounts': gather_allelecounts,
                   'count_reads_task': count_reads_task}
   
          
    if phase_snps:
        download_phasing_panel_task = HATCHET_download_phasing_panel()
        phase_snps_task = HATCHET_phase_snps(inputs = {"ref_fasta": localization_task["ref_fasta"],
                                                       "ref_fasta_idx": localization_task["ref_fasta_idx"],
                                                       "ref_fasta_dict": localization_task["ref_fasta_dict"],
                                                       "reference_panel_dir": [download_phasing_panel_task["ref_panel_dir"]],
                                                       "snps_vcf": [genotype_snps_task["germline_snp_vcfs"]],
                                                       "reference_genome_version": reference_genome_version,
                                                       "chr_notation": chr_notation
                                                      },
                                             preemptible = False
        )   
        output_dict['phase_snps_task'] = phase_snps_task


   
    preprocess_task = HATCHET_preprocess(inputs = {"totals_paths": [count_reads_task["coverage_totals"]],
                                                   "thresholds_paths": [count_reads_task["snp_thresholds"]],
                                                   "tumor_baf_path": gather_allelecounts["tumor_snps"] if not dummy_normal else filter_normal_sample["tumor_snps"],
                                                   "sample_name" : sample_name,
                                                   "dummy_normal": dummy_normal
                                                   }
                                        )
    if upload_bucket is not None:
        preprocess_upload_task = UploadToBucket(files = [preprocess_task["interval_counts"],
                                              preprocess_task["pos_counts"],
                                              preprocess_task["snp_counts"],
                                              preprocess_task["read_combined"]],
                                     bucket = upload_bucket.rstrip("/") + '/hatchet/preprocess_data/') 

        read_totals_upload_task = UploadToBucket(files = count_reads_task["coverage_totals"],
                                     bucket = upload_bucket.rstrip("/") + '/hatchet/raw_data/read_counts/totals/')

        read_thresholds_upload_task = UploadToBucket(files = count_reads_task["snp_thresholds"],
                                     bucket = upload_bucket.rstrip("/") + '/hatchet/raw_data/read_counts/thresholds/')

        if dummy_normal:
            allele_raw_upload_task = UploadToBucket(files = [filter_normal_sample["tumor_snps"],
                                                             filter_normal_sample["normal_snps"]],
                                         bucket = upload_bucket.rstrip("/") + '/hatchet/raw_data/allele_counts/')
        else:
            allele_raw_upload_task = UploadToBucket(files = [gather_allelecounts["tumor_snps"],
                                                             gather_allelecounts["normal_snps"]],
                                         bucket = upload_bucket.rstrip("/") + '/hatchet/raw_data/allele_counts/')

        if phase_snps:
            upload_phased_snps = UploadToBucket(files = phase_snps_task["phased_vcf"],
                                                bucket = upload_bucket.rstrip("/") + '/hatchet/')
    
    return output_dict
        

def Hatchet_Main(tumor_snp_depths=None,
                 count_reads_dir="", # path to dir containing either A) all count-reads outputs or B) all thesholds outputs
                 count_reads_array="", # path to txt file with paths to all thesholds outputs, additional totals must be passed seperately
                 total_counts_file=None,
                 reference_genome_version="hg38",
                 phased_vcf = None,
                 samples_file = "", # if simulation, must pass sim samples txt file here
                 additional_totals_files=None, # here, and will be softlinked into the directory containing the thresholds
                                               # if totals are being simulated, they may be passed seperately
                 fit_cn=False # run downstream cn fitting and plotting               
                ):                           
    
    combine_counts_task = HATCHET_combine_counts(inputs = {"tumor_baf": tumor_snp_depths,
                                                           "count_reads_dir": count_reads_dir,
                                                           "count_reads_array": count_reads_array,
                                                           "total_counts_file": total_counts_file,
                                                           "reference_genome_version": reference_genome_version,
                                                           "phased_vcf_file": phased_vcf,
                                                           "samples_file": samples_file,
                                                           "additional_totals_files" : additional_totals_files if additional_totals_files is not None else ""})
    
    cluster_bins_task = HATCHET_cluster_bins(inputs = {"bb_file": combine_counts_task["binned_counts_file"]})
    
    plot_bins_task = HATCHET_plot_bins(inputs = {
                                                           "clustered_bins": cluster_bins_task["clustered_bins"],
                                                           "clustered_segs": cluster_bins_task["clustered_segments"]
                                                        }
                                        )

    output_dict =  {
          'hatchet_seg_file': cluster_bins_task["clustered_segments"],
          'hatchet_bin_file': cluster_bins_task["clustered_bins"],
          'RD_plot': plot_bins_task["RD_plot"],
          'BAF_plot': plot_bins_task["BAF_plot"]
    }          
    

    if fit_cn:
        fit_cn_task = HATCHET_compute_cn(inputs = {'cluster_bins_bbc': cluster_bins_task["clustered_bins"],
                                                   'cluster_bins_seg': cluster_bins_task["clustered_segments"]
                                                  }
                                        )

        plot_cn_task = HATCHET_plot_cn(inputs = {'opt_bbc': fit_cn_task['best_bins_file']})
        
        output_dict['fit_cn_bbc'] = fit_cn_task['best_bins_file']
        output_dict['fit_cn_seg'] = fit_cn_task['best_segments_file']
        output_dict['fit_cn_plots'] = plot_cn_task['output_plots']

    return output_dict

def Hatchet_Run_Standard(sample_name = None,
                   ref_fasta=None, # ref genome fasta file
                   ref_fasta_idx=None, # ref genome fai file  
                   ref_fasta_dict=None, # ref genome fai file  
                   reference_genome_version=None,
                   tumor_bam = None,
                   tumor_bai = None,
                   normal_bam = None,
                   normal_bai = None,
                   common_snp_vcf="", # vcf file containing common snp sites
                   common_snp_vcf_idx="", # common snp vcf index
                   cytoband_file=None,
                   sample_names = "normal_id tumor_id",
                   chromosomes = "",
                   phase_snps = True,
                   upload_bucket = None, # upload to gcloud if not none
                   localization_token=None,
                   persistent_disk_dry_run=False
                   ):
    
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
        common_snp_vcf = common_snp_vcf,
        common_snp_vcf_idx = common_snp_vcf_idx,
        )
    )
 
    raw_data = Hatchet_Generate_Raw(sample_name = sample_name,
                   ref_fasta=localization_task["ref_fasta"],
                   ref_fasta_idx=localization_task["ref_fasta_idx"],
                   ref_fasta_dict=localization_task["ref_fasta_dict"],
                   reference_genome_version=reference_genome_version,
                   tumor_bam = t_bam,
                   tumor_bai = t_bai,
                   normal_bam = n_bam,
                   normal_bai = n_bai,
                   common_snp_vcf=localization_task["common_snp_vcf"],
                   common_snp_vcf_idx=localization_task["common_snp_vcf_idx"],
                   sample_names = sample_names,
                   chromosomes = chromosomes,
                   phase_snps = phase_snps,
                   )
    
    hatchet_run = Hatchet_Main(tumor_snp_depths=raw_data['gather_allelecounts']['tumor_snps'],
                 count_reads_array=[raw_data['count_reads_task']['snp_thresholds']],
                 total_counts_file= raw_data['count_reads_task']['total_counts_file'],
                 reference_genome_version=reference_genome_version,
                 phased_vcf = raw_data['phase_snps_task']['phased_vcf'],
                 samples_file = raw_data['count_reads_task']['sample_names_file'],
                 additional_totals_files=[raw_data['count_reads_task']['coverage_totals']],
                 fit_cn=True)

    # plot standard
    Standard_Hatchet_Plotting(inputs = {"hatchet_seg_file":hatchet_run['hatchet_seg_file'],
                                        "hatchet_bin_file": hatchet_run['hatchet_bin_file'],
                                        "sample_name": sample_name,
                                        "ref_fasta": localization_task["ref_fasta"],
                                        "cytoband_file": cytoband_file,
                                        }
                             )
    return hatchet_run
