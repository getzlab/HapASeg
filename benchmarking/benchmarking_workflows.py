import wolf
import sys
import numpy as np

from sim_file_generation_tasks import *
from output_comparison_tasks import *
from wolf.localization import LocalizeToDisk
from gatk_wolf_task.gatk_wolf_task import GATK_CNV_Workflow
from ascat_wolf_task.ascat_wolf_task import ASCAT
from facets_wolf_task.facets_wolf_task import Facets
#from hatchet_wolf_task.hatchet_wolf_task import HATCHET_Depths, HATCHET_Main

sys.path.append('../wolF/')
from workflow import workflow as HapASeg_Workflow

#HapASeg
def HapASeg_Sim_Workflow(sim_profile=None,
                         purity=None,
                         sample_label=None,
                         normal_vcf_path=None,
                         phased_vcf_path=None, # path to phased vcf (cached eagle output)
                         hetsite_depth_path=None,
                         covcollect_path=None,
                         genotype_file=None, # normal genotype of NA12878
                         ref_build=None,
                         ref_fasta=None,
                         cytoband_file=None,
                         ground_truth_seg_file=None,
                         target_list=2000,
                         cleanup_disks=False
                ):
    localization_task = LocalizeToDisk(files = {
                                    "normal_vcf":normal_vcf_path,
                                    "hetsite_depth":hetsite_depth_path,
                                    "covcollect": covcollect_path,
                                    "phased_vcf":phased_vcf_path,
                                    "target_list" : target_list if not isinstance(target_list, int) else "",
                                    "ref_fasta" : ref_fasta,
                                    "cytoband_file" : cytoband_file
                                    })
    generate_hapaseg_files_task = Generate_HapASeg_Sim_Data( inputs = {
                                    "sim_profile": sim_profile,
                                    "purity":purity,
                                    "sample_label":sample_label,
                                    "normal_vcf_path":localization_task["normal_vcf"],
                                    "hetsite_depth_path": localization_task["hetsite_depth"],
                                    "covcollect_path":localization_task["covcollect"]
                                    })

    hapaseg_seg_file = HapASeg_Workflow(hetsites_file = generate_hapaseg_files_task["hapaseg_hets"],
                                   tumor_coverage_bed = generate_hapaseg_files_task["hapaseg_coverage_bed"],
                                   phased_vcf = localization_task["phased_vcf"],
                                   ref_genome_build = ref_build,
                                   target_list = target_list if isinstance(target_list, int) else localization_task["target_list"],
                                   cleanup_disks=cleanup_disks
                                   )
    
    hapaseg_downstream = Downstream_HapASeg_Analysis(inputs = {
                                   "hapaseg_seg_file": hapaseg_seg_file,
                                   "ground_truth_seg_file": ground_truth_seg_file,
                                   "sample_name": f"{sample_label}_{np.around(purity, 2)}",
                                   "ref_fasta": localization_task["ref_fasta"],
                                   "cytoband_file": localization_task["cytoband_file"]
                        })
    
#GATK
def GATK_Sim_Workflow(sim_profile = None,
                      purity = None,
                      sample_label = None,
                      normal_vcf_path = None,
                      variant_depth_path = None,
                      coverage_tsv_path = None,
                      sim_normal_allelecounts_path=None,
                      raw_gatk_allelecounts_path=None,
                      raw_gatk_coverage_path=None,
                      sequence_dictionary=None,
                      count_panel="",
                      annotated_intervals="",
                      ground_truth_seg_file=None,
                      ref_fasta = None,
                      cytoband_file=None
        ):

    if (count_panel == "" and annotated_intervals == "") or (count_panel != "" and annotated_intervals != ""):
        raise ValueError("Must provide either a count panel OR a annotated intervals file")
    
    localization_task = LocalizeToDisk(files = {
                            "normal_vcf": normal_vcf_path,
                            "variant_depth":variant_depth_path,
                            "coverage_tsv":coverage_tsv_path,
                            "sim_normal_allelecounts":sim_normal_allelecounts_path,
                            "raw_gatk_allelecounts":raw_gatk_allelecounts_path,
                            "raw_gatk_coverage":raw_gatk_coverage_path,
                            "seq_dict":sequence_dictionary,
                            "count_panel":count_panel,
                            "annotated_intervals":annotated_intervals,
                            "ref_fasta": ref_fasta,
                            "cytoband_file":cytoband_file
                         })
                            
    generate_gatk_data_task = Generate_GATK_Sim_Data(
              inputs = {"sim_profile": sim_profile,
                        "purity": purity,
                        "sample_label": sample_label,
                        "normal_vcf_path": localization_task["normal_vcf"],
                        "variant_depth_path": localization_task["variant_depth"],
                        "coverage_tsv_path": localization_task["coverage_tsv"],
                        "sim_normal_allelecounts_path": localization_task["sim_normal_allelecounts"],
                        "raw_gatk_allelecounts_path":localization_task["raw_gatk_allelecounts"],
                        "raw_gatk_coverage_path":localization_task["raw_gatk_coverage"]
                       }
                )    
    gatk_segfile = GATK_CNV_Workflow(sample_name = sample_label + f'_{np.around(purity, 2)}',
                      tumor_frag_counts_hdf5 = generate_gatk_data_task["tumor_frag_counts"],
                      tumor_allele_counts = generate_gatk_data_task["tumor_allele_counts"],
                      count_panel = localization_task["count_panel"] if count_panel != "" else "",
                      annotated_intervals=localization_task["annotated_intervals"] if annotated_intervals != "" else "",
                      sequence_dictionary = localization_task["seq_dict"]
                     )

    gatk_downstream = Downstream_GATK_Analysis(
              inputs={"gatk_sim_cov_input": generate_gatk_data_task["tumor_coverage_tsv"],
                      "gatk_sim_acounts" : generate_gatk_data_task["tumor_allele_counts"],
                      "gatk_seg_file" : gatk_segfile,
                      "ground_truth_seg_file": ground_truth_seg_file,
                      "sample_name": f"{sample_label}_{np.around(purity, 2)}",
                      "ref_fasta": localization_task["ref_fasta"],
                      "cytoband_file": localization_task["cytoband_file"]
                    }
            )

# Facets
def Facets_Sim_Workflow(sim_profile = None,
                      purity = None,
                      sample_label = None,
                      normal_vcf_path = None,
                      variant_depth_path = None,
                      filtered_variants_path = None,
                      ground_truth_seg_file=None,
                      ref_fasta=None,
                      cytoband_file=None
                ):
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "variant_depth":variant_depth_path,
                                                "filtered_variants":filtered_variants_path,
                                                "ref_fasta": ref_fasta,
                                                "cytoband_file": cytoband_file
                                               }
                                     ) 

    generate_facets_data_task = Generate_Facets_Sim_Data(inputs = {
                                    "sim_profile" : sim_profile,
                                    "purity": purity,
                                    "sample_label": sample_label,
                                    "normal_vcf_path":localization_task["normal_vcf"],
                                    "variant_depth_path": localization_task["variant_depth"],
                                    "filtered_variants_path": localization_task["filtered_variants"]
                                    })

    run_facets_task = Facets(inputs = {"snp_counts": generate_facets_data_task["facets_input_counts"]})
    
    facets_downstream = Downstream_Facets_Analysis(
                            inputs={"facets_input_counts": generate_facets_data_task["facets_input_counts"],
                                    "facets_seg_file": run_facets_task["facets_seg_file"],
                                     "ground_truth_seg_file": ground_truth_seg_file,
                                     "sample_name": f"{sample_label}_{np.around(purity, 2)}",
                                     "ref_fasta": localization_task["ref_fasta"],
                                     "cytoband_file": localization_task["cytoband_file"]
                                   }
                            )
                                    
                                    
# ASCAT
def ASCAT_Sim_Workflow(sim_profile=None,
                       purity=None,
                       sample_label=None,
                       normal_vcf_path=None,
                       variant_depth_path=None,
                       filtered_variants_path=None,
                       GC_correction_file=None,
                       RT_correction_file=None,
                       ground_truth_seg_file=None,
                       ref_fasta=None,
                       cytoband_file=None
            ):
    
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "variant_depth":variant_depth_path,
                                                "filtered_variants":filtered_variants_path,
                                                "GC_correction_file":GC_correction_file,
                                                "RT_correction_file": RT_correction_file,
                                                "ref_fasta" : ref_fasta,
                                                "cytoband_file" : cytoband_file
                                               }
                                     )
    
    generate_ascat_data_task = Generate_ASCAT_Sim_Data(inputs = { 
                                        "sim_profile": sim_profile,
                                        "purity": purity,
                                        "sample_label": sample_label,
                                        "normal_vcf_path":localization_task["normal_vcf"],
                                        "variant_depth_path": localization_task["variant_depth"],
                                        "filtered_variants_path": localization_task["filtered_variants"]
                                        })

    run_ascat_task = ASCAT(inputs = {"tumor_LogR" : generate_ascat_data_task["ascat_tumor_logR"],
                                      "tumor_BAF" : generate_ascat_data_task["ascat_tumor_BAF"],
                                      "normal_LogR" : generate_ascat_data_task["ascat_normal_logR"],
                                      "normal_BAF" : generate_ascat_data_task["ascat_tumor_BAF"],
                                      "GC_correction_file": localization_task["GC_correction_file"],
                                      "RT_correction_file": localization_task["RT_correction_file"]
                                    }
                         )
    
    ascat_downstream = Downstream_ASCAT_Analysis(
                                inputs={"ascat_t_logr": generate_ascat_data_task["ascat_tumor_logR"],
                                        "ascat_t_baf": generate_ascat_data_task["ascat_tumor_BAF"],
                                        "ascat_seg_file": run_ascat_task["ascat_raw_segments"],
                                        "ground_truth_seg_file": ground_truth_seg_file,
                                        "sample_name": f"{sample_label}_{np.around(purity, 2)}",
                                        "ref_fasta": localization_task["ref_fasta"],
                                        "cytoband_file": localization_task["cytoband_file"]
                                       }
                                )
    
# HATCHet
def HATCHet_Sim_Workflow(sim_profile=None,
                         purity=None,
                         sample_label=None,
                         normal_vcf_path=None,  # most be compressed VCF file
                         reference_genome_path=None,  # FASTA  
                         normal_bam=None,
                         tumor_bam=None,
                         sample_names = None,
                         chromosomes = [],  # empty list if whole genome
                         reference_genome_version=None,
                         chr_notation = False,  # True if contigs have "chr" prefix
                         phase_snps = True
            ):
    
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "normal_bam": normal_bam,
                                                "tumor_bam": tumor_bam,
                                                "reference_genome_path": reference_genome_path
                                })
    
    run_hatchet_depths_workflow = HATCHET_Depths( reference_genome_path = LocalizeToDisk["reference_genome_path"],  
                     normal_bam = LocalizeToDisk["normal_bam"],
                     tumor_bam = LocalizeToDisk["tumor_bam"],
                     normal_vcf_path = LocalizeToDisk["normal_vcf_path"],
                     sample_names = sample_names,
                     chromosomes = chromosomes,
                     reference_genome_version = reference_genome_version,
                     chr_notation = chr_notation,
                     phase_snps = phase_snps
                     )
    
    
    generate_hatchet_sim_task = Generate_HATCHet_Sim_Data(inputs = { 
                                        "sim_profile": sim_profile,
                                        "purity": purity,
                                        "sample_label": sample_label,
                                        "normal_vcf_path":localization_task["normal_vcf"],
                                        "tumor_baf_path": run_hatchet_depths_workflow["count_alleles_task"]["tumor_snp_depths"],
                                        "total_reads_paths": run_hatchet_depths_workflow["count_reads_task"]["coverage_totals"],
                                        "thresholds_snps_paths": run_hatchet_depths_workflow["count_reads_task"]["snp_thresholds"],
                                        "total_tsv_path": run_hatchet_depths_workflow["count_reads_task"]["total_counts_file"]
                                        })

    run_hatchet_main_workflow = HATCHET_Main(tumor_snp_depths = generate_hatchet_sim_task["hatchet_tumor_snp_depths"],
                 count_reads_dir = generate_hatchet_sim_task["hatchet_total_bin_reads_dir"],  # todo - check
                 total_counts_file = generate_hatchet_sim_task["hatchet_total_counts"],
                 reference_genome_version = reference_genome_version,
                 phase_snps = phase_snps,
                 phased_vcf = None if not phase_snps else generate_hatchet_sim_task["phase_snps_task"]["phased_vcf"]
                 )
            
# run all pipelines

def Run_Sim_Workflows(sim_profile=None,
                      purity=None,
                      sample_label=None,
                      normal_vcf_path=None,
                      ref_build=None,
                      ref_fasta=None,
                      cytoband_file=None,
                      ground_truth_purity=0.7, # compare all samples to a standard purity gt for MAD score consistancy
                      hapaseg_hetsite_depth_path=None,
                      hapaseg_covcollect_path=None,
                      hapaseg_target_list=2000,
                      hapaseg_phased_vcf_path=None, # path to cached eagle combined output
                      hapaseg_cleanup_disks=False,
                      gatk_variant_depth_path = None,
                      gatk_coverage_tsv_path = None,
                      gatk_sim_normal_allelecounts_path=None,
                      gatk_raw_gatk_allelecounts_path=None,
                      gatk_raw_gatk_coverage_path=None,
                      gatk_sequence_dictionary=None,
                      gatk_count_panel="",
                      gatk_annotated_intervals="",
                      facets_variant_depth_path = None,
                      facets_filtered_variants_path = None,
                      ascat_variant_depth_path=None,
                      ascat_filtered_variants_path=None,
                      ascat_GC_correction_file=None,
                      ascat_RT_correction_file=None,
                      hatchet_reference_genome_path=None,
                      hatchet_normal_bam=None,
                      hatchet_tumor_bam=None,
                      hatchet_sample_names = None,
                      hatchet_chromosomes = [],
                      hatchet_reference_genome_version="hg38",
                      hatchet_chr_notation = False,
                      hatchet_phase_snps = True
                      ):
    seg_file_gen_task = Generate_Groundtruth_Segfile(inputs= {
                            "sample_label": sample_label,
                            "purity":ground_truth_purity if ground_truth_purity is not None else purity,
                            "sim_profile":sim_profile,
                            "normal_vcf_path":normal_vcf_path,
                            "hapaseg_hetsite_depth_path": hapaseg_hetsite_depth_path,
                            "hapaseg_coverage_tsv":hapaseg_covcollect_path
                        }
                    ) 

    HapASeg_Sim_Workflow(sim_profile = sim_profile,
                        purity = purity,
                        sample_label = sample_label,
                        normal_vcf_path = normal_vcf_path,
                        hetsite_depth_path = hapaseg_hetsite_depth_path,
                        covcollect_path = hapaseg_covcollect_path,
                        phased_vcf_path = hapaseg_phased_vcf_path,
                        ref_build=ref_build,
                        ref_fasta=ref_fasta,
                        cytoband_file=cytoband_file,
                        ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"],
                        target_list = hapaseg_target_list,
                        cleanup_disks = hapaseg_cleanup_disks
                        )

    GATK_Sim_Workflow(sim_profile=sim_profile,
                      purity=purity,
                      sample_label=sample_label,
                      normal_vcf_path = normal_vcf_path,
                      variant_depth_path = gatk_variant_depth_path,
                      coverage_tsv_path = gatk_coverage_tsv_path,
                      sim_normal_allelecounts_path = gatk_sim_normal_allelecounts_path,
                      raw_gatk_allelecounts_path = gatk_raw_gatk_allelecounts_path,
                      raw_gatk_coverage_path = gatk_raw_gatk_coverage_path,
                      sequence_dictionary = gatk_sequence_dictionary,
                      count_panel = gatk_count_panel,
                      annotated_intervals = gatk_annotated_intervals,
                      ref_fasta=ref_fasta,
                      cytoband_file=cytoband_file,
                      ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"]
                      )

    Facets_Sim_Workflow(sim_profile=sim_profile,
                        purity=purity,
                        sample_label=sample_label,
                        normal_vcf_path = normal_vcf_path,
                        variant_depth_path = facets_variant_depth_path,
                        filtered_variants_path = facets_filtered_variants_path,
                        ref_fasta=ref_fasta,
                        cytoband_file=cytoband_file,
                        ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"]
                        )
    
    ASCAT_Sim_Workflow(sim_profile = sim_profile,
                       purity = purity,
                       sample_label = sample_label,
                       normal_vcf_path = normal_vcf_path,
                       variant_depth_path = ascat_variant_depth_path,
                       filtered_variants_path = ascat_filtered_variants_path,
                       GC_correction_file = ascat_GC_correction_file,
                       RT_correction_file = ascat_RT_correction_file,
                       ref_fasta=ref_fasta,
                       cytoband_file=cytoband_file,
                       ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"]
                      )

#    HATCHet_Sim_Workflow(sim_profile=sim_profile,
#                         purity=purity,
#                         sample_label=sample_label,
#                         normal_vcf_path=normal_vcf_path,
#                         reference_genome_path=hatchet_reference_genome_path,
#                         normal_bam=hatchet_normal_bam,
#                         tumor_bam=hatchet_tumor_bam,
#                         sample_names = hatchet_sample_names,
#                         chromosomes = hatchet_chromosomes,
#                         reference_genome_version=hatchet_reference_genome_version,
#                         chr_notation = hatchet_chr_notation,
#                         phase_snps = hatchet_phase_snps
#            )
