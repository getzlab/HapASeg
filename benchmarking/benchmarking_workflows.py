import wolf
import sys
import numpy as np

from sim_file_generation_tasks import *
from output_comparison_tasks import *
from wolf.localization import LocalizeToDisk
from gatk_wolf_task.gatk_wolf_task import GATK_CNV_Workflow
from ascat_wolf_task.ascat_wolf_task import ASCAT
from facets_wolf_task.facets_wolf_task import Facets
from hatchet_wolf_task.workflow import Hatchet_Generate_Raw, Hatchet_Main

hapaseg = wolf.ImportTask('../')

#HapASeg
def HapASeg_Sim_Workflow(sim_profile=None,
                         purity=None,
                         sample_label=None,
                         normal_vcf_path=None,
                         phased_vcf_path=None, # path to phased vcf (cached eagle output)
                         hetsite_depth_path=None,
                         covcollect_path=None,
                         normal_covcollect_path="", # optional, will be used as covaraite if passed
                         genotype_file=None, # normal genotype of NA12878
                         ref_build=None,
                         ref_fasta=None,
                         cytoband_file=None,
                         ground_truth_seg_file=None,
                         target_list=2000,
                         run_cdp=False, # only applies to  WES, optionaly run CDP before ACDP
                         cleanup_disks=False,
                         is_ffpe=False, # flag to use faire tracks
                ):
    localization_task = LocalizeToDisk(files = {
                                    "normal_vcf":normal_vcf_path,
                                    "hetsite_depth":hetsite_depth_path,
                                    "covcollect": covcollect_path,
                                    "normal_covcollect": normal_covcollect_path,
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

    hapaseg_outputs = hapaseg.hapaseg_workflow(hetsites_file = generate_hapaseg_files_task["hapaseg_hets"],
                                   tumor_coverage_bed = generate_hapaseg_files_task["hapaseg_coverage_bed"],
                                   normal_coverage_bed = localization_task["normal_covcollect"] if normal_covcollect_path != "" else None,
                                   phased_vcf = localization_task["phased_vcf"],
                                   ref_genome_build = ref_build,
                                   run_cdp = run_cdp,
                                   target_list = target_list if isinstance(target_list, int) else localization_task["target_list"],
                                   cleanup_disks=cleanup_disks,
                                   is_ffpe=is_ffpe
                                   )
    
    hapaseg_downstream = Downstream_HapASeg_Analysis(inputs = {
                                   "sim_profile" : sim_profile,
                                   "hapaseg_seg_file": hapaseg_outputs["hapaseg_segfile"],
                                   "ground_truth_seg_file": ground_truth_seg_file,
                                   "sample_name": sample_label,
                                   "ref_fasta": localization_task["ref_fasta"],
                                   "cytoband_file": localization_task["cytoband_file"]
                        })
    
    hapaseg_downstream_unclustered = Downstream_HapASeg_Analysis(inputs = {
                                   "sim_profile" : sim_profile,
                                   "hapaseg_seg_file": hapaseg_outputs["hapaseg_skip_acdp_segfile"],
                                   "ground_truth_seg_file": ground_truth_seg_file,
                                   "sample_name": sample_label + "_unclustered",
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
              inputs={
                      "sim_profile" : sim_profile,
                      "gatk_sim_cov_input": generate_gatk_data_task["tumor_coverage_tsv"],
                      "gatk_sim_acounts" : generate_gatk_data_task["tumor_allele_counts"],
                      "gatk_seg_file" : gatk_segfile,
                      "ground_truth_seg_file": ground_truth_seg_file,
                      "sample_name": sample_label,
                      "ref_fasta": localization_task["ref_fasta"],
                      "cytoband_file": localization_task["cytoband_file"]
                    }
            )

# Facets
def Facets_Sim_Workflow(sim_profile = None,
                      purity = None,
                      sample_label = None,
                      normal_vcf_path = None,
                      variant_depth_path = "",
                      facets_allelecounts_path="", # pass facets raw counts instead of cs
                      filtered_variants_path = "", # pass tumor filtered cs to use fake normal counts
                      normal_callstats_path = "", # pass nomral cs to use real normal counts
                      unmatched_normal_callstats = False, # pass true if normal does not match vcf
                      ground_truth_seg_file=None,
                      ref_fasta=None,
                      cytoband_file=None
                ):
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "variant_depth":variant_depth_path,
                                                "filtered_variants":filtered_variants_path,
                                                "facets_allelecounts": facets_allelecounts_path,
                                                "normal_callstats": normal_callstats_path,
                                                "ref_fasta": ref_fasta,
                                                "cytoband_file": cytoband_file
                                               }
                                     ) 

    generate_facets_data_task = Generate_Facets_Sim_Data(inputs = {
                                    "sim_profile" : sim_profile,
                                    "purity": purity,
                                    "sample_label": sample_label,
                                    "normal_vcf_path":localization_task["normal_vcf"],
                                    "variant_depth_path": localization_task["variant_depth"] if variant_depth_path != "" else "",
                                    "facets_allelecounts_path": localization_task["facets_allelecounts"] if facets_allelecounts_path != "" else "",
                                    "filtered_variants_path": localization_task["filtered_variants"] if filtered_variants_path != "" else "",
                                    "normal_callstats_path": localization_task["normal_callstats"] if normal_callstats_path != "" else "",
                                    "unmatched_normal_callstats":unmatched_normal_callstats                      
              })

    run_facets_task = Facets(inputs = {"snp_counts": generate_facets_data_task["facets_input_counts"]})
    
    facets_downstream = Downstream_Facets_Analysis(
                            inputs={
                                     "sim_profile" : sim_profile,
                                     "facets_input_counts": generate_facets_data_task["facets_input_counts"],
                                     "facets_seg_file": run_facets_task["facets_seg_file"],
                                     "ground_truth_seg_file": ground_truth_seg_file,
                                     "sample_name": sample_label,
                                     "ref_fasta": localization_task["ref_fasta"],
                                     "cytoband_file": localization_task["cytoband_file"]
                                   }
                            )
                                    
                                    
# ASCAT
def ASCAT_Sim_Workflow(sim_profile=None,
                       purity=None,
                       sample_label=None,
                       normal_vcf_path=None,
                       variant_depth_path="",
                       filtered_variants_path = "", # pass tumor filtered cs to use fake normal counts
                       normal_callstats_path = "", # pass nomral cs to use real normal counts
                       unmatched_normal_callstats = False, # pass true if normal does not match vcf
                       GC_correction_file=None,
                       RT_correction_file=None,
                       ground_truth_seg_file=None,
                       ref_fasta=None,
                       cytoband_file=None
            ):
    
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "variant_depth":variant_depth_path,
                                                "filtered_variants":filtered_variants_path,
                                                "normal_callstats": normal_callstats_path,
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
                                        "filtered_variants_path": localization_task["filtered_variants"] if filtered_variants_path != "" else "",
                                        "normal_callstats_path": localization_task["normal_callstats"] if normal_callstats_path != "" else "",
                                        "unmatched_normal_callstats": unmatched_normal_callstats
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
                                inputs={
                                        "sim_profile" : sim_profile,
                                        "ascat_t_logr": generate_ascat_data_task["ascat_tumor_logR"],
                                        "ascat_t_baf": generate_ascat_data_task["ascat_tumor_BAF"],
                                        "ascat_seg_file": run_ascat_task["ascat_raw_segments"],
                                        "ground_truth_seg_file": ground_truth_seg_file,
                                        "sample_name": sample_label,
                                        "ref_fasta": localization_task["ref_fasta"],
                                        "cytoband_file": localization_task["cytoband_file"]
                                       }
                                )
    
# Hatchet
def Hatchet_Sim_Workflow(sim_profile=None,
                         purity=None,
                         sample_label=None,
                         normal_vcf_path=None,  # most be compressed VCF file
                         ref_build = None, #hg19 or hg38
                         ref_fasta=None,
                         tumor_baf_path=None,
                         thresholds_files=None,
                         int_counts_sim_file=None,
                         pos_counts_sim_file=None,
                         snp_counts_sim_file=None,
                         read_combined_file=None,
                         hatchet_phased_vcf=None, 
                         ground_truth_seg_file=None,
                         cytoband_file = None
         ):
    
    localization_task = LocalizeToDisk(files = 
                                {"normal_vcf":normal_vcf_path,
                                 "ref_fasta": ref_fasta,
                                 "thresholds_files": thresholds_files,
                                 "tumor_baf": tumor_baf_path, 
                                 "int_counts": int_counts_sim_file, 
                                 "pos_counts": pos_counts_sim_file,
                                 "snp_counts": snp_counts_sim_file,
                                 "read_combined": read_combined_file,
                                 "hatchet_phased_vcf": hatchet_phased_vcf,
                                 "cytoband_file": cytoband_file
                                })
    
    generate_hatchet_sim_task = Generate_HATCHet_Sim_Data(inputs = { 
                                        "sim_profile": sim_profile,
                                        "purity": purity,
                                        "sample_label": sample_label,
                                        "normal_vcf_path": localization_task["normal_vcf"],
                                        "tumor_baf_path": localization_task["tumor_baf"], 
                                        "int_counts_file": localization_task["int_counts"], 
                                        "pos_counts_file": localization_task["pos_counts"],
                                        "snp_counts_file": localization_task["snp_counts"],
                                        "read_combined_file": localization_task["read_combined"]
                                        })

    hatchet_workflow_results = Hatchet_Main(tumor_snp_depths = generate_hatchet_sim_task["tumor_snp_counts"],
                 count_reads_dir = localization_task["thresholds_files"],
                 total_counts_file = generate_hatchet_sim_task["total_count"],
                 reference_genome_version = ref_build,
                 phased_vcf = localization_task["hatchet_phased_vcf"] if hatchet_phased_vcf is not None else "",
                 additional_totals_files = [generate_hatchet_sim_task["totals_files"]],
                 samples_file = generate_hatchet_sim_task["samples_file"]
                )
    
    downstream_hatchet_workflow = Downstream_Hatchet_Analysis(inputs = { 
                                        "sim_profile" : sim_profile,
                                        "hatchet_seg_file": hatchet_workflow_results["hatchet_seg_file"],
                                        "hatchet_bin_file": hatchet_workflow_results["hatchet_bin_file"], 
                                        "ground_truth_seg_file": ground_truth_seg_file,
                                        "sample_name": sample_label,
                                        "ref_fasta": localization_task["ref_fasta"],
                                        "cytoband_file": localization_task["cytoband_file"]
                                        }
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
                      normal_callstats_path = "", # path to mutect callstats file for the normal sample. will be used as normal allelecounts if passed
                      unmatched_normal_callstats=False, # pass true if normal does not match vcf (e.g. ffpe with NA12878 normal)
                      # HapASeg
                      hapaseg_hetsite_depth_path=None,
                      hapaseg_covcollect_path=None,
                      hapaseg_target_list=2000,
                      hapaseg_phased_vcf_path=None, # path to cached eagle combined output
                      hapaseg_cleanup_disks=False,
                      hapaseg_normal_covcollect_path="", # optional, will be used as coviatiate if passed
                      hapaseg_run_cdp=False, # only applies to  WES, optionaly run CDP before ACDP
                      is_ffpe=False, # whether to use faire tracks
                      # GATK
                      gatk_variant_depth_path = None,
                      gatk_coverage_tsv_path = None,
                      gatk_sim_normal_allelecounts_path=None,
                      gatk_raw_gatk_allelecounts_path=None,
                      gatk_raw_gatk_coverage_path=None,
                      gatk_sequence_dictionary=None,
                      gatk_count_panel="",
                      gatk_annotated_intervals="",
                      # facets
                      facets_variant_depth_path = None,
                      facets_filtered_variants_path = "", # will be used to generate fake normal allelecounts if normal_cs not passed
                      # ascat
                      ascat_variant_depth_path=None,
                      ascat_filtered_variants_path="", # will be used to generate fake normal allelecounts if normal_cs not passed
                      ascat_GC_correction_file=None,
                      ascat_RT_correction_file=None,
                      # hatchet
                      hatchet_tumor_baf = None,
                      hatchet_thresholds_files_dir = None,
                      hatchet_int_counts_file = None,
                      hatchet_pos_counts_file = None,
                      hatchet_snp_counts_file = None,
                      hatchet_read_combined_file = None,
                      hatchet_phased_vcf = None 
                      ):
    # important to save space
    gt_localization_task = LocalizeToDisk(files = {
                            "sim_profile":sim_profile,
                            "normal_vcf_path":normal_vcf_path,
                            "hapaseg_hetsite_depth_path": hapaseg_hetsite_depth_path,
                            "hapaseg_covcollect_path":hapaseg_covcollect_path
                            }
                        )


    seg_file_gen_task = Generate_Groundtruth_Segfile(inputs= {
                            "sample_label": sample_label,
                            "purity": ground_truth_purity if ground_truth_purity is not None else purity,
                            "sim_profile": gt_localization_task["sim_profile"],
                            "normal_vcf_path": gt_localization_task["normal_vcf_path"],
                            "hapaseg_hetsite_depth_path": gt_localization_task["hapaseg_hetsite_depth_path"],
                            "hapaseg_coverage_tsv": gt_localization_task["hapaseg_covcollect_path"]
                        }
                    ) 

    HapASeg_Sim_Workflow(sim_profile = sim_profile,
                        purity = purity,
                        sample_label = sample_label,
                        normal_vcf_path = normal_vcf_path,
                        hetsite_depth_path = hapaseg_hetsite_depth_path,
                        covcollect_path = hapaseg_covcollect_path,
                        normal_covcollect_path = hapaseg_normal_covcollect_path,
                        phased_vcf_path = hapaseg_phased_vcf_path,
                        ref_build=ref_build,
                        ref_fasta=ref_fasta,
                        cytoband_file=cytoband_file,
                        ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"],
                        target_list = hapaseg_target_list,
                        run_cdp = hapaseg_run_cdp,
                        cleanup_disks = hapaseg_cleanup_disks,
                        is_ffpe = is_ffpe
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
                        normal_callstats_path = normal_callstats_path,
                        unmatched_normal_callstats = unmatched_normal_callstats,
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
                       normal_callstats_path = normal_callstats_path,
                       unmatched_normal_callstats = unmatched_normal_callstats,
                       GC_correction_file = ascat_GC_correction_file,
                       RT_correction_file = ascat_RT_correction_file,
                       ref_fasta=ref_fasta,
                       cytoband_file=cytoband_file,
                       ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"]
                      )

    Hatchet_Sim_Workflow(sim_profile=sim_profile,
                         purity=purity,
                         sample_label=sample_label,
                         normal_vcf_path=normal_vcf_path,
                         ref_build = ref_build,
                         ref_fasta = ref_fasta,
                         tumor_baf_path = hatchet_tumor_baf,
                         thresholds_files = hatchet_thresholds_files_dir,
                         int_counts_sim_file=hatchet_int_counts_file,
                         pos_counts_sim_file=hatchet_pos_counts_file,
                         snp_counts_sim_file=hatchet_snp_counts_file,
                         read_combined_file=hatchet_read_combined_file,
                         hatchet_phased_vcf=hatchet_phased_vcf, 
                         cytoband_file=cytoband_file,
                         ground_truth_seg_file=seg_file_gen_task["ground_truth_seg_file"],
            )
