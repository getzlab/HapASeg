import wolf
import sys

from sim_file_generation_tasks import *
from output_comparison_tasks import *
from wolf.localization import LocalizeToDisk
from gatk_wolf_task.gatk_wolf_task import GATK_CNV_Workflow
from ascat_wolf_task.ascat_wolf_task import ASCAT
from facets_wolf_task.facets_wolf_task import Facets

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
                         target_list=2000
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
                                   target_list = target_list if isinstance(target_list, int) else localization_task["target_list"]
                                   )
    
    hapaseg_downstream = Downstream_HapASeg_Analysis(inputs = {
                                   "hapaseg_seg_file": hapaseg_seg_file,
                                   "ground_truth_seg_file": ground_truth_seg_file,
                                   "sample_name": f"{sample_label}_{purity}",
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
    gatk_segfile = GATK_CNV_Workflow(sample_name = sample_label + f'_{purity}',
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
                      "sample_name": f"{sample_label}_{purity}",
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
                                     "sample_name": f"{sample_label}_{purity}",
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
                                        "sample_name": f"{sample_label}_{purity}",
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
                      ground_truth_seg_file=None, 
                      hapaseg_hetsite_depth_path=None,
                      hapaseg_covcollect_path=None,
                      hapaseg_target_list=2000,
                      hapaseg_phased_vcf_path=None, # path to cached eagle combined output
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
                      ascat_RT_correction_file=None
                      ):
    
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
                        ground_truth_seg_file=ground_truth_seg_file,
                        target_list = hapaseg_target_list
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
                      ground_truth_seg_file=ground_truth_seg_file
                      )

    Facets_Sim_Workflow(sim_profile=sim_profile,
                        purity=purity,
                        sample_label=sample_label,
                        normal_vcf_path = normal_vcf_path,
                        variant_depth_path = facets_variant_depth_path,
                        filtered_variants_path = facets_filtered_variants_path,
                        ref_fasta=ref_fasta,
                        cytoband_file=cytoband_file,
                        ground_truth_seg_file=ground_truth_seg_file
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
                       ground_truth_seg_file=ground_truth_seg_file
                      )

