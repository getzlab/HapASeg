import wolf
import sys

from sim_file_generation_tasks import *
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
                         hetsite_depth_path=None,
                         covcollect_path=None,
                         genotype_file=None, # normal genotype of NA12878
                         ref_build=None,
                         target_list=2000
                ):
    localization_task = LocalizeToDisk(files = {
                                    "normal_vcf":normal_vcf_path,
                                    "hetsite_depth":hetsite_depth_path,
                                    "covcollect": covcollect_path,
                                    "genotype_file":genotype_file,
                                    "target_list" : target_list if not isinstance(target_list, int) else ""
                                    })
    generate_hapaseg_files_task = Generate_HapASeg_Sim_Data( inputs = {
                                    "sim_profile": sim_profile,
                                    "purity":purity,
                                    "sample_label":sample_label,
                                    "normal_vcf_path":localization_task["normal_vcf"],
                                    "hetsite_depth_path": localization_task["hetsite_depth"],
                                    "covcollect_path":localization_task["covcollect"]
                                    })

    run_hapaseg = HapASeg_Workflow(hetsites_file = generate_hapaseg_files_task["hapaseg_hets"],
                                   tumor_coverage_bed = generate_hapaseg_files_task["hapaseg_coverage_bed"],
                                   genotype_file = localization_task["genotype_file"],
                                   ref_genome_build = ref_build,
                                   target_list = target_list if isinstance(target_list, int) else localization_task["target_list"]
                                   )
                         
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
                      annotated_intervals=""
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
                            "annotated_intervals":annotated_intervals
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
    GATK_CNV_Workflow(sample_name = sample_label + f'_{purity}',
                      tumor_frag_counts_hdf5 = generate_gatk_data_task["tumor_frag_counts"],
                      tumor_allele_counts = generate_gatk_data_task["tumor_allele_counts"],
                      count_panel = localization_task["count_panel"] if count_panel != "" else "",
                      annotated_intervals=localization_task["annotated_intervals"] if annotated_intervals != "" else "",
                      sequence_dictionary = localization_task["seq_dict"]
                     )

# Facets
def Facets_Sim_Workflow(sim_profile = None,
                      purity = None,
                      sample_label = None,
                      normal_vcf_path = None,
                      variant_depth_path = None,
                      filtered_variants_path = None,
                ):
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "variant_depth":variant_depth_path,
                                                "filtered_variants":filtered_variants_path
                                                }) 

    generate_facets_data_task = Generate_Facets_Sim_Data(inputs = {
                                    "sim_profile" : sim_profile,
                                    "purity": purity,
                                    "sample_label": sample_label,
                                    "normal_vcf_path":localization_task["normal_vcf"],
                                    "variant_depth_path": localization_task["variant_depth"],
                                    "filtered_variants_path": localization_task["filtered_variants"]
                                    })

    run_facets_task = Facets(inputs = {"snp_counts": generate_facets_data_task["facets_input_counts"]})

# ASCAT
def ASCAT_Sim_Workflow(sim_profile=None,
                       purity=None,
                       sample_label=None,
                       normal_vcf_path=None,
                       variant_depth_path=None,
                       filtered_variants_path=None,
                       GC_correction_file=None,
                       RT_correction_file=None
            ):
    
    localization_task = LocalizeToDisk(files = {"normal_vcf":normal_vcf_path,
                                                "variant_depth":variant_depth_path,
                                                "filtered_variants":filtered_variants_path,
                                                "GC_correction_file":GC_correction_file,
                                                "RT_correction_file": RT_correction_file
                                })
    
    generate_ascat_data_task = Generate_ASCAT_Sim_Data(inputs = { 
                                        "sim_profile": sim_profile,
                                        "purity": purity,
                                        "sample_label": sample_label,
                                        "normal_vcf_path":localization_task["normal_vcf"],
                                        "variant_depth_path": localization_task["variant_depth"],
                                        "filtered_variants_path": localization_task["filtered_variants"]
                                        })

    run_ascat_task = ASCAT( inputs = {"tumor_LogR" : generate_ascat_data_task["ascat_tumor_logR"],
                                      "tumor_BAF" : generate_ascat_data_task["ascat_tumor_BAF"],
                                      "normal_LogR" : generate_ascat_data_task["ascat_normal_logR"],
                                      "normal_BAF" : generate_ascat_data_task["ascat_tumor_BAF"],
                                      "GC_correction_file": localization_task["GC_correction_file"],
                                      "RT_correction_file": localization_task["RT_correction_file"]
                            })

# run all pipelines

def Run_Sim_Workflows(sim_profile=None,
                      purity=None,
                      sample_label=None,
                      normal_vcf_path=None,
                      ref_build=None,
                      hapaseg_hetsite_depth_path=None,
                      hapaseg_covcollect_path=None,
                      hapaseg_genotype_file=None, # normal genotype of NA12878
                      hapaseg_target_list=2000,
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
                        genotype_file = hapaseg_genotype_file,
                        ref_build=ref_build,
                        target_list = hapaseg_target_list)

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
                      )

    Facets_Sim_Workflow(sim_profile=sim_profile,
                        purity=purity,
                        sample_label=sample_label,
                        normal_vcf_path = normal_vcf_path,
                        variant_depth_path = facets_variant_depth_path,
                        filtered_variants_path = facets_filtered_variants_path
                        )
    
    ASCAT_Sim_Workflow(sim_profile = sim_profile,
                       purity = purity,
                       sample_label = sample_label,
                       normal_vcf_path = normal_vcf_path,
                       variant_depth_path = ascat_variant_depth_path,
                       filtered_variants_path = ascat_filtered_variants_path,
                       GC_correction_file = ascat_GC_correction_file,
                       RT_correction_file = ascat_RT_correction_file
                      )

