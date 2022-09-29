import wolf
from sim_file_generation_tasks import *
from wolf.localization import LocalizeToDisk
from gatk_wolf_task.gatk_wolf_task import GATK_CNV_Workflow

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
                        "normal_vcf_path": localization_task["normal_vcf_path"],
                        "variant_depth_path": localization_task["variant_depth_path"],
                        "coverage_tsv_path": localization_task["coverage_tsv_path"],
                        "sim_normal_allelecounts_path": localization_task["sim_normal_allelecounts_path"],
                        "raw_gatk_allelecounts_path":localization_task["raw_gatk_allelecounts_path"],
                        "raw_gatk_coverage_path":localization_task["raw_gatk_coverage_path"]
                       }
                )    
    GATK_CNV_Workflow(sample_name = sample_label + f'_{purity}',
                      tumor_frag_counts_hdf5 = generate_gatk_data_task["tumor_frag_counts"],
                      tumor_allele_counts = generate_gatk_data_task["tumor_allelecounts"],
                      count_panel = localization_task["count_panel"] if count_panel != "" else "",
                      annotated_intervals=localization_task["annotated_intervals"] if annotated_intervals != "" else "",
                      sequence_dictionary = localization_task["seq_dict"]
                     )
