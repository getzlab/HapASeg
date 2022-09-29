import wolf

class GATK_CNV_denoise(wolf.Task):
    inputs = {"tumor_frag_counts_hdf5": None,
              "sample_name": None,
              "annotated_intervals": "",
              "count_panel" : ""}
    
    def script(self):
        script = """
        gatk --java-options "-Xmx12g" DenoiseReadCounts -I ${tumor_frag_counts_hdf5}\
        --standardized-copy-ratios ${sample_name}_gatk_standardizedCR.tsv\
         --denoised-copy-ratios ${sample_name}_gatk_denoisedCR.tsv\
        """
        
        if self.conf["inputs"]["annotated_intervals"] != "":
            script += "--annotated-intervals ${annotated_intervals}"
        elif self.conf["inputs"]["count_panel"] != "":
            script += "--count-panel-of-normals ${count_panel}"
        return script
 
    output_patterns = {
    "std_copy_ratios": "*_gatk_standardizedCR.tsv",
    "denoised_copy_ratios": "*_gatk_denoisedCR.tsv"
    }
    
    resources = {"cpus-per-task": 4, "mem" : "10G"}
    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_CNV_model_segments(wolf.Task):

    inputs = {"denoised_copy_ratios": None,
              "tumor_allele_counts": None,
              "sample_name":None
             }
    
    script = """
    gatk --java-options "-Xmx4g" ModelSegments --denoised-copy-ratios ${denoised_copy_ratios}\
    --allelic-counts ${tumor_allele_counts}\
    --output . --output-prefix ${sample_name}
    """
    output_patterns = {
    'model_segments_pre_smoothing':'*.modelBegin.seg',
    'model_segments_post_smoothing': '*.modelFinal.seg',
    'allele_frac_params_pre_smoothing': '*.modelBegin.af.param',
    'allele_frac_params_post_smoothing': '*.modelFinal.af.param',
    'copy_ratio_params_pre_smoothing': '*.modelBegin.cr.param',
    'copy_ratio_params_post_smoothing': '*.modelFinal.cr.param',
    'copy_ratio_segments': '*.cr.seg', # same as model segments post smoothing but in dif format
    'igv_copy_ratio_segs' : '*.cr.igv.seg',
    'igv_af_segs': '*.af.igv.seg',
    'called_hets': '*.hets.tsv'
    }

    resources = {"cpus-per-task": 4, "mem" : "10G"}
    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_CNV_call_cr_segs(wolf.Task):
    inputs = {"sample_name":None,
              "input_cr_seg":None
             }

    script = """
    gatk CallCopyRatioSegments --input ${input_cr_seg} --output ${sample_name}_cr.called.seg
    """
    
    output_patterns = {"called_cr_segs": "*_cr.called.seg"}
    resources = {"cpus-per-task": 1, "mem" : "4G"}
    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_CNV_plot_model_segs(wolf.Task):
    inputs = {"denoised_copy_ratios":None,
              "tumor_het_counts": None,
              "segments" : None,
              "seq_dictionary": None,
              "minimum_contig_length":46709983,
              "sample_name":None}

    script = """
    gatk PlotModeledSegments --denoised-copy-ratios ${denoised_copy_ratios}\
    --allelic-counts ${tumor_het_counts} --segments ${segments}\
    --sequence-dictionary ${seq_dictionary} --minimum-contig-length ${minimum_contig_length}\
    --output ./ --output-prefix ${sample_name}    
    """

    output_patterns = {"segments_plot":"*.denoised.png"}
    resources = {"cpus-per-task": 1, "mem" : "4G"}
    docker = "broadinstitute/gatk:4.0.1.1"

def GATK_CNV_Workflow(sample_name = None,
                      tumor_frag_counts_hdf5 = None,
                      annotated_intervals = "",
                      count_panel = "",
                      tumor_allele_counts = None,
                      sequence_dictionary = None
                      ):
    
    if count_panel != "" and annotated_intervals != "":
        raise ValueException("only one of count_panel and annotated_intervals expected, got both!")
    gatk_denoise_task = GATK_CNV_denoise(inputs = {"tumor_frag_counts_hdf5":tumor_frag_counts_hdf5,
                               "annotated_intervals":annotated_intervals,
                               "count_panel": count_panel,
                               "sample_name": sample_name}
                    )

    gatk_model_segments_task = GATK_CNV_model_segments(inputs = {"denoised_copy_ratios":gatk_denoise_task["denoised_copy_ratios"],
                                      "tumor_allele_counts":tumor_allele_counts,
                                      "sample_name": sample_name}
                           )
    gatk_call_cr_task = GATK_CNV_call_cr_segs(inputs={"sample_name" : sample_name,
                                              "input_cr_seg" : gatk_model_segments_task["copy_ratio_segments"]})
    
    gatk_plot_segs_task = GATK_CNV_plot_model_segs(inputs = {"denoised_copy_ratios":gatk_denoise_task["denoised_copy_ratios"],
                                                         "tumor_het_counts":gatk_model_segments_task["called_hets"],
                                                         "segments": gatk_model_segments_task["model_segments_post_smoothing"],
                                                         "seq_dictionary":sequence_dictionary,
                                                         "sample_name":sample_name})
