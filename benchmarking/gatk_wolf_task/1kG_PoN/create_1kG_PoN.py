import os
import wolf
import pandas as pd

from wolf.localization import LocalizeToDisk

class GATK_Collect_Frag(wolf.Task):
    inputs = {'cram_file' : None,
              'crai_file' : None,
              'ref_genome': None,
              'interval_list': None,
              'sample_name': None 
             }

    script = """
    gatk CollectFragmentCounts -I ${cram_file} -L ${interval_list} --interval-merging-rule OVERLAPPING_ONLY -R ${ref_genome} -O "${PWD}/${sample_name}_CollectedFragCounts.hdf5" --exclude-intervals chrX --exclude-intervals chrY
    """
    
    output_patterns = {'sample_counts' : '*.hdf5'}

    docker = 'broadinstitute/gatk:4.0.1.1'

    resources = {'cpus-per-task': 4, 'mem': '25G'}

class GATK_create_PoN(wolf.Task):
    inputs = {'count_files': None,
              'annotation_file': None,
              'sample_name': None}
    
    script = """\
        input_str=$(cat ${count_files}  | tr '\\n' ' ' | sed 's/\\s*$//g' | sed  's/ / -I /g')
        gatk --java-options '-Xmx10g' CreateReadCountPanelOfNormals -I ${input_str} \
        --minimum-interval-median-percentile 5.0 -O ${sample_name}.hdf5 \
        --annotated-intervals ${annotation_file} \
    """
    
    output_patterns = {'PoN_hdf' : '*.hdf5'}
    
    docker = 'broadinstitute/gatk:4.0.1.1'

    resources = {"cpus-per-task": 4, 'mem': '12G'}

class GATK_AnnotateIntervals(wolf.Task):
    inputs = {"ref_fasta": None,
              "ref_fasta_idx":None,
              "ref_fasta_dict":None,
              "interval_list":None,
              "interval_name":None,
             }

    script = """
        gatk AnnotateIntervals -R ${ref_fasta} -L ${interval_list} \
        --interval-merging-rule OVERLAPPING_ONLY -O ${interval_name}_gatk.annotated_intervals.tsv \
        --exclude-intervals chrX --exclude-intervals chrY
        """

    output_patterns = {"gatk_annotated_intervals":"*_gatk.annotated_intervals.tsv"}
    
    docker = "broadinstitute/gatk:4.0.1.1"


def gatk_make_pon_workflow(cram_files = None,
                           crai_files = None,
                           ref_genome = None, # must match cram, taken from gatk bucket
                           ref_index = None,
                           ref_dict = None,
                           sample_names = None,
                           pon_name = None,
                           annotation_file = None,
                           interval_list = None # taken from gatk bucket
    ):
    
    ref_localization = LocalizeToDisk(files = dict(ref_genome = ref_genome,
                                                   ref_index = ref_index,
                                                   ref_dict = ref_dict,
                                                   interval_list = interval_list))
    
    fragcount_task = GATK_Collect_Frag( inputs = {'cram_file' : cram_files,
                                                'crai_file': crai_files,
                                                'ref_genome': ref_localization['ref_genome'],
                                                'ref_index': ref_localization['ref_index'],
                                                'ref_dict': ref_localization['ref_dict'],
                                                'sample_name' : sample_names,
                                                'interval_list': ref_localization['interval_list']
                                        }, preemptible = False)
    
    if annotation_file is None:
        # generate_annotation_file
        annot_task = GATK_AnnotateIntervals(inputs = {'ref_fasta' : ref_localization['ref_genome'],
                                                      'ref_fasta_idx' : ref_localization['ref_index'],
                                                      'ref_fasta_dict' : ref_localization['ref_dict'],
                                                      'interval_list' : ref_localization['interval_list'],
                                                      'interval_name' : os.path.basename(interval_list).rstrip('interval_list')
                                                     })
        annotation_file = annot_task['gatk_annotated_intervals']
     
    gatk_pon = GATK_create_PoN(inputs = {'count_files' : [fragcount_task['sample_counts']],
                                         'annotation_file' : annotation_file,
                                         'sample_name': pon_name})
    
