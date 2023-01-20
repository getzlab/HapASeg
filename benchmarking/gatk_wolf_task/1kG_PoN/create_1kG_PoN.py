import os
import wolf
import pandas as pd

from wolf.localization import LocalizeToDisk
import sys
sys.path.append('../')
from gatk_generate_raw_task import GATK_Preprocess_Intervals, Filter_GATK_Interval_List, GATK_CollectFragmentCounts, GATK_AnnotateIntervals

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

def gatk_make_pon_workflow(cram_files = None,
                           crai_files = None,
                           ref_genome = None, # must match cram, taken from gatk bucket
                           ref_index = None,
                           ref_dict = None,
                           sample_names = None,
                           bin_length=1000, # should be 0 for WES
                           padding=0, # should be 250 for WES
                           exclude_sex=True,
                           pon_name = None,
                           annotation_file = None,
                           interval_list = None, # taken from gatk bucket
                           exclude_chroms = "" # contigs to exclude (other than sex chromosomes, if exclude_sex)
    ):
    
    if exclude_sex:
        exclude_chroms += " chrX chrY"
    
    if exclude_chroms != "":
        filter_intv_list_task = Filter_GATK_Interval_List(inputs = {
                                            "interval_list": interval_list,
                                            "chrs_to_exclude": exclude_chroms
                                            }
                                            )

    ref_localization = LocalizeToDisk(files = dict(ref_genome = ref_genome,
                                                   ref_index = ref_index,
                                                   ref_dict = ref_dict,
                                                   )
                                                )

    preprocess_intervals_task = GATK_Preprocess_Intervals(inputs = {
                                  "interval_list":interval_list if exclude_chroms == "" else filter_intv_list_task["interval_list"],
                                  "ref_fasta": ref_localization["ref_genome"],
                                  "ref_fasta_idx": ref_localization["ref_index"],
                                  "ref_fasta_dict": ref_localization["ref_dict"],
                                  "bin_length": bin_length,
                                  "padding": padding,
                                  "interval_name": os.path.basename(interval_list).rstrip('interval_list')
                                 })
    
    fragcount_task =  GATK_CollectFragmentCounts( inputs = {'input_bam' : cram_files,
                                                'input_bai': crai_files,
                                                'ref_fasta': ref_localization['ref_genome'],
                                                'ref_fasta_idx': ref_localization['ref_index'],
                                                'ref_fasta_dict': ref_localization['ref_dict'],
                                                'sample_name' : sample_names,
                                                'interval_list': preprocess_intervals_task['gatk_interval_list'],
                                                'exclude_sex':exclude_sex
                                        }, preemptible = False)
    
    if annotation_file is None:
        # generate_annotation_file
        annot_task = GATK_AnnotateIntervals(inputs = {'ref_fasta' : ref_localization['ref_genome'],
                                                      'ref_fasta_idx' : ref_localization['ref_index'],
                                                      'ref_fasta_dict' : ref_localization['ref_dict'],
                                                      'interval_list' : preprocess_intervals_task['gatk_interval_list'],
                                                      'interval_name' : os.path.basename(interval_list).rstrip('interval_list')
                                                     })
        annotation_file = annot_task['gatk_annotated_intervals']
     
    gatk_pon = GATK_create_PoN(inputs = {'count_files' : [fragcount_task['frag_counts_hdf']],
                                         'annotation_file' : annotation_file,
                                         'sample_name': pon_name})
    
