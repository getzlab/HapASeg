import wolf
from wolf import LocalizeToDisk, UploadToBucket
import prefect
import h5py
import re
import subprocess

class GATK_Preprocess_Intervals(wolf.Task):
    inputs = {"interval_list":None,
              "ref_fasta":None,
              "ref_fasta_idx":None,
              "ref_fasta_dict":None,
              "bin_length":1000,
              "interval_name":None
             }
    
    script = """
    gatk PreprocessIntervals -L ${interval_list} -R ${ref_fasta} --bin-length ${bin_length}\
    --interval-merging-rule OVERLAPPING_ONLY -O ${interval_name}.gatk.interval_list
    """
    
    output_patterns = {"gatk_interval_list": "*.gatk.interval_list"}

    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_CollectAlleleCounts(wolf.Task):
    inputs = {"vcf_file": None,
              "input_bam": None,
              "input_bai":None,
              "ref_fasta":None,
              "ref_fasta_idx":None,
              "ref_fasta_dict":None,
              "sample_name":None,
              "exclude_sex":False
             }

    def script(self):
        script = """
        gatk CollectAllelicCounts -L ${vcf_file} -I ${input_bam} -R ${ref_fasta} \
        -O ${sample_name}_gatk.allelecounts.tsv"""
 
        if self.conf["inputs"]["exclude_sex"]:
            script += " --exclude-intervals chrX --exclude-intervals chrY"
        
        return script

    output_patterns = {"allele_counts_tsv" : "*_gatk.allelecounts.tsv"}
    
    resources = {"cpus-per-task":4, "mem":"6G"}

    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_CollectFragmentCounts(wolf.Task):
    inputs = {"input_bam": None,
              "input_bai": None,
              "interval_list": None,
              "sample_name": None,
              "exclude_sex": False
             }

    def script(self):
        script = """
        gatk CollectFragmentCounts -I ${input_bam} -L ${interval_list} \
        --interval-merging-rule OVERLAPPING_ONLY\
        -O ${sample_name}_gatk.frag.counts.hdf5"""

        if self.conf["inputs"]["exclude_sex"]:
            script += " --exclude-intervals chrX --exclude-intervals chrY"
        
        return script
    
    output_patterns = {"frag_counts_hdf" : "*_gatk.frag.counts.hdf5"}
    
    resources = {"cpus-per-task":4, "mem":"4G"}

    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_AnnotateIntervals(wolf.Task):
    inputs = {"ref_fasta": None,
              "ref_fasta_idx":None,
              "ref_fasta_dict":None,
              "interval_list":None,
              "interval_name":None,
             }

    script = """
        gatk AnnotateIntervals -R ${ref_fasta} -L ${interval_list} \
        --interval-merging-rule OVERLAPPING_ONLY -O ${interval_name}_gatk.annotated_intervals.tsv
        """

    output_patterns = {"gatk_annotated_intervals":"*_gatk.annotated_intervals.tsv"}
    
    docker = "broadinstitute/gatk:4.0.1.1"


class GATK_DenoiseReadCounts(wolf.Task):
    inputs = {"frag_count_hdf5": None,
              "sample_name":None,
              "panel_of_normals": "",
              "annotated_intervals":""
             }
    
    def script(self):
        script = """
        gatk --java-options "-Xmx4g" DenoiseReadCounts -I ${frag_count_hdf5}\
        --standardized-copy-ratios ${sample_name}_standardizedCR.tsv\
        --denoised-copy-ratios ${sample_name}_denoisedCR.tsv"""

        if self.conf["inputs"]["panel_of_normals"] != "":
            script += " --count-panel-of-normals ${panel_of_normals}"
        elif self.conf["inputs"]["annotated_intervals"] != "":
            script += " --annotated-intervals ${annotated_intervals}"
        return script

    output_patterns = {"std_copy_ratios": "*_standardizedCR.tsv",
                       "denoised_copy_ratios": "*_denoisedCR.tsv"
                      }

    resources = {"mem" : "4G"}

    docker = "broadinstitute/gatk:4.0.1.1"

class GATK_Preprocess_Data(wolf.Task):
    inputs = {
               "frag_counts": None,
               "allele_counts": None,
               "sample_name": None
             }

    script = """
    preprocess_raw_data.py --sample_name ${sample_name}\
    --outdir ./ gatk --frag_counts ${frag_counts} --allele_counts ${allele_counts}
    """
    output_patterns = {
                        "gatk_cov_counts": "*_gatk_cov_counts.tsv",
                        "gatk_sim_normal_cov_counts": "*_gatk_sim_normal_frag.counts.hdf5",
                        "gatk_var_depth" : "*_gatk_var_depth.tsv",
                        "gatk_sim_normal_allele_counts" : "*_gatk_sim_normal_allele_counts.tsv"
                      }
    resources = {"mem": "12G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:coverage_mcmc_integration_lnp_jh_v623"

def GATK_Generate_Raw_Data(input_bam=None,
                           input_bai=None,
                           vcf_file=None,
                           ref_fasta=None,
                           ref_fasta_idx=None,
                           ref_fasta_dict=None,
                           interval_list=None,
                           interval_name=None,
                           sample_name=None,
                           bin_length=1000,
                           exclude_sex=False,
                           panel_of_normals=None,
                           upload_bucket=None,
                           persistent_dry_run = False # skip localization of files
                          ):
   
    bam_localization_task = LocalizeToDisk(files = {"bam" : input_bam,
                                                    "bai" : input_bai
                                                    },
                                           persistent_disk_dry_run = persistent_dry_run
                                           )
 
    localization_task = LocalizeToDisk(files = {
                                "vcf_file": vcf_file,
                                "ref_fasta": ref_fasta,
                                "ref_fasta_idx": ref_fasta_idx,
                                "ref_fasta_dict": ref_fasta_dict,
                                "panel_of_normals": panel_of_normals if panel_of_normals is not None else ""
                                }
                            )

    preprocess_intervals_task = GATK_Preprocess_Intervals(inputs = {
                                  "interval_list":interval_list,
                                  "ref_fasta":localization_task["ref_fasta"],
                                  "ref_fasta_idx":localization_task["ref_fasta_idx"],
                                  "ref_fasta_dict":localization_task["ref_fasta_dict"],
                                  "bin_length":1000,
                                  "interval_name":interval_name
                                 })

    
    collect_acounts_task = GATK_CollectAlleleCounts(inputs = {
                                  "vcf_file": localization_task["vcf_file"],
                                  "input_bam": bam_localization_task["bam"],
                                  "input_bai": bam_localization_task["bai"],
                                  "ref_fasta": localization_task["ref_fasta"],
                                  "ref_fasta_idx": localization_task["ref_fasta_idx"],
                                  "ref_fasta_dict": localization_task["ref_fasta_dict"],
                                  "sample_name":sample_name,
                                  "exclude_sex":exclude_sex
                                  }
                            )

    collect_fragcounts_task = GATK_CollectFragmentCounts(inputs = {
                                  "input_bam": bam_localization_task["bam"],
                                  "input_bai": bam_localization_task["bai"],
                                  "interval_list": preprocess_intervals_task["gatk_interval_list"],
                                  "sample_name": sample_name,
                                  "exclude_sex": exclude_sex
                                 }
                            )

    interval_annotation_task = GATK_AnnotateIntervals(inputs = {
                                  "ref_fasta": localization_task["ref_fasta"],
                                  "ref_fasta_idx": localization_task["ref_fasta_idx"],
                                  "ref_fasta_dict": localization_task["ref_fasta_dict"],
                                  "interval_list": preprocess_intervals_task["gatk_interval_list"],
                                  "interval_name":interval_name,
                                 }
                            )

    # shim task for fixing annotation headers
    @prefect.task
    def fix_seq_headers(hdf_path, annot_interval_path, exclude_sex):
        f = h5py.File(hdf_path, 'r')
        dict_line = f['locatable_metadata/sequence_dictionary'][:][0].decode().split('\n')[1]
        search_res = re.search(r".*\tUR:(.*?)\t.*$", dict_line)
        if search_res is not None:
            # if ur tag exists in the fragcounts meta replace it with the tag from the intervals
            ur = search_res.groups()[0]
            with open(annot_interval_path, 'r') as f:
                lines = f.readlines()

            newlines = [re.sub(r'UR:.*?\t', 'UR:' + ur + '\t', l) for l in lines] 
        
            outpath = annot_interval_path[:-4] + 'reformatted.tsv'
            with open(outpath, 'w') as f:
                for l in newlines:
                    f.write(l)
        else:
            outpath = annot_interval_path    
    
        if exclude_sex:
            # remove sex chromosomes from annotated intervals 
            # since GATK will otherwise throw error
            autosomal_outpath = outpath[:-4] + '_no_sex.tsv'
            subprocess.Popen(f"cat {outpath} | grep -v '^chrX' | grep -v 'chrY' > {autosomal_outpath}", shell=True)
            if search_res is not None:
                # remove file with sex chromosomes
                 subprocess.Popen(f"rm -f {outpath}", shell=True)
            outpath = autosomal_outpath
    
        return outpath

    reformatted_dict = fix_seq_headers(collect_fragcounts_task["frag_counts_hdf"], interval_annotation_task["gatk_annotated_intervals"], exclude_sex)

    # process the raw data counts into formats the simulator can interpret
    preprocess_raw_task = GATK_Preprocess_Data( inputs = {"frag_counts": collect_fragcounts_task["frag_counts_hdf"],
                                                          "allele_counts": collect_acounts_task["allele_counts_tsv"],
                                                          "sample_name": sample_name
                                                         }
                                              )

    if upload_bucket is not None:
        upload_task = UploadToBucket(files = [collect_acounts_task["allele_counts_tsv"],
                                              collect_fragcounts_task["frag_counts_hdf"],
                                              reformatted_dict,
                                              preprocess_raw_task["gatk_cov_counts"],
                                              preprocess_raw_task["gatk_sim_normal_cov_counts"],
                                              preprocess_raw_task["gatk_var_depth"],
                                              preprocess_raw_task["gatk_sim_normal_allele_counts"]],
                                     bucket = upload_bucket
                                    )

