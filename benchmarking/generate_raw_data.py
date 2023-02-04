import sys
import wolf
import pandas as pd
import glob
import numpy as np
import os
import pickle
import prefect
import subprocess
import tempfile

from wolf.localization import LocalizeToDisk, DeleteDisk, UploadToBucket

sys.path.append('../wolF/')
from workflow import _hg38_config_gen, _hg19_config_gen

# import external tasks

split_intervals = wolf.ImportTask(
  task_path = "git@github.com:getzlab/split_intervals_TOOL.git",
  task_name = "split_intervals",
  commit = "dc102d8"
)

cov_collect = wolf.ImportTask(
  task_path = "git@github.com:getzlab/covcollect.git",
  task_name = "covcollect"
)

mutect = wolf.ImportTask(
  task_path = "git@github.com:getzlab/MuTect1_TOOL.git",
  task_name =  "M1")

phasing = wolf.ImportTask(
  task_path = "git@github.com:getzlab/phasing_TOOL.git",
  task_name = "phasing"
)

het_pulldown = wolf.ImportTask(
  task_path = 'git@github.com:getzlab/het_pulldown_from_callstats_TOOL.git',
  task_name = "het_pulldown"
)

# define preprocessing tasks that rely on covcollect and mutect callstats
class HapASeg_Preprocess_Callstats(wolf.Task):
    inputs = {
              "callstats": None,
              "sample_name":None,
              "dummy_normal":False
             }
    def script(self):
    
        script = """
        preprocess_raw_data.py --sample_name ${sample_name}\
        --outdir ./ hapaseg --callstats ${callstats}"""
        if self.conf["inputs"]["dummy_normal"]:
            script += " --dummy_normal"
        return script

    output_patterns = {"hapaseg_hetsite_depths": "*depth.tsv",
                       "hapaseg_filtered_cs": "*_filtered.tsv", 
                       "hapaseg_genotype": "*_genotype.tsv"
                      }
    resources = {"mem": "12G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1138"

class Facets_Preprocess_Callstats(wolf.Task):
    inputs = {
              "callstats": None,
              "sample_name":None,
              "db_snp_vcf":None,
             }
    script = """
    preprocess_raw_data.py --sample_name ${sample_name}\
    --outdir ./ facets --callstats ${callstats} --db_snp_vcf ${db_snp_vcf}
    """
    output_patterns = {
                        "facets_variant_depths": "*variant_depths.tsv",
                        "facets_filtered_variants": "*filtered.tsv"
                      }
    resources = {"mem": "12G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1138"

class ASCAT_Preprocess_Callstats(wolf.Task):
    inputs = {
               "callstats": None,
               "sample_name": None,
               "ascat_loci_list": None
             }
    script = """
    preprocess_raw_data.py --sample_name ${sample_name}\
    --outdir ./ ascat --callstats ${callstats} --ascat_loci_list ${ascat_loci_list}
    """
    output_patterns = {
                        "ascat_variant_depths": "*variant_depths.tsv",
                        "ascat_filtered_variants": "*filtered.tsv"
                      }
    resources = {"mem": "12G"}
    docker = "gcr.io/broad-getzlab-workflows/hapaseg:v1138"

###########
#
# workflows for generating all data necessary for benchmarking
#
###########

def Generate_Hapaseg_Raw_Data_Workflow(
             sample_name = None,
             bam = None,
             bai = None,
             ref_genome_build=None, #must be hg19 or hg38
             vcf = None,
             target_list = None,
             wgs=True,
             preprocess=True, # whether or not to do phasing and preprocessing tasks
             db_snp_vcf=None, # for facets variant filtering
             ascat_loci_list=None, # for ascat variant filtering
             dummy_normal=False, # pass true if vcf does not matach bam
             upload_bucket_gs_path=None,
             persistent_dry_run=False,
             chrs_to_exclude="" # space seperated list of chomosomes to exclude
             ):
    
    if ref_genome_build == "hg38":
        primary_contigs = ['chr{}'.format(i) for i in range(1,23)]
        primary_contigs.extend(['chrX','chrY','chrM'])
        ref_config = _hg38_config_gen(wgs)
    elif ref_genome_build == "hg19":
        primary_contigs = [str(x) for x in range(1, 23)] + ["X", "Y", "M"]
        ref_config = _hg19_config_gen(wgs)
    else:
        raise ValueError(f"did not recognize ref build {ref_genome_build}. Expected hg38 or hg19")
    
    bam_localization_task = wolf.LocalizeToDisk(
          files = {
            "bam" : bam,
            "bai" : bai,
          },
          persistent_disk_dry_run = persistent_dry_run
    )
    
    localization_task = wolf.LocalizeToDisk(
            files = dict(
                vcf = vcf,
                db_snp_vcf = db_snp_vcf,
                ascat_loci_list = ascat_loci_list,
                ref_fasta = ref_config["ref_fasta"],
                ref_fasta_idx = ref_config["ref_fasta_idx"],
                ref_fasta_dict = ref_config["ref_fasta_dict"],

                genetic_map_file = ref_config["genetic_map_file"],
                  
                common_snp_list = ref_config["common_snp_list"],
          
                cytoband_file = ref_config["cytoband_file"],
        
                # reference panel
                **ref_config["ref_panel_1000g"]
                    )
        )
    # create mutect scatter chunks
  
    split_vcf = wolf.Task(
      name = "split_vcf",
      inputs = { "vcf" : localization_task["vcf"] },
      script = """
      grep '^#' ${vcf} > header
      sed '/^#/d' ${vcf} | split -l 10000 -d -a 3 --filter='cat header /dev/stdin > $FILE' - VCF_chunk
      """,
      outputs = { "shards" : "VCF_chunk*" }
    )
    
    # create coverage scatter intervals
    split_intervals_task = split_intervals.split_intervals(
      bam = bam_localization_task["bam"],
      bai = bam_localization_task["bai"],
      interval_type = "bed",
      selected_chrs = primary_contigs
    )

    # shim task to transform split_intervals files into subset parameters for covcollect task
    @prefect.task
    def interval_gather(interval_files, primary_contigs):
        ints = []
        for f in interval_files:
            ints.append(pd.read_csv(f, sep = "\t", header = None, names = ["chr", "start", "end"]))
        #filter non-primary contigs
        full_bed = pd.concat(ints).sort_values(["chr", "start", "end"]).astype({ "chr" : str })
        filtered_bed = full_bed.loc[full_bed.chr.isin(primary_contigs)]
        return filtered_bed

    subset_intervals = interval_gather(
      split_intervals_task["interval_files"],
      primary_contigs
    )

    # dispatch coverage scatter
    tumor_cov_collect_task = cov_collect.Covcollect(
      inputs = dict(
        bam = bam_localization_task["bam"],
        bai = bam_localization_task["bai"],
        intervals = target_list,
        subset_chr = subset_intervals["chr"],
        subset_start = subset_intervals["start"],
        subset_end = subset_intervals["end"],
      )
    )

    # gather tumor coverage
    tumor_cov_gather_task = wolf.Task(
      name = "gather_coverage",
      inputs = { "coverage_beds" : [tumor_cov_collect_task["coverage"]],
                 "sample_name" : sample_name },
      script = """cat $(cat ${coverage_beds}) > ${sample_name}_covcollect.bed""",
      outputs = { "coverage" : "*covcollect.bed" }
    )

    # Mutect1
    m1_scatter = mutect.mutect1(
      inputs = {
        "pairName" : "platinum",
        "caseName" : "platinum",
        "t_bam" : bam_localization_task["bam"],
        "t_bai" : bam_localization_task["bai"],
        "force_calling" : True,
        "intervals" : split_vcf["shards"],
        "fracContam" : 0,
        "refFasta" : localization_task["ref_fasta"],
        "refFastaIdx" : localization_task["ref_fasta_idx"],
        "refFastaDict" :  localization_task["ref_fasta_dict"]
      }
    )

    # gather mutect shards
    m1_gather = wolf.Task(
      name = "m1_gather",
      inputs = { "callstats_array" : [m1_scatter["mutect1_cs"]],
                 "sample_name" : sample_name },
      script = """
      head -n2 $(head -n1 ${callstats_array}) > header
      while read -r i; do
        sed '1,2d' $i
      done < ${callstats_array} | sort -k1,1V -k2,2n > cs_sorted
      cat header cs_sorted > ${sample_name}_mutect_callstats.tsv
      """,
      outputs = { "cs_gather" : "*mutect_callstats.tsv" }
    )
    
    if chrs_to_exclude != "":
        exclude_chrs_list = '|'.join(chrs_to_exclude.split(' '))
        chr_exclusion_task = wolf.Task(name = 'exclude_chrs',
                                       inputs = {'mutect_cs': m1_gather["cs_gather"],
                                                 'covcollect_bed': tumor_cov_gather_task["coverage"],
                                                 'chr_ex_string': exclude_chrs_list},
                                       script = """
                                       cat ${mutect_cs} | grep -vE "^(${chr_ex_string})" > $(basename ${mutect_cs:0:-4}_chrs_excluded.tsv) && cat ${covcollect_bed} | grep -vE "^(${chr_ex_string})" > $(basename ${covcollect_bed:0:-4}_chrs_excluded.bed)""",
                                       outputs = {"mutect_cs": "*.tsv",
                                                  "covcollect_bed": "*.bed"
                                                 }
                                       )
        mutect_callstats_output = chr_exclusion_task["mutect_cs"]
        covcollect_output = chr_exclusion_task["covcollect_bed"]
    else:
        mutect_callstats_output = m1_gather["cs_gather"]
        covcollect_output = tumor_cov_gather_task["coverage"]
        
    if preprocess:

        # use hapaseg callstats filtering to grab the genotypes
        hapaseg_cs_task = HapASeg_Preprocess_Callstats(inputs = {
                                            "callstats": mutect_callstats_output,
                                            "sample_name": sample_name,
                                            "dummy_normal": dummy_normal
                                            }
                                        )

        if not dummy_normal:
            #only run phasing if the vcf matches the normal
            # phasing will be the same for all benchmarking runs so we will also cache this result

            convert_task = wolf.Task(
              name = "convert_het_pulldown",
              inputs = {
                "genotype_file" : hapaseg_cs_task["hapaseg_genotype"],
                "sample_name" : "test", # TODO: allow to be specified
                "ref_fasta" : localization_task["ref_fasta"],
                "ref_fasta_idx" : localization_task["ref_fasta_idx"],
                "ref_fasta_dict" : localization_task["ref_fasta_dict"],
              }, 
              script = r"""
            set -x
            bcftools convert --tsv2vcf ${genotype_file} -c CHROM,POS,AA -s ${sample_name} \
              -f ${ref_fasta} -Ou -o all_chrs.bcf && bcftools index all_chrs.bcf
            for chr in $(bcftools view -h all_chrs.bcf | ssed -nR '/^##contig/s/.*ID=(.*),.*/\1/p' | head -n24); do
              bcftools view -Ou -r ${chr} -o ${chr}.chrsplit.bcf all_chrs.bcf && bcftools index ${chr}.chrsplit.bcf
            done
            """,
              outputs = {
                "bcf" : "*.chrsplit.bcf",
                "bcf_idx" : "*.chrsplit.bcf.csi"
              },
              docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"
            )

            #
            # ensure that BCFs/indices/reference BCFs are in the same order
            @prefect.task
            def order_indices(bcf_path, bcf_idx_path, localization_task):
                # BCFs
                F = pd.DataFrame(dict(bcf_path = bcf_path))
                F = F.set_index(F["bcf_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

                # indices
                F2 = pd.DataFrame(dict(bcf_idx_path = bcf_idx_path))
                F2 = F2.set_index(F2["bcf_idx_path"].apply(os.path.basename).str.replace(r"^((?:chr)?(?:[^.]+)).*", r"\1"))

                F = F.join(F2)

                # prepend "chr" to F's index if it's missing
                idx = ~F.index.str.contains("^chr")
                if idx.any():
                    new_index = F.index.values
                    new_index[idx] = "chr" + F.index[idx]
                    F = F.set_index(new_index)

                # reference panel BCFs
                R = pd.DataFrame({ "path" : localization_task } ).reset_index()
                F = F.join(R.join(R.loc[R["index"].str.contains("^chr.*_bcf$"), "index"].str.extract(r"(?P<chr>chr[^_]+)"), how = "right").set_index("chr").drop(columns = ["index"]).rename(columns = { "path" : "ref_bcf" }), how = "inner")
                F = F.join(R.join(R.loc[R["index"].str.contains("^chr.*csi$"), "index"].str.extract(r"(?P<chr>chr[^_]+)"), how = "right").set_index("chr").drop(columns = ["index"]).rename(columns = { "path" : "ref_bcf_idx" }), how = "inner")

                return F

            F = order_indices(convert_task["bcf"], convert_task["bcf_idx"], localization_task)

            #
            # run Eagle, per chromosome
            eagle_task = phasing.eagle(
              inputs = dict(
                genetic_map_file = localization_task["genetic_map_file"],
                vcf_in = F["bcf_path"],
                vcf_idx_in = F["bcf_idx_path"],
                vcf_ref = F["ref_bcf"],
                vcf_ref_idx = F["ref_bcf_idx"],
                output_file_prefix = "foo",
                num_threads = 1,
              ),
              resources = { "cpus-per-task" : 2, "mem":'8G'},
              outputs = {"phased_vcf" : "foo.vcf"}
            )

            # combine VCFs
            combine_vcf_task = wolf.Task(
              name = "combine_vcfs",
              inputs = { "vcf_array" : [eagle_task["phased_vcf"]],
                         "sample_name" : sample_name },
              script = "bcftools concat -O u $(cat ${vcf_array} | tr '\n' ' ') | bcftools sort -O v -o ${sample_name}_eagle_phased.vcf",
              outputs = { "combined_vcf" : "*phased.vcf" },
              docker = "gcr.io/broad-getzlab-workflows/base_image:v0.0.5"
            )

        # preprocess callstats

        facets_cs_task = Facets_Preprocess_Callstats(inputs = {
                                            "callstats": mutect_callstats_output,
                                            "sample_name": sample_name,
                                            "db_snp_vcf": localization_task["db_snp_vcf"]
                                            }
                                        )

        ascat_cs_task = ASCAT_Preprocess_Callstats(inputs = {
                                            "callstats": mutect_callstats_output,
                                            "sample_name": sample_name,
                                            "ascat_loci_list": localization_task["ascat_loci_list"]
                                            }
                                        )
    
    if upload_bucket_gs_path is not None:
        # upload files to bucket. upload bucket_path is the ~root of the bucket ~directory
        # of the given sample type. method specific results go in apropriately named subdirs
        upload_root_task = UploadToBucket(files = [mutect_callstats_output,
                                                   covcollect_output],
                       bucket = upload_bucket_gs_path
                      )

        if preprocess:
            if not dummy_normal: 
                upload_phasing_task = UploadToBucket(files = combine_vcf_task["combined_vcf"],
                               bucket = upload_bucket_gs_path
                              )

            upload_hapaseg_task = UploadToBucket(files = [hapaseg_cs_task["hapaseg_hetsite_depths"],
                                                          hapaseg_cs_task["hapaseg_filtered_cs"]],
                           bucket = upload_bucket_gs_path.rstrip('/') + '/hapaseg/'
                          )

            upload_facets_task = UploadToBucket(files = [facets_cs_task["facets_variant_depths"],
                                                         facets_cs_task["facets_filtered_variants"]],
                           bucket = upload_bucket_gs_path.rstrip('/') + '/facets/'
                          )
     
            upload_ascat_task = UploadToBucket(files = [ascat_cs_task["ascat_variant_depths"],
                                                        ascat_cs_task["ascat_filtered_variants"]],
                           bucket = upload_bucket_gs_path.rstrip('/') + '/ascat/'
                          )

