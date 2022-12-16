import wolf
from benchmarking_workflows import *

with wolf.Workflow(workflow=Run_Sim_Workflows) as w:
    w.run(run_name='recomb_easy_sim_1',
          sim_profile='/home/opriebe/data/cnv_sim/benchmarking/sim_samples/cnv_profiles/easy_profile_1.pickle',
          purity = 0.7,
          sample_label='easy_sim_1',
          normal_vcf_path = '/home/opriebe/data/cnv_sim/NA12878/NA12878.vcf',
          ref_build = "hg38",
          ref_fasta = "/home/opriebe/data/ref/hg38/GRCh38.d1.vd1.fa",
          cytoband_file = "/home/opriebe/data/ref/hg38/cytoBand.txt",
          hapaseg_hetsite_depth_path='/home/opriebe/data/cnv_sim/benchmarking/hapaseg/NA12878_hetsites_depth.tsv',
          hapaseg_covcollect_path='/home/opriebe/data/cnv_sim/NA12878/NA12878_2kb_coverage_num_reads_g300.bed',
          hapaseg_phased_vcf_path='/home/opriebe/data/cnv_sim/benchmarking/hapaseg/NA12878_eagle_phasing.vcf',
          gatk_variant_depth_path = '/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_gatk_var_depth.tsv',
          gatk_coverage_tsv_path = '/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_gatk_cov_counts.tsv',
          gatk_sim_normal_allelecounts_path='/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_gatk_cs_sim_normal.tsv',
          gatk_raw_gatk_allelecounts_path='/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_platinum_all_vars_no_sex.allelecounts.tsv',
          gatk_raw_gatk_coverage_path='/home/opriebe/data/cnv_sim/benchmarking/gatk/NA12878_hg38_wgs_1kb_gatk_no_sex_frag.counts.hdf5',
          gatk_sequence_dictionary='/home/opriebe/data/cnv_sim/benchmarking/gatk/1kG_PoN/Homo_sapiens_assembly38.dict',
          gatk_count_panel='/home/opriebe/data/cnv_sim/benchmarking/gatk/1kG_PoN/GATK_PoN_50samples_1kG.hdf5',
          facets_variant_depth_path = '/home/opriebe/data/cnv_sim/benchmarking/facets/facets_cs_variant_depths.tsv',
          facets_filtered_variants_path = '/home/opriebe/data/cnv_sim/benchmarking/facets/facets_cs_variant_filtered.tsv',
          ascat_variant_depth_path = '/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_cs_variant_depths.tsv',
          ascat_filtered_variants_path = '/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_cs_variant_filtered.tsv',
          ascat_GC_correction_file = '/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_loci/GC_G1000_hg38.txt',
          ascat_RT_correction_file='/home/opriebe/data/cnv_sim/benchmarking/ascat/ascat_loci/RT_G1000_hg38.txt'
          )

## Run on large range of simulated profiles
import numpy as np
import pandas as pd
import subprocess

purities = pd.Series(np.r_[0.1:1:0.1], name = "purities")
sim_profiles = pd.Series(
  subprocess.run(
    "gsutil ls gs://hapaseg-pub/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile*.p*",
    capture_output = True,
    shell = True
  ).stdout.decode().rstrip().split("\n")
).str.extract(
  r"(?P<file>.*profile_(?P<entropy>\d+)_(?P<id>\d+)\.(?P<ext>pickle|png))"
).astype(
  { "entropy" : int, "id" : int }
).pivot(index = ["id", "entropy"], columns = "ext", values = "file").reset_index()

sim_profiles = sim_profiles.sample(50, random_state = 1337)

sim_profiles = sim_profiles.merge(purities, how = "cross")

with wolf.Workflow(workflow=Run_Sim_Workflows, namespace = "CNV_benchmark", scheduler_processes = 10, common_task_opts = { "cleanup_job_workdir" : True }) as w:
    for _, profile in sim_profiles.iterrows():
        purity = np.around(profile['purities'], 1)
        name = f"{profile['entropy']}_{profile['id']}_{purity}"
        w.run(run_name = name,
              sim_profile = profile["pickle"],
              purity = profile["purities"],
              sample_label = name,
              normal_vcf_path = 'gs://hapaseg-pub/cnv_sim/NA12878/NA12878.vcf',
              ref_build = "hg38",
              ref_fasta = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
              cytoband_file = "gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",

              hapaseg_hetsite_depth_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_hetsites_depth.tsv',
              hapaseg_covcollect_path='gs://hapaseg-pub/cnv_sim/NA12878/NA12878_2kb_coverage_num_reads_g300.bed',
              hapaseg_phased_vcf_path='gs://hapaseg-pub/cnv_sim/benchmarking/hapaseg/NA12878_eagle_phasing.vcf',

              gatk_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_var_depth.tsv',
              gatk_coverage_tsv_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cov_counts.tsv',
              gatk_sim_normal_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_gatk_cs_sim_normal.tsv',
              gatk_raw_gatk_allelecounts_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_platinum_all_vars_no_sex.allelecounts.tsv',
              gatk_raw_gatk_coverage_path='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/NA12878_hg38_wgs_1kb_gatk_no_sex_frag.counts.hdf5',
              gatk_sequence_dictionary='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/Homo_sapiens_assembly38.dict',
              gatk_count_panel='gs://hapaseg-pub/cnv_sim/benchmarking/gatk/1kG_PoN/GATK_PoN_50samples_1kG.hdf5',

              facets_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_depths.tsv',
              facets_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/facets/facets_cs_variant_filtered.tsv',

              ascat_variant_depth_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_depths.tsv',
              ascat_filtered_variants_path = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_cs_variant_filtered.tsv',
              ascat_GC_correction_file = 'gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/GC_G1000_hg38.txt',
              ascat_RT_correction_file='gs://hapaseg-pub/cnv_sim/benchmarking/ascat/ascat_loci/RT_G1000_hg38.txt'
              )

pd.to_pickle(w.flow_results, "benchmarking.50.pickle")

## re-run comparator with standardized purity

F = pd.read_pickle("benchmarking.50.pickle")

def comparison_workflow(
  sim_profile=None,
  sample_label=None,

  normal_vcf_path=None,
  hapaseg_hetsite_depth_path=None,
  hapaseg_covcollect_path=None,
  input_purity=None,

  ref_fasta=None,
  cytoband_file=None,

  hapaseg_seg_file = None,

  gatk_tumor_coverage = None,
  gatk_tumor_allele_counts = None,
  gatk_seg_file = None,

  facets_input_counts = None,
  facets_seg_file = None,

  ascat_tumor_logR = None,
  ascat_tumor_BAF = None,
  ascat_raw_segments = None,
):
    seg_file_gen_task = Generate_Groundtruth_Segfile(inputs= {
                            "sample_label": sample_label,
                            "purity":0.7,
                            "sim_profile":sim_profile,
                            "normal_vcf_path":normal_vcf_path,
                            "hapaseg_hetsite_depth_path": hapaseg_hetsite_depth_path,
                            "hapaseg_coverage_tsv":hapaseg_covcollect_path
                        },
                        extra_localization_args = { "localize_to_persistent_disk" : True }
                    ) 

    localization_task = wolf.LocalizeToDisk(files = {
                                    "ref_fasta" : ref_fasta,
                                    "cytoband_file" : cytoband_file
                                    })

    hapaseg_downstream = Downstream_HapASeg_Analysis(inputs = {
                                   "hapaseg_seg_file": hapaseg_seg_file,
                                   "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
                                   "sample_name": f"{sample_label}_{input_purity}",
                                   "ref_fasta": localization_task["ref_fasta"],
                                   "cytoband_file": localization_task["cytoband_file"]
                        }, job_avoid = False)

    gatk_downstream = Downstream_GATK_Analysis(
              inputs={"gatk_sim_cov_input": gatk_tumor_coverage,
                      "gatk_sim_acounts" : gatk_tumor_allele_counts,
                      "gatk_seg_file" : gatk_seg_file,
                      "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
                      "sample_name": f"{sample_label}_{input_purity}",
                      "ref_fasta": localization_task["ref_fasta"],
                      "cytoband_file": localization_task["cytoband_file"]
                    }
            )

    facets_downstream = Downstream_Facets_Analysis(
                            inputs={"facets_input_counts": facets_input_counts,
                                    "facets_seg_file": facets_seg_file,
                                     "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
                                     "sample_name": f"{sample_label}_{input_purity}",
                                     "ref_fasta": localization_task["ref_fasta"],
                                     "cytoband_file": localization_task["cytoband_file"]
                                   }
                            )

    ascat_downstream = Downstream_ASCAT_Analysis(
                                inputs={"ascat_t_logr": ascat_tumor_logR,
                                        "ascat_t_baf": ascat_tumor_BAF,
                                        "ascat_seg_file": ascat_raw_segments,
                                        "ground_truth_seg_file": seg_file_gen_task["ground_truth_seg_file"],
                                        "sample_name": f"{sample_label}_{input_purity}",
                                        "ref_fasta": localization_task["ref_fasta"],
                                        "cytoband_file": localization_task["cytoband_file"]
                                       }
                                )

with wolf.Workflow(workflow = comparison_workflow, namespace = "CNV_benchmark") as w:
    for samp in F.keys():
        sim_inputs = F[samp]["Generate_Groundtruth_Segfile"].results["inputs"].iloc[0].to_dict()

        # in case some samples don't have various outputs
        try:
            w.run(
              RUN_NAME = samp + "_recompare",

              sim_profile=sim_inputs["sim_profile"],
              sample_label=samp,

              normal_vcf_path=sim_inputs["normal_vcf_path"],
              hapaseg_hetsite_depth_path=sim_inputs["hapaseg_hetsite_depth_path"],
              hapaseg_covcollect_path=sim_inputs["hapaseg_coverage_tsv"],
              input_purity=sim_inputs["purity"],

              ref_fasta="gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
              cytoband_file="gs://getzlab-workflows-reference_files-oa/hg38/cytoBand.txt",

              hapaseg_seg_file = F[samp]["Hapaseg_run_acdp"]["acdp_segfile"],

              gatk_tumor_coverage = F[samp]["Generate_GATK_Sim_Data"]["tumor_coverage_tsv"],
              gatk_tumor_allele_counts = F[samp]["Generate_GATK_Sim_Data"]["tumor_allele_counts"],
              gatk_seg_file = F[samp]["GATK_CNV_model_segments"]["model_segments_post_smoothing"],

              facets_input_counts = F[samp]["Generate_Facets_Sim_Data"]["facets_input_counts"],
              facets_seg_file = F[samp]["Facets"]["facets_seg_file"],

              ascat_tumor_logR = F[samp]["Generate_ASCAT_Sim_Data"]["ascat_tumor_logR"],
              ascat_tumor_BAF = F[samp]["Generate_ASCAT_Sim_Data"]["ascat_tumor_BAF"],
              ascat_raw_segments = F[samp]["ASCAT"]["ascat_raw_segments"]
            )
        except:
            pass

pd.to_pickle(w.flow_results, "benchmarking.50.rescale.pickle")

## load in and plot results

F2 = pd.read_pickle("benchmarking.50.rescale.pickle")

methods = ['Downstream_ASCAT_Analysis', 'Downstream_Facets_Analysis', 'Downstream_GATK_Analysis', 'Downstream_HapASeg_Analysis']
D = { samp : { method : F2[samp][method] for method in methods } for samp in F2.keys() }

D = pd.DataFrame.from_dict(D, orient = "index")
D = pd.concat([D.reset_index(drop = True), D.index.str.extract(r"(?P<entropy>\d+)_\d+_(?P<purity>[\d\.]+)_recompare")], axis = 1).astype({ "entropy" : int, "purity" : float })

for method in methods:
    D[method + "_MAD"] = np.nan

for idx, row in D.iterrows():
    for method in methods:
        try:
            score = pd.read_csv(row[method]["comparison_results"], sep = "\t")
            D.loc[idx, method + "_MAD"] = score["mad_score"][0]
        except:
            pass

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(11); plt.clf()

minent = D["entropy"].min()

for ent, idx in D.groupby("entropy").groups.items():
    jit = lambda x : np.random.rand(len(x))/20 - 0.025
    sz = 5 + (ent - minent)**2/500
    l2 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_GATK_Analysis_MAD"], s = sz, color = colors[1], alpha = 0.8)
    l3 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_ASCAT_Analysis_MAD"], s = sz, color = colors[2], alpha = 0.8)
    l4 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_Facets_Analysis_MAD"], s = sz, color = colors[3], alpha = 0.8)
    l1 = plt.scatter(D.loc[idx, "purity"] + jit(idx), D.loc[idx, "Downstream_HapASeg_Analysis_MAD"], s = sz, color = colors[0], alpha = 0.8)

plt.xticks(np.r_[0.1:1:0.1])

plt.yscale("log")
plt.xlabel("Purity")
plt.ylabel("MAD")

plt.legend([l1, l2, l3, l4], ["HapASeg", "GATK4 CNV", "ASCAT", "FACETS"], loc = "lower center", ncol = 4)
plt.ylim([0.07, 140])
