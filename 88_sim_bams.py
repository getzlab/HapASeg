import dalmatian
import pandas as pd
import subprocess
import wolf

## import alignment workflow
alignment = wolf.ImportTask(
  "git@github.com:getzlab/alignment_pipeline_wolF.git",
  "alignment",
  commit = "7a71d3e"
)

alignment = wolf.ImportTask(
  "/mnt/j/proj/pipe/20220831_wolfalign",
  "alignment"
)

def revert_and_upload(**kwargs):
    bam, bai = alignment.revert_and_alignment_workflow(**kwargs)

    sample_name = kwargs["sample_name"]
    wolf.UploadToBucket(
      files = [bam, bai],
      bucket = f"gs://jh-xfer/realignment/{sample_name}/"
    )

## get NA12878 samples
gspaths = subprocess.run("gsutil ls gs://jh-xfer/NA12878_twist/*.ba?", capture_output = True, shell = True).stdout.decode().rstrip().split("\n")

# TWIST: grab samples from Broad (uploaded to personal bucket)
WES_TWIST = pd.Series(gspaths).str.extract(r"(?P<path>.*_(?P<rep>\d+).(?P<ext>ba[mi]))$").pivot(
    index = "rep", columns = "ext", values = "path"
).loc[:, ["bam", "bai"]]
WES_TWIST.index = "NA12878_TWIST_" + WES_TWIST.index.astype(str)

# ICE, get from Tag Team workspace
WM = dalmatian.WorkspaceManager("broadtagteam/tag_781_Twist_vs_ICE_HapMap_SomaticWES")
S = WM.get_samples()

WES_ICE = S.loc[S.index.str.contains("NA12878.*ICE"), ["bam", "bam_index"]]

## get RT samples
WM = dalmatian.WorkspaceManager("broad-firecloud-ibmwatson/Getz_Wu_Richters_WGS_UK")
S = WM.get_samples()
WGS_RT = S.loc[S["sample_type"] != "Normal", ["output_bam", "output_bam_index"]]

## RT normals
S = WM.get_samples()
WGS_RT_N = S.loc[S["sample_type"] == "Normal", ["output_bam", "output_bam_index"]]

## now run everything
with wolf.Workflow(workflow = revert_and_upload, namespace = "hapaseg_sim_realign", max_concurrent_flows = 20, max_concurrent_flow_tasks = 20, scheduler_processes = 8) as w:
    # TWIST NA12878
    for samp, bam, bai in WES_TWIST.itertuples():
        w.run(
          RUN_NAME = samp, 

          bam = bam,
          bai = bai,
          sample_name = samp,
          bwa_index_dir = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/",
          n_revert_shards = 15,
          n_bwa_shards = 15
        )

#    # ICE NA12878
#    for samp, bam, bai in WES_ICE.itertuples():
#        w.run(
#          RUN_NAME = samp, 
#
#          bam = bam,
#          bai = bai,
#          sample_name = samp,
#          bwa_index_dir = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/",
#          n_revert_shards = 15,
#          n_bwa_shards = 15
#        )

    # FFPE WGS RT/CLL
    for samp, bam, bai in WGS_RT.itertuples():
        w.run(
          RUN_NAME = samp, 

          bam = bam,
          bai = bai,
          sample_name = samp,
          bwa_index_dir = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/",
          n_revert_shards = 30,
          n_bwa_shards = 50
        )

    for samp, bam, bai in WGS_RT_N.itertuples():
        w.run(
          RUN_NAME = samp, 

          bam = bam,
          bai = bai,
          sample_name = samp,
          bwa_index_dir = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/",
          n_revert_shards = 30,
          n_bwa_shards = 50
        )

    # platinum
    w.run(
      RUN_NAME = "NA12878_platinum", 

      bam = "gs://jh-xfer/NA12878_bwamem_illumina_platinum_bed.bam",
      bai = "gs://jh-xfer/NA12878_bwamem_illumina_platinum_bed.bam.bai",
      sample_name = "NA12878_platinum",
      bwa_index_dir = "gs://getzlab-workflows-reference_files-oa/hg38/gdc/",
      n_revert_shards = 30,
      n_bwa_shards = 50
    )

## run covcollect on FFPE BAMs
from wolF import workflow
import prefect

def covcollect_workflow(tumor_bam, tumor_bai):
    tumor_bam_localization_task = wolf.LocalizeToDisk(
      files = {
        "t_bam" : tumor_bam,
        "t_bai" : tumor_bai,
      },
    )

    primary_contigs = ['chr{}'.format(i) for i in range(1,23)]
    primary_contigs.extend(['chrX','chrY','chrM'])

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

    tumor_split_intervals_task = workflow.split_intervals.split_intervals(
      bam = tumor_bam_localization_task["t_bam"],
      bai = tumor_bam_localization_task["t_bai"],
      interval_type = "bed",
      selected_chrs = primary_contigs
    )

    tumor_subset_intervals = interval_gather(
      tumor_split_intervals_task["interval_files"],
      primary_contigs
    )

    # dispatch coverage scatter
    tumor_cov_collect_task = workflow.cov_collect.Covcollect(
      inputs = dict(
        bam = tumor_bam_localization_task["t_bam"],
        bai = tumor_bam_localization_task["t_bai"],
        intervals = 2000,
        subset_chr = tumor_subset_intervals["chr"],
        subset_start = tumor_subset_intervals["start"],
        subset_end = tumor_subset_intervals["end"],
      )
    )

    # gather tumor coverage
    tumor_cov_gather_task = wolf.Task(
      name = "gather_coverage",
      inputs = { "coverage_beds" : [tumor_cov_collect_task["coverage"]] },
      script = """cat $(cat ${coverage_beds}) > coverage_cat.bed""",
      outputs = { "coverage" : "coverage_cat.bed" }
    )

B = pd.Series(subprocess.run("gsutil ls gs://jh-xfer/realignment/*/*.ba?", shell = True, capture_output = True).stdout.decode().rstrip().split("\n"))
B = B.str.extract(r"(?P<file>.*realignment/(?P<samp>.*)/.*\.(?P<ext>ba[im]))").pivot(index = "samp", columns = "ext", values = "file")

with wolf.Workflow(workflow = covcollect_workflow, namespace = "FFPE_covcollect") as w:
    for samp, b in B.loc[B.index.str.contains("(1001|1008|1022|1032)LN")].iterrows():
        w.run(RUN_NAME = samp, tumor_bam = b["bam"], tumor_bai = b["bai"])

# subset to diploid regions
from capy import mut

# 1022: 1-3, 5, 8, 9, 11, 12, 14-
C = pd.read_csv("/mnt/nfs/FFPE_covcollect/CH1022LN/gather_coverage__2022-11-14--06-58-36_g414gwy_tbhx1ki_oy0wbil3gnmtk/jobs/0/workspace/coverage_cat.bed", header = None, sep = "\t", names=["chr", "start", "end", "covcorr", "mean_frag_len", "std_frag_len", "num_frags", "tot_reads", "fail_reads"])
c2 = mut.convert_chr(C["chr"])
C_sub = C.loc[~c2.isin([4, 6, 7, 10, 13])]
C_sub.to_csv("benchmarking_data/1022_cov.bed", sep = "\t", header = None, index = False)

# for GATK -- set empty regions to 0, set covcorr = total reads
C_sub = C.copy()
C_sub["covcorr"] = C_sub["tot_reads"]
C_sub.loc[c2.isin([4, 6, 7, 10, 13]), "covcorr"] = 0
C_sub.to_csv("benchmarking_data/1022_cov_totreads.bed", sep = "\t", header = None, index = False)


# 1032: 1-8, 10-13, 15-
C = pd.read_csv("/mnt/nfs/FFPE_covcollect/CH1032LN/gather_coverage__2022-11-14--07-33-38_g414gwy_tbhx1ki_jeubgtdcgcp22/jobs/0/workspace/coverage_cat.bed", header = None, sep = "\t", names=["chr", "start", "end", "covcorr", "mean_frag_len", "std_frag_len", "num_frags", "tot_reads", "fail_reads"])
c2 = mut.convert_chr(C["chr"])
C_sub = C.loc[~c2.isin([9, 14])]
C_sub.to_csv("benchmarking_data/1032_cov.bed", sep = "\t", header = None, index = False)

# for GATK
C_sub = C.copy()
C_sub["covcorr"] = C_sub["tot_reads"]
C_sub.loc[c2.isin([9, 14]), "covcorr"] = 0
C_sub.to_csv("benchmarking_data/1032_cov_totreads.bed", sep = "\t", header = None, index = False)

# TODO: get depths at NA12878 positions for Facets/ASCAT/HapASeg
