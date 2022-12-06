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
