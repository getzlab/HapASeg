import pandas as pd
import wolf
from wolF import workflow
from wolf.fc import SyncToWorkspace
import dalmatian

workspace = 'mgh-getzlab-ovarianspore-t/getzlab-ovarianspore_REBC_methods'
wm = dalmatian.WorkspaceManager(workspace)

wic = wolf.fc.WorkspaceInputConnector(workspace)
Pj = wic.get_pairs_as_joint_samples()

with pd.option_context('display.max_columns', None):
    print(Pj)

with wolf.Workflow(workflow = workflow.workflow, namespace = "HapASeg_ovarianspore") as w:
    for pair, p in Pj.loc[Pj["sample_type_T"] == "Tumor"].iterrows():
        print('### pair,p  #### ',pair,p)
        w.run(
            RUN_NAME = pair,
            workspace = workspace,
            workspace_pairname = pair,
            tumor_bam = p["clean_bam_file_capture_T"],
            tumor_bai = p["clean_bai_file_capture_T"],
            normal_bam = p["clean_bam_file_capture_N"],
            normal_bai = p["clean_bai_file_capture_N"],
            target_list = 2000,
            ref_genome_build = "hg19",
            annotation_postfix = "HapSeg_V1",  
            sync = True        
        )
        
