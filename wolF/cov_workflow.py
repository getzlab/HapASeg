import glob
import numpy as np
import os
import pandas as pd
import pickle
import prefect
import subprocess
import tempfile
import wolf

#
# import tasks

# for genotyping het sites/getting het site coverage
het_pulldown = wolf.ImportTask(
    task_path="git@github.com:getzlab/het_pulldown_from_callstats_TOOL.git",
    task_name="het_pulldown",
)

mutect1 = wolf.ImportTask(
    task_path="git@github.com:getzlab/MuTect1_TOOL.git", task_name="mutect1"
)

# for phasing
phasing = wolf.ImportTask(
    task_path="git@github.com:getzlab/phasing_TOOL.git", task_name="phasing"
)

# for Hapaseg itself
hapaseg = wolf.ImportTask(
    task_path="../",  # TODO: make remote
    task_name="hapaseg",
)

# for coverage collection
split_intervals = wolf.ImportTask(
    task_path="git@github.com:getzlab/split_intervals_TOOL.git",
    task_name="split_intervals",
)

cov_collect = wolf.ImportTask(
    task_path="git@github.com:getzlab/covcollect.git", task_name="covcollect"
)

####
# defining reference config generators for hg19 and hg38


# hg19
def _hg19_config_gen(wgs):
    hg19_ref_panel = pd.DataFrame(
        {
            "path": subprocess.check_output(
                "gsutil ls gs://getzlab-workflows-reference_files-oa/hg19/1000genomes/*.bcf*",
                shell=True,
            )
            .decode()
            .rstrip()
            .split("\n")
        }
    )
    hg19_ref_panel = hg19_ref_panel.join(
        hg19_ref_panel["path"].str.extract(
            ".*(?P<chr>chr[^.]+)\.(?P<ext>bcf(?:\.csi)?)"
        )
    )
    hg19_ref_panel["key"] = hg19_ref_panel["chr"] + "_" + hg19_ref_panel["ext"]
    hg19_ref_dict = (
        hg19_ref_panel.loc[:, ["key", "path"]]
        .set_index("key")["path"]
        .to_dict()
    )

    hg19_ref_config = dict(
        ref_fasta="gs://getzlab-workflows-reference_files-oa/hg19/Homo_sapiens_assembly19.fasta",
        ref_fasta_idx="gs://getzlab-workflows-reference_files-oa/hg19/Homo_sapiens_assembly19.fasta.fai",
        ref_fasta_dict="gs://getzlab-workflows-reference_files-oa/hg19/Homo_sapiens_assembly19.dict",
        genetic_map_file="gs://getzlab-workflows-reference_files-oa/hg19/eagle/genetic_map_hg19_withX.txt.gz",
        common_snp_list="gs://getzlab-workflows-reference_files-oa/hg19/gnomad/gnomAD_MAF10_80pct_45prob.txt",
        cytoband_file="gs://getzlab-workflows-reference_files-oa/hg19/cytoBand.txt",
        repl_file="gs://opriebe-tmp/GSE137764_H1.hg19_liftover.pickle",
        ref_panel_1000g=hg19_ref_dict,
    )
    # if we're using whole genome we can use the precomputed gc file for 200 bp bins
    hg19_ref_config["gc_file"] = (
        "gs://opriebe-tmp/GC_hg19_200bp.pickle" if wgs else ""
    )
    return hg19_ref_config


# hg38
def _hg38_config_gen(wgs):
    hg38_ref_panel = pd.DataFrame(
        {
            "path": subprocess.check_output(
                "gsutil ls gs://getzlab-workflows-reference_files-oa/hg38/1000genomes/*.bcf*",
                shell=True,
            )
            .decode()
            .rstrip()
            .split("\n")
        }
    )
    hg38_ref_panel = hg38_ref_panel.join(
        hg38_ref_panel["path"].str.extract(
            ".*(?P<chr>chr[^.]+)\.(?P<ext>bcf(?:\.csi)?)"
        )
    )
    hg38_ref_panel["key"] = hg38_ref_panel["chr"] + "_" + hg38_ref_panel["ext"]
    hg38_ref_dict = (
        hg38_ref_panel.loc[:, ["key", "path"]]
        .set_index("key")["path"]
        .to_dict()
    )

    hg38_ref_config = dict(
        ref_fasta="gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa",
        ref_fasta_idx="gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai",
        ref_fasta_dict="gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict",
        genetic_map_file="gs://getzlab-workflows-reference_files-oa/hg38/eagle/genetic_map_hg38_withX.txt.gz",
        common_snp_list="gs://getzlab-workflows-reference_files-oa/hg38/gnomad/gnomAD_MAF10_50pct_45prob_hg38_final.txt",
        cytoband_file="gs://getzlab-workflows-reference_files-oa/hg19/cytoBand.txt",
        repl_file="gs://opriebe-tmp/GSE137764_H1.hg38.pickle",
        ref_panel_1000g=hg38_ref_dict,
    )
    # if we're using whole genome we can use the precomputed gc file for 200 bp bins
    # hg38_ref_config['gc_file'] = 'gs://opriebe-tmp/GC_hg38_2kb.pickle' if wgs else ""
    hg38_ref_config["gc_file"] = ""
    return hg38_ref_config


def workflow(
    callstats_file=None,
    tumor_bam=None,
    tumor_bai=None,
    tumor_coverage_bed=None,
    normal_bam=None,
    normal_bai=None,
    normal_coverage_bed=None,
    ref_genome_build=None,  # must be hg19 or hg38
    target_list=None,
    localization_token=None,
    persistant_dry_run=False,
):
    ###
    # integer target list implies wgs
    bin_width = target_list if isinstance(target_list, int) else 1
    wgs = True if bin_width > 1 else False

    # Select config based on ref genome choice
    if ref_genome_build is None:
        raise ValueError(
            "Reference genome must be specified! Options are 'hg19' or hg38'"
        )
    elif ref_genome_build == "hg19":
        ref_config = _hg19_config_gen(wgs)
    elif ref_genome_build == "hg38":
        ref_config = _hg38_config_gen(wgs)
    else:
        raise ValueError(
            "Reference genome options are 'hg19' or hg38', got {}".format(
                ref_genome_build
            )
        )

    localization_task = wolf.LocalizeToDisk(
        files=dict(
            ref_fasta=ref_config["ref_fasta"],
            ref_fasta_idx=ref_config["ref_fasta_idx"],
            ref_fasta_dict=ref_config["ref_fasta_dict"],
            genetic_map_file=ref_config["genetic_map_file"],
            common_snp_list=ref_config["common_snp_list"],
            cytoband_file=ref_config["cytoband_file"],
            repl_file=ref_config["repl_file"],
            gc_file=ref_config["gc_file"],
            # reference panel
            **ref_config["ref_panel_1000g"],
        )
    )

    #
    # localize BAMs to RODISK
    if tumor_bam is not None and tumor_bai is not None:
        tumor_bam_localization_task = wolf.LocalizeToDisk(
            files={
                "t_bam": tumor_bam,
                "t_bai": tumor_bai,
            },
            token=localization_token,
            persistent_disk_dry_run=persistant_dry_run,
        )
        collect_tumor_coverage = True
    elif tumor_coverage_bed is not None:
        collect_tumor_coverage = False
    else:
        raise ValueError(
            "You must supply either a tumor BAM+BAI or a tumor coverage BED file!"
        )

    use_normal_coverage = True
    if normal_bam is not None and normal_bai is not None:
        normal_bam_localization_task = wolf.LocalizeToDisk(
            files={"n_bam": normal_bam, "n_bai": normal_bai},
            token=localization_token,
            persistent_disk_dry_run=persistant_dry_run,
        )
        collect_normal_coverage = True
    elif normal_coverage_bed is not None:
        collect_normal_coverage = False
    else:
        print(
            "Normal coverage will not be used as a covariate; ability to regress out germline CNVs may suffer."
        )
        use_normal_coverage = False

    if tumor_coverage_bed is not None:
        collect_tumor_coverage = False

    #
    # collect or load coverage
    # tumor
    if collect_tumor_coverage:
        primary_contigs = ["chr{}".format(i) for i in range(1, 23)]
        primary_contigs.extend(["chrX", "chrY", "chrM"])
        # create scatter intervals
        split_intervals_task = split_intervals.split_intervals(
            bam=tumor_bam_localization_task["t_bam"],
            bai=tumor_bam_localization_task["t_bai"],
            interval_type="bed",
            selected_chrs=primary_contigs,
        )

        # shim task to transform split_intervals files into subset parameters for covcollect task
        @prefect.task
        def interval_gather(interval_files):
            ints = []
            for f in interval_files:
                ints.append(
                    pd.read_csv(
                        f, sep="\t", header=None, names=["chr", "start", "end"]
                    )
                )
            # filter non-primary contigs
            primary_contigs = ["chr{}".format(i) for i in range(1, 23)]
            primary_contigs.extend(["chrX", "chrY", "chrM"])
            full_bed = pd.concat(ints).sort_values(["chr", "start", "end"])
            filtered_bed = full_bed.loc[full_bed.chr.isin(primary_contigs)]
            return filtered_bed

        subset_intervals = interval_gather(
            split_intervals_task["interval_files"]
        )

        # dispatch coverage scatter
        tumor_cov_collect_task = cov_collect.Covcollect(
            inputs=dict(
                bam=tumor_bam_localization_task["t_bam"],
                bai=tumor_bam_localization_task["t_bai"],
                intervals=target_list,
                subset_chr=subset_intervals["chr"],
                subset_start=subset_intervals["start"],
                subset_end=subset_intervals["end"],
            )
        )

        # gather tumor coverage
        tumor_cov_gather_task = wolf.Task(
            name="gather_coverage",
            inputs={"coverage_beds": [tumor_cov_collect_task["coverage"]]},
            script="""cat $(cat ${coverage_beds}) > coverage_cat.bed""",
            outputs={"coverage": "coverage_cat.bed"},
        )
    else:
        tumor_cov_gather_task = {"coverage": tumor_coverage_bed}
    # load from supplied BED file
    ### coverage tasks ####

    # prepare coverage MCMC
    prep_cov_mcmc_task = hapaseg.Hapaseg_prepare_coverage_mcmc(
        inputs={
            "coverage_csv": tumor_cov_gather_task[
                "coverage"
            ],  # each scatter result is the same
            "allelic_clusters_object": "/mnt/nfs/workspace/hg38_e89211cf-b7a1-474b-9acb-78e83be42f13_d360fc68-e36c-49a4-8856-e3f9317e9b90/Hapaseg_collect_adp__2022-03-09--21-23-34_yaikkxa_xkoahjy_o5fx22guvr2gi/jobs/0/workspace/full_dp_results.npz",
            "SNPs_pickle": "/mnt/nfs/workspace/hg38_e89211cf-b7a1-474b-9acb-78e83be42f13_d360fc68-e36c-49a4-8856-e3f9317e9b90/Hapaseg_allelic_DP__2022-03-09--20-59-03_bubnria_xkoahjy_qx055rjjklgli/jobs/0/workspace/all_SNPs.pickle",  # each scatter result is the same
            "repl_pickle": localization_task["repl_file"],
            "ref_file_path": localization_task["ref_fasta"],
        }
    )

    # get the cluster indices from the preprocess data and generate the burnin indices
    @prefect.task(nout=4)
    def _get_ADP_cluster_list(preprocess_data_obj):
        range_size = 2000
        data = np.load(preprocess_data_obj)

        Pi = data["Pi"]
        r = data["r"]
        C = data["C"]

        num_clusters = Pi.shape[1]

        c_assignments = np.argmax(Pi, axis=1)
        cluster_list = []
        range_list = []

        # iterate through clusters and generate ranges
        for i in range(num_clusters):
            cluster_mask = c_assignments == i
            clust_size = len(r[cluster_mask])
            for j in range(int(np.ceil(clust_size / range_size))):
                cluster_list.append(i)
                range_list.append(
                    "{}-{}".format(
                        j * range_size, min((j + 1) * range_size, clust_size)
                    )
                )

        # also return a plain list of indices for the post-burnin run
        cluster_idxs = [i for i in np.arange(num_clusters)]
        print(cluster_idxs, cluster_list, range_list)
        return len(cluster_idxs), cluster_idxs, cluster_list, range_list

    num_clusters, cluster_idxs, cluster_list, range_list = (
        _get_ADP_cluster_list(prep_cov_mcmc_task["preprocess_data"])
    )

    # coverage MCMC burnin
    cov_mcmc_burnin_task = hapaseg.Hapaseg_coverage_mcmc_burnin(
        inputs={
            "preprocess_data": prep_cov_mcmc_task["preprocess_data"],
            "num_draws": 10,
            "cluster_num": cluster_list,
            "bin_width": bin_width,
            "range": range_list,
        }
    )
    num_cov_samples = 5  # TODO move to workflow args

    # coverage MCMC scatter post-burnin
    cov_mcmc_scatter_task = hapaseg.Hapaseg_coverage_mcmc(
        inputs={
            "preprocess_data": prep_cov_mcmc_task["preprocess_data"],
            "num_draws": num_cov_samples,
            "cluster_num": cluster_idxs,
            "bin_width": bin_width,
            "burnin_files": [cov_mcmc_burnin_task["burnin_data"]]
            * num_clusters,  # this is to account for a wolf input len bug
        }
    )

    # collect coverage MCMC
    cov_mcmc_gather_task = hapaseg.Hapaseg_collect_coverage_mcmc(
        inputs={
            "cov_mcmc_files": [cov_mcmc_scatter_task["cov_segmentation_data"]],
            "cov_df_pickle": prep_cov_mcmc_task["cov_df_pickle"],
            "bin_width": bin_width,
        }
    )
    # coverage DP
    cov_dp_task = hapaseg.Hapaseg_coverage_dp(
        inputs={
            "f_cov_df": prep_cov_mcmc_task["cov_df_pickle"],
            "cov_mcmc_data": cov_mcmc_gather_task["cov_collected_data"],
            "num_segmentation_samples": num_cov_samples,
            "num_dp_samples": 5,
            "sample_idx": list(range(num_cov_samples)),
            "bin_width": bin_width,
        }
    )

    # get the adp draw number from the preprocess data object
    @prefect.task
    def _get_ADP_draw_num(preprocess_data_obj):
        return int(np.load(preprocess_data_obj)["adp_cluster"])

    adp_draw_num = _get_ADP_draw_num(prep_cov_mcmc_task["preprocess_data"])

    # generate acdp dataframe

    gen_acdp_task = hapaseg.Hapaseg_acdp_generate_df(
        inputs={
            "SNPs_pickle": "/mnt/nfs/workspace/hg38_e89211cf-b7a1-474b-9acb-78e83be42f13_d360fc68-e36c-49a4-8856-e3f9317e9b90/Hapaseg_allelic_DP__2022-03-09--20-59-03_bubnria_xkoahjy_qx055rjjklgli/jobs/0/workspace/all_SNPs.pickle",
            "allelic_clusters_object": "/mnt/nfs/workspace/hg38_e89211cf-b7a1-474b-9acb-78e83be42f13_d360fc68-e36c-49a4-8856-e3f9317e9b90/Hapaseg_collect_adp__2022-03-09--21-23-34_yaikkxa_xkoahjy_o5fx22guvr2gi/jobs/0/workspace/full_dp_results.npz",
            "cdp_filepaths": [cov_dp_task["cov_dp_object"]],
            "allelic_draw_index": adp_draw_num,
            "ref_file_path": localization_task["ref_fasta"],
            "bin_width": bin_width,
        }
    )

    # run acdp
    acdp_task = hapaseg.Hapaseg_run_acdp(
        inputs={
            "coverage_dp_object": cov_dp_task["cov_dp_object"][0],
            "acdp_df": gen_acdp_task["acdp_df_pickle"],
            "num_samples": num_cov_samples,
            "cytoband_file": localization_task["cytoband_file"],
        }
    )
