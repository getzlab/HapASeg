import argparse
import dask.distributed as dd
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc

from capy import mut

from .load import HapasegSNPs
from .run_allelic_MCMC import AllelicMCMCRunner
from .allelic_MCMC import A_MCMC

from .allelic_DP import A_DP, DPinstance
from . import utils as hs_utils

from .NB_coverage_MCMC import NB_MCMC_SingleCluster
from .run_coverage_MCMC import CoverageMCMCRunner, aggregate_clusters, aggregate_burnin_files 
from .coverage_DP import Coverage_DP
from .a_cov_DP import generate_acdp_df, AllelicCoverage_DP


def parse_args():
    parser = argparse.ArgumentParser(description="Call somatic copynumber alterations taking advantage of SNP phasing")

    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--capy_ref_path", default="/home/opriebe/data/ref/hg19/Homo_sapiens_assembly19.fasta")
    subparsers = parser.add_subparsers(dest="command")

    ## run
    standalone = subparsers.add_parser("run", help="Run HapASeg as a standalone module")
    standalone.add_argument("--n_workers", default=multiprocessing.cpu_count() - 1)
    standalone.add_argument("--n_iter", default=20000)

    input_group = standalone.add_mutually_exclusive_group(
        required=True
        # "Mutually exclusive inputs",
        # "Hapaseg can either take an already phased VCF annotated with alt/refcounts, or a MuTect callstats file. In the case of the latter, Hapaseg will perform phasing.",
    )
    input_group.add_argument("--phased_VCF")
    input_group.add_argument("--callstats_file")

    # if we are performing phasing ourselves, we need these reference files
    phasing_ref_group = standalone.add_argument_group(
        "Phasing ref. files",
        "Required files for Hapaseg to pull down het sites and impute phasing.",
    )
    phasing_ref_group.add_argument("--phasing_ref_panel_dir")
    phasing_ref_group.add_argument("--phasing_genetic_map_file")
    phasing_ref_group.add_argument("--SNP_list")
    phasing_ref_group.add_argument("--ref_fasta")
    phasing_ref_group.add_argument("--bam", help="BAM to use for read-backed phasing")
    phasing_ref_group.add_argument("--bai", help="BAI to use for read-backed phasing")

    # if we are taking a pre-phased VCF, we also may want a read-backed phased VCF
    prephased_group = standalone.add_argument_group(
        "Inputs if phasing has already been imputed",
    )
    prephased_group.add_argument("--read_backed_phased_VCF", help="Optional.")
    prephased_group.add_argument("--allele_counts_T", help="Required.")
    prephased_group.add_argument("--allele_counts_N", help="Required.")

    ai_seg_params = standalone.add_argument_group(
        "Parameters for allelic imbalance segmentation",
    )
    ai_seg_params.add_argument("--phase_correct", action="store_true")
    ai_seg_params.add_argument("--misphase_prior", default="0.001")

    ref_group = standalone.add_argument_group(
        "Required reference files",
    )
    ref_group.add_argument("--cytoband_file", required=True)

    ## load
    scatter = subparsers.add_parser("load_snps", help="Load in phased VCF")
    scatter.add_argument("--chunk_size", default=5000)
    scatter.add_argument("--phased_VCF", required=True)

    scatter.add_argument("--read_backed_phased_VCF")
    scatter.add_argument("--allele_counts_T", required=True)
    scatter.add_argument("--allele_counts_N", required=True)
    scatter.add_argument("--cytoband_file", required=True)

    ## amcmc
    amcmc = subparsers.add_parser("amcmc", help="Run allelic MCMC on a range of SNPs")

    input_group = amcmc.add_mutually_exclusive_group(
        required=True
    )
    input_group.add_argument("--snp_dataframe")
    input_group.add_argument("--amcmc_object")

    amcmc.add_argument("--start", default=0)
    amcmc.add_argument("--end", default=-1)
    amcmc.add_argument("--stop_after_burnin", action="store_true")
    amcmc.add_argument("--ref_bias", default=1.0)
    amcmc.add_argument("--n_iter", default=20000)

    ## concat
    concat = subparsers.add_parser("concat", help="Concatenate burned-in chunks")
    concat.add_argument("--chunks", required=True, nargs="+")
    concat.add_argument("--scatter_intervals", required=True)

    ## concat arms
    concat_arms = subparsers.add_parser("concat_arms", help="concat arm segs")
    concat_arms.add_argument("--arm_results", help="file containing paths to arm mcmc objects",
                            required=True)
    ## DP
    dp = subparsers.add_parser("dp", help="Run DP clustering on allelic imbalance segments")
    dp.add_argument("--seg_dataframe", required=True)
    dp.add_argument("--n_dp_iter", default=10)
    dp.add_argument("--seg_samp_idx", default=0)
    dp.add_argument("--ref_fasta",
                    required=True)  # TODO: only useful for chrpos->gpos; will be removed when this is passed from load
    dp.add_argument("--cytoband_file",
                    required=True)  # TODO: only useful for chrpos->gpos; will be removed when this is passed from load
  
    ## coverage MCMC
    coverage_mcmc = subparsers.add_parser("coverage_mcmc",
                                          help="Run TCR segmentation on all allelic imbalance clusters")
    coverage_mcmc.add_argument("--coverage_csv",
                               help="csv file containing '['chr', 'start', 'end', 'covcorr', 'covraw'] data")
    coverage_mcmc.add_argument("--allelic_clusters_object",
                               help="npy file containing allelic dp segs-to-clusters results")
    coverage_mcmc.add_argument("--SNPs_pickle", help="pickled dataframe containing SNPs")
    coverage_mcmc.add_argument("--covariate_dir",
                               help="path to covariate directory with covariates all in pickled files")
    coverage_mcmc.add_argument("--num_draws", type=int,
                               help="number of draws to take from coverage segmentation MCMC", default=50)
    coverage_mcmc.add_argument("--allelic_sample", type=int,
                               help="index of sample clustering from allelic DP to use as seed for segmentation",
                               default=None)

    #collect ADP data
    collect_adp = subparsers.add_parser("collect_adp", help="collect ADP resuts from shards")
    collect_adp.add_argument("--dp_results", help="path to txt file with paths to dp results")

    ## proprocessing coverage MCMC data for scatter tasks
    preprocess_coverage_mcmc = subparsers.add_parser("coverage_mcmc_preprocess",
                                                     help="Preform preprocessing on ADP results to allow for coverage mcmc scatter jobs")
    preprocess_coverage_mcmc.add_argument("--coverage_csv",
                                          help="csv file containing '['chr', 'start', 'end', 'covcorr', 'covraw'] data", required=True)
    preprocess_coverage_mcmc.add_argument("--allelic_clusters_object",
                                          help="npy file containing allelic dp segs-to-clusters results", required=True)
    preprocess_coverage_mcmc.add_argument("--SNPs_pickle", help="pickled dataframe containing SNPs", required=True)
    preprocess_coverage_mcmc.add_argument("--repl_pickle", help="pickled dataframe containing replication timing data", required=True)
    preprocess_coverage_mcmc.add_argument("--gc_pickle", help="pickled dataframe containing precomputed gc content. This is not required but will speed up runtime if passed", default=None)
    preprocess_coverage_mcmc.add_argument("--allelic_sample", type=int,
                                          help="index of sample clustering from allelic DP to use as seed for segmentation. Will use most likely clustering by default",
                                          default=None)

    ## running coverage mcmc on single cluster for scatter task
    coverage_mcmc_shard = subparsers.add_parser("coverage_mcmc_shard",
                                                help="run coverage mcmc on single ADP cluster")
    coverage_mcmc_shard.add_argument("--preprocess_data", help='path to numpy object containing preprocessed data',
                                     required=True)
    coverage_mcmc_shard.add_argument("--num_draws", type=int,
                               help="number of draws to take from coverage segmentation MCMC", default=50)
    coverage_mcmc_shard.add_argument("--cluster_num", type=int,
                               help="cluster index for this worker to run on. If unspecified method will simulate "
                                    "all clusters on the same machine", default=None)
    coverage_mcmc_shard.add_argument("--bin_width", type=int, default=1, help="size of uniform bins if using. Otherwise 1.")
    coverage_mcmc_shard.add_argument("--range", type=str, help="range of coverage bins within the cluster to burnin. should be in start-end form. Note that this will cause num draws to be overridden to 1")
    coverage_mcmc_shard.add_argument("--burnin_files", type=str, help="txt file containing burnt in segment assignments")

    ## collect coverage MCMC shards
    collect_cov_mcmc = subparsers.add_parser("collect_cov_mcmc", help="collect sharded cov mcmc results")
    collect_cov_mcmc.add_argument("--coverage_dir", help="path to the directory containing the coverage mcmc results")
    collect_cov_mcmc.add_argument("--cov_mcmc_files",
                                  help="path to txt file with each line containing a path to a cov mcmc shard result")
    collect_cov_mcmc.add_argument("--cov_df_pickle",
                                  help="path to cov_df pickle file. Required for using --cov_mcmc_files option")
    collect_cov_mcmc.add_argument("--bin_width", type=int, help="size of uniform bins if using. otherwise 1")

    ## Coverage DP
    coverage_dp = subparsers.add_parser("coverage_dp", help="Run DP clustering on coverage segmentations")
    coverage_dp.add_argument("--f_cov_df", help="path to saved filtered coverage dataframe")
    coverage_dp.add_argument("--cov_mcmc_data",
                             help="path to numpy savez file containing bins to segments array and global beta")
    coverage_dp.add_argument("--num_segmentation_samples", type=int, help="number of segmentation samples to use")
    coverage_dp.add_argument("--sample_idx", type=int, help="index of segmentation draw to use if scattering over segmentation draws")
    coverage_dp.add_argument("--num_dp_samples", type=int,
                             help="number of thinned draws from the coverage dp to take after burn in")
    coverage_dp.add_argument("--bin_width", type=int, default=1, help="size of uniform bins if using. Otherwise 1.")

    ## Allelic Coverage DP

    # generate df
    gen_acdp_df = subparsers.add_parser("generate_acdp_df", help="generate dataframe for acdp clustering")

    gen_acdp_df.add_argument("--snp_dataframe", help="path to dataframe containing snps")
    gen_acdp_df.add_argument("--cdp_object", help="path to coverage DP output object")
    gen_acdp_df.add_argument("--cdp_filepaths", help="paths of scattered DP objects")
    gen_acdp_df.add_argument("--allelic_clusters_object",
                             help="npy file containing allelic dp segs-to-clusters results")
    gen_acdp_df.add_argument("--allelic_draw_index", help="index of ADP draw used for coverage MCMC", type=int, default=-1)
    gen_acdp_df.add_argument("--bin_width", help="size of uniform bins if using. Otherwise 1", default=1, type=int)

    # run acdp clustering 
    ac_dp = subparsers.add_parser("allelic_coverage_dp", help="Run DP clustering on allelic coverage tuples")
    ac_dp.add_argument("--coverage_dp_object", help="path to coverage DP output object")
    ac_dp.add_argument("--acdp_df_path", help="path to acdp dataframe")
    ac_dp.add_argument("--num_samples", type=int, help="number of samples to take")
    ac_dp.add_argument("--cytoband_file", help="path to cytoband txt file")
    ac_dp.add_argument("--warmstart", type=bool, default=True, help="run clustering with warmstart")
    args = parser.parse_args()

    # validate arguments
    #    if args.phased_VCF is not None:
    #        if args.allele_counts_T is None or args.allele_counts_N is None:
    #            raise ValueError("Must supply allele counts from HetPulldown if not imputing phasing")
    #    # TODO: if callstats_file is given, then all phasing_ref_group args must be present

    return args


def main():
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.isdir(args.output_dir + "/figures"):
        os.mkdir(args.output_dir + "/figures")
    output_dir = os.path.realpath(args.output_dir)

    if args.command == "run":
        dask_client = dd.Client(n_workers=int(args.n_workers))

        snps = HapasegSNPs(
            phased_VCF=args.phased_VCF,
            readbacked_phased_VCF=args.read_backed_phased_VCF,
            allele_counts=args.allele_counts_T,
            allele_counts_N=args.allele_counts_N
        )

        runner = AllelicMCMCRunner(
            snps.allele_counts,
            snps.chromosome_intervals,
            dask_client,
            phase_correct=args.phase_correct
        )

        allelic_segs = runner.run_all()

        # TODO: checkpoint here
        allelic_segs.to_pickle(output_dir + "/allelic_imbalance_segments.pickle")

        # TODO: save per-chromosome plots of raw allelic segmentations

    elif args.command == "load_snps":
        # load from VCF
        snps = HapasegSNPs(
            phased_VCF=args.phased_VCF,
            readbacked_phased_VCF=args.read_backed_phased_VCF,
            allele_counts=args.allele_counts_T,
            allele_counts_N=args.allele_counts_N
        )

        # create chunks
        chromosome_intervals = hs_utils.parse_cytoband(args.cytoband_file)

        t = mut.map_mutations_to_targets(snps.allele_counts, chromosome_intervals, inplace=False)
        # ranges of SNPs spanned by each arm
        groups = t.groupby(t).apply(lambda x: [x.index.min(), x.index.max()]).to_frame(name="bdy")
        # remove arms with too few SNPs (e.g. acrocentric chromosomes)
        groups = groups.loc[groups["bdy"].apply(np.diff) > 5]  # XXX: is 5 SNPs too small?

        # chunk arms
        groups["ranges"] = groups["bdy"].apply(lambda x: np.r_[x[0]:x[1]:args.chunk_size, x[1]])

        # for arms with > 1 chunk, merge the last chunk with penultimate chunk if last chunk is short
        short_last_chunks = groups.loc[
            (groups["ranges"].apply(len) > 2) & \
            (groups["ranges"].apply(lambda x: x[-1] - x[-2]) < args.chunk_size * 0.05),
            "ranges"
        ]
        groups.loc[short_last_chunks.index, "ranges"] = short_last_chunks.apply(lambda x: np.r_[x[:-2], x[-1]])

        # create chunk scatter dataframe
        chunks = pd.DataFrame(
            np.vstack([
                np.hstack(np.broadcast_arrays(k, np.c_[y[0:-1], y[1:]]))
                for k, y in groups["ranges"].iteritems()
            ])[:, 1:],
            columns=["arm", "start", "end"]
        )

        # save to disk
        snps.allele_counts.to_pickle(output_dir + "/allele_counts.pickle")
        chunks.to_csv(output_dir + "/scatter_chunks.tsv", sep="\t", index=False)

    elif args.command == "load_coverage":
        pass

    elif args.command == "amcmc":
        # loading from SNP dataframe produced by `hapaseg load`
        if args.snp_dataframe is not None:
            P = pd.read_pickle(args.snp_dataframe)
            H = A_MCMC(
                P.iloc[int(args.start):int(args.end)],
                quit_after_burnin=args.stop_after_burnin,
                ref_bias=float(args.ref_bias),
                n_iter=int(args.n_iter)
            )

        # loading from allelic MCMC results object produced by `hapaseg amcmc`
        else:
            with open(args.amcmc_object, "rb") as f:
                H = pickle.load(f)

            # update some class properties set in the constructor
            H._set_ref_bias(float(args.ref_bias))
            H.n_iter = int(args.n_iter)
            H.quit_after_burnin = args.stop_after_burnin

            if len(H.marg_lik) > H.n_iter:
                H.marg_lik = H.marg_lik[:H.n_iter]

        with open(output_dir + "/amcmc_results.pickle", "wb") as f:
            pickle.dump(H.run(), f)

    elif args.command == "concat":
        #
        # load scatter intervals
        intervals = pd.read_csv(args.scatter_intervals, sep="\t")

        if len(intervals) != len(args.chunks):
            raise ValueError("Length mismatch in supplied chunks and interval file!")

        # load results
        R = []
        for chunk_path in args.chunks:
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            R.append(chunk)
        R = pd.DataFrame({"results": R})

        # ensure results are in the correct order
        R["first"] = R["results"].apply(lambda x: x.P.loc[0, "index"])
        R = R.sort_values("first", ignore_index=True)

        # concat with intervals
        R = pd.concat([R, intervals], axis=1).drop(columns=["first"])

        #
        # compute reference bias
        X = []
        j = 0
        for chunk in R["results"]:
            bpl = np.array(chunk.breakpoints);
            bpl = np.c_[bpl[0:-1], bpl[1:]]

            for i, (st, en) in enumerate(bpl):
                x = chunk.P.iloc[st:en].groupby("allele_A")[["MIN_COUNT", "MAJ_COUNT"]].sum()
                x["idx"] = i + j
                X.append(x)

            j += len(chunk.P)

        X = pd.concat(X)
        g = X.groupby("idx").size() == 2
        Y = X.loc[X["idx"].isin(g[g].index)]

        f = np.zeros([len(Y) // 2, 100])
        l = np.zeros([len(Y) // 2, 2])
        for i, (_, g) in enumerate(Y.groupby("idx")):
            f[i, :] = s.beta.rvs(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1, size=100) / s.beta.rvs(
                g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1, size=100)
            l[i, :] = np.r_[
                ss.betaln(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1),
                ss.betaln(g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1),
            ]

        # weight mean by negative log marginal likelihoods
        # take smaller of the two likelihoods to account for power imbalance
        w = np.min(-l, 1, keepdims=True)

        ref_bias = (f * w).sum() / (100 * w.sum())

        with open(output_dir + "/ref_bias.txt", "w") as f:
            f.write(str(ref_bias))

        #
        # concat burned in chunks for each arm
        for arm, Ra in R.groupby("arm"):
            A = A_MCMC(
                pd.concat([x.P for x in Ra["results"]], ignore_index=True),
                # other class properties will be filled in with their correct values later
            )

            # replicate constructor steps to define initial breakpoint set and
            # marginal likelihood dict
            breakpoints = [None] * len(Ra)
            A.seg_marg_liks = sc.SortedDict()
            for j, Ras in enumerate(Ra.itertuples()):
                start = Ras.start - Ra["start"].iloc[0]
                breakpoints[j] = np.array(Ras.results.breakpoints) + start
                for k, v in Ras.results.seg_marg_liks.items():
                    A.seg_marg_liks[k + start] = v
            A.breakpoints = sc.SortedSet(np.hstack(breakpoints))

            A.marg_lik = np.full(A.n_iter, np.nan)  # n_iter and size of this array will be reset later
            A.marg_lik[0] = np.array(A.seg_marg_liks.values()).sum()

            with open(output_dir + f"/AMCMC-arm{arm}.pickle", "wb") as f:
                pickle.dump(A, f)
    
    elif args.command == "concat_arms":
        with open(args.arm_results, 'r') as f:
	        arm_results = f.readlines()
        
        A = []
        for arm_file in arm_results:
            with open(arm_file.rstrip('\n'), "rb") as f:
                H = pickle.load(f)
            A.append(pd.Series({ "chr" : H.P["chr"].iloc[0], "start" : H.P["pos"].iloc[0], "end" : H.P["pos"].iloc[-1], "results" : H }))

        # get into order
        A = pd.concat(A, axis = 1).T.sort_values(["chr", "start", "end"]).reset_index(drop = True)

        # save
        A.to_pickle(os.path.join(output_dir, "arm_results.pickle"))

        # get number of MCMC samples
        n_samps = int(np.minimum(np.inf, A.loc[~A["results"].isna(), "results"].apply(lambda x : len(x.breakpoint_list))).min())

        np.savez(os.path.join(output_dir, "num_arm_samples"), n_samps = n_samps)
        
    elif args.command == "dp":
        # load allelic segmentation samples
        A = A_DP(args.seg_dataframe, ref_fasta=args.ref_fasta)

        # run DP
        # TODO: when we have better type checking, drop the int coersion here
        # N_seg_samps = A.n_samp - 1 if int(args.n_seg_samps) == 0 else int(args.n_seg_samps)
        # TODO: if we decide to drop support for chained sampling altogether, remove N_seg_samps logic altogether
        snps_to_clusters, snps_to_phases, likelihoods = A.run(
            seg_sample_idx=int(args.seg_samp_idx),
            # N_seg_samps = N_seg_samps,
            N_clust_samps=int(args.n_dp_iter)
        )

        # save DP results
        np.savez(output_dir + "/allelic_DP_SNP_clusts_and_phase_assignments.npz",
                 snps_to_clusters=snps_to_clusters,
                 snps_to_phases=snps_to_phases,
                 likelihoods=likelihoods
                 )

        A.SNPs.to_pickle(output_dir + "/all_SNPs.pickle")

        #
        # plot DP results

        # 1. phased SNP visualization
        f = plt.figure(figsize=[17.56, 5.67])
        hs_utils.plot_chrbdy(args.cytoband_file)
        A.visualize_SNPs(snps_to_phases, color=True, f=f)
        A.visualize_clusts(snps_to_clusters, f=f, thick=True, nocolor=True)
        plt.ylabel("Haplotypic imbalance")
        plt.title("SNP phasing/segmentation")
        plt.savefig(output_dir + "/figures/SNPs.png", dpi=300)
        plt.close()

        # 2. pre-clustering segments
        f = plt.figure(figsize=[17.56, 5.67])
        hs_utils.plot_chrbdy(args.cytoband_file)
        A.visualize_SNPs(snps_to_phases, color=False, f=f)
        A.visualize_segs(snps_to_clusters, f=f)
        plt.ylabel("Haplotypic imbalance")
        plt.title("Allelic segmentation, pre-DP clustering")
        plt.savefig(output_dir + "/figures/allelic_imbalance_preDP.png", dpi=300)
        plt.close()

        # 3. post-clustering segments
        f = plt.figure(figsize=[17.56, 5.67])
        hs_utils.plot_chrbdy(args.cytoband_file)
        A.visualize_SNPs(snps_to_phases, color=False, f=f)
        A.visualize_clusts(snps_to_clusters, f=f, thick=True)
        plt.ylabel("Haplotypic imbalance")
        plt.title("Allelic segmentation, post-DP clustering")
        plt.savefig(output_dir + "/figures/allelic_imbalance_postDP.png", dpi=300)
        plt.close()
    
    #collect adp run data
    elif args.command == "collect_adp":
        with open(args.dp_results, 'r') as f:
	        dp_results = f.readlines()
        accum_clusts = []
        accum_phases = []
        accum_liks = []
        
        for dp_shard in dp_results:
            obj = np.load(dp_shard.rstrip('\n'))
            accum_clusts.append(obj['snps_to_clusters'])
            accum_phases.append(obj['snps_to_phases'])
            accum_liks.append(obj['likelihoods'])
        all_clusts = np.vstack(accum_clusts)
        all_phases = np.vstack(accum_phases)
        all_liks = np.vstack(accum_liks)
        # save
        np.savez(os.path.join(output_dir, "full_dp_results"), snps_to_clusters=all_clusts, snps_to_phases=all_phases, likelihoods=all_liks)


    ## running coverage mcmc on all clusters

    elif args.command == "coverage_mcmc":
        cov_mcmc_runner = CoverageMCMCRunner(args.coverage_csv,
                                             args.allelic_clusters_object,
                                             args.SNPs_pickle,
                                             args.covariate_dir,
                                             num_draws=args.num_draws,
                                             allelic_sample=args.allelic_sample)

        seg_samples, beta, mu_i_samples, filtered_cov_df = cov_mcmc_runner.run()

        # save_results
        with open(os.path.join(output_dir, 'cov_mcmc_model.pickle'), 'wb') as f:
            pickle.dump(cov_mcmc_runner.model, f)

        np.savez(os.path.join(output_dir, 'cov_mcmc_data'),
                 seg_samples=seg_samples, beta=beta, mu_i_samples=mu_i_samples)
        filtered_cov_df.to_pickle(os.path.join(output_dir, 'cov_df.pickle'))

    ## preprocess ADP data to run scattered coverage mcmc jobs on each ADP cluster
    elif args.command == "coverage_mcmc_preprocess":
        cov_mcmc_runner = CoverageMCMCRunner(args.coverage_csv,
                                             args.allelic_clusters_object,
                                             args.SNPs_pickle,
                                             f_repl=args.repl_pickle,
                                             f_GC=args.gc_pickle,
                                             allelic_sample=args.allelic_sample)
        Pi, r, C, all_mu, global_beta, cov_df, adp_cluster = cov_mcmc_runner.prepare_single_cluster()
        np.savez(os.path.join(output_dir, 'preprocess_data'), Pi=Pi, r=r, C=C, all_mu=all_mu,
                 global_beta=global_beta, adp_cluster=adp_cluster)
        cov_df.to_pickle(os.path.join(output_dir, 'cov_df.pickle'))

    ## run scattered coverage mcmc job using preprocessed data
    elif args.command == "coverage_mcmc_shard":
        # load preprocessed data
        preprocess_data = np.load(args.preprocess_data)
        # check to make sure that the cluster index is within the range
        Pi = preprocess_data['Pi']
        if args.cluster_num > Pi.shape[1] - 1:
            raise ValueError("Received cluster number {}, which is out of range".format(args.cluster_num))
        
        # extract preprocessed data from this cluster
        mu = preprocess_data["all_mu"][args.cluster_num]
        beta = preprocess_data["global_beta"]
        c_assignments = np.argmax(Pi, axis=1)
        cluster_mask = (c_assignments == args.cluster_num)
        r = preprocess_data['r'][cluster_mask]
        C = preprocess_data['C'][cluster_mask]
        
        # if we get a range argument well be doing burnin on a subset of the coverage bins
        if args.range is not None:
            #parse range from string
            range_lst = args.range.split('-')
            st,en = int(range_lst[0]), int(range_lst[1]) 
            if st > en or st < 0 or en > len(r):
                raise ValueError("invalid range! got range {} for cluster {} with size {}".format(args.range, args.cluster_num, len(r)))
            
            #trim data to our desired range
            r = r[st:en]
            C = C[st:en]
            num_draws = 1
            
            # if we're just burning in a subset use different save strings
            model_save_str = 'cov_mcmc_model_cluster_{}_{}.pickle'.format(args.cluster_num, args.range)
            data_save_str = 'cov_mcmc_data_cluster_{}_{}'.format(args.cluster_num, args.range)
            figure_save_str = 'cov_mcmc_cluster_{}_{}_visual'.format(args.cluster_num, args.range)
            
        else:
            #if not in burnin use the specified number of draws
            num_draws = args.num_draws
            
            
            model_save_str = 'cov_mcmc_model_cluster_{}.pickle'.format(args.cluster_num)
            data_save_str = 'cov_mcmc_data_cluster_{}'.format(args.cluster_num)
            figure_save_str = 'cov_mcmc_cluster_{}_visual'.format(args.cluster_num)
        
        # run on the specified cluster
        cov_mcmc = NB_MCMC_SingleCluster(num_draws, r, C, mu, beta, args.cluster_num, args.bin_width)
        
        # if we're using burnin results load them now
        if args.burnin_files is not None:
            with open(args.burnin_files, 'r') as f:
                file_list = f.read().splitlines()
            assignments_arr = aggregate_burnin_files(file_list, args.cluster_num)
            cov_mcmc.init_burnin(assignments_arr)

        cov_mcmc.run()

        # collect the results
        segment_samples, global_beta, mu_i_samples = cov_mcmc.prepare_results()
        
        # save samples
        with open(os.path.join(output_dir, model_save_str), 'wb') as f:
            pickle.dump(cov_mcmc, f)

        np.savez(os.path.join(output_dir, data_save_str),
                 seg_samples=segment_samples, beta=global_beta, mu_i_samples=mu_i_samples)

        # save visualization
        cov_mcmc.visualize_cluster_samples(
            os.path.join(output_dir, figure_save_str))

    elif args.command == "collect_cov_mcmc":
        if args.coverage_dir:
            full_segmentation, beta = aggregate_clusters(coverage_dir=args.coverage_dir, cov_df_pickle=args.cov_df_pickle)

        elif args.cov_mcmc_files:
            if args.cov_df_pickle is None:
                raise ValueError("cov_df_pickle argument required for passing shard file")
            full_segmentation, beta = aggregate_clusters(f_file_list=args.cov_mcmc_files, cov_df_pickle=args.cov_df_pickle, bin_width=args.bin_width)
        else:
            # need to pass in one or the other
            raise ValueError("must pass in either a directory or a txt file listing mcmc results")

        ## save these results to new aggregated file
        if args.coverage_dir:
            np.savez(os.path.join(args.coverage_dir, 'cov_mcmc_collected_data'), seg_samples=full_segmentation,
                     beta=beta)
        else:
            np.savez('./cov_mcmc_collected_data', seg_samples=full_segmentation, beta=beta)

    elif args.command == "coverage_dp":
        cov_df = pd.read_pickle(args.f_cov_df)
        mcmc_data = np.load(args.cov_mcmc_data)
        segmentation_samples = mcmc_data['seg_samples']
        beta = mcmc_data['beta']

        cov_dp_runner = Coverage_DP(segmentation_samples, beta, cov_df, args.bin_width)
        #print(args.sample_idx, args.num_segmentation_samples, args.num_dp_samples, flush=True) 
        if args.sample_idx is not None:
            
            cov_dp_runner.run_dp(1, args.num_dp_samples, sample_idx=args.sample_idx)
            model_save_path = "Cov_DP_model_{}.pickle".format(args.sample_idx)
            viz_sample = 0
        else:
            cov_dp_runner.run_dp(args.num_segmentation_samples, args.num_dp_samples)
            model_save_path = "Cov_DP_model.pickle"
            viz_sample = args.num_segmentation_samples - 1
        with open(os.path.join(args.output_dir, model_save_path), "wb") as f:
            pickle.dump(cov_dp_runner, f)

        # save visualization
        cov_dp_runner.visualize_DP_run(viz_sample,
                                       os.path.join(output_dir, 'cov_dp_visual_draw_{}'.format(args.num_dp_samples - 1)))

    elif args.command == "generate_acdp_df":
        if args.cdp_object is not None:
            #all of our dp runs are in one object
            acdp_df, beta = generate_acdp_df(args.snp_dataframe,
                                             args.allelic_clusters_object,
                                             cdp_object_path=args.cdp_object,
                                             bin_width=args.bin_width,
                                             ADP_draw_index=args.allelic_draw_index)
        
        if args.cdp_filepaths is not None:
            #all of our dp runs are in one object
            acdp_df, beta = generate_acdp_df(args.snp_dataframe,
                                             args.allelic_clusters_object,
                                             cdp_scatter_files=args.cdp_filepaths,
                                             bin_width=args.bin_width,
                                             ADP_draw_index=args.allelic_draw_index)
        
        acdp_df.to_pickle(os.path.join(output_dir, "acdp_df.pickle"))

    elif args.command == "allelic_coverage_dp":
        acdp_df = pd.read_pickle(args.acdp_df_path)
        # may want to switch this to a preferred method of beta loading
        with open(args.coverage_dp_object, "rb") as f:
            cdp_pickle = pickle.load(f)
        beta = cdp_pickle.beta
        acdp = AllelicCoverage_DP(acdp_df, beta, args.cytoband_file, args.warmstart)
        acdp.run(args.num_samples)
        print("visualizing run")
        acdp.visualize_ACDP(output_dir)

        with open(os.path.join(output_dir, "acdp_model.pickle"), "wb") as f:
            pickle.dump(acdp, f)


if __name__ == "__main__":
    main()
