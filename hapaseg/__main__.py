import argparse
import dask.distributed as dd
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc

from capy import mut

from .load import HapasegReference
from .run_allelic_MCMC import AllelicMCMCRunner
from .allelic_MCMC import A_MCMC
from .run_coverage_MCMC import CoverageMCMCRunner, aggregate_clusters
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
    scatter = subparsers.add_parser("load", help="Load in phased VCF")
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

    ## DP (TODO: will include gather step)
    dp = subparsers.add_parser("dp", help="Run DP clustering on allelic imbalance segments")

    ## coverage MCMC
    coverage_mcmc = subparsers.add_parser("coverage_mcmc", help="Run TCR segmentation on allelic imbalance clusters")
    coverage_mcmc.add_argument("--coverage_csv",
                               help="csv file containing '['chr', 'start', 'end', 'covcorr', 'covraw'] data")
    coverage_mcmc.add_argument("--allelic_clusters_object",
                               help="npy file containing allelic dp segs-to-clusters results")
    coverage_mcmc.add_argument("--SNPs_pickle", help="pickled dataframe containing SNPs")
    coverage_mcmc.add_argument("--covariate_dir",
                               help="path to covariate directory with covariates all in pickled files")
    coverage_mcmc.add_argument("--num_draws", type=int,
                               help="number of draws to take from coverage segmentation MCMC", default=50)
    coverage_mcmc.add_argument("--cluster_num", type=int,
                               help="cluster index for this worker to run on. If unspecified method will simulate "
                                    "all clusters on the same machine", default=None)
    coverage_mcmc.add_argument("--allelic_sample", type=int,
                               help="index of sample clustering from allelic DP to use as seed for segmentation",
                               default=None)

    ## collect coverage MCMC shards
    collect_cov_mcmc = subparsers.add_parser("collect_cov_mcmc", help="collect sharded cov mcmc results")
    collect_cov_mcmc.add_argument("--coverage_dir", help="path to the directory containing the coverage mcmc results")

    ## Coverage DP
    coverage_dp = subparsers.add_parser("coverage_dp", help="Run DP clustering on coverage segmentations")
    coverage_dp.add_argument("--f_cov_df", help="path to saved filtered coverage dataframe")
    coverage_dp.add_argument("--cov_mcmc_data", help="path to numpy savez file containing bins to segments array and global beta")
    coverage_dp.add_argument("--num_segmentation_samples", type=int, help="number of segmentation samples to use")
    coverage_dp.add_argument("--num_draws", type=int,
                             help="number of thinned draws from the coverage dp to take after burn in")

    ## Allelic Coverage DP

    # generate df
    gen_acdp_df = subparsers.add_parser("generate_acdp_df", help="generate dataframe for acdp clustering")

    gen_acdp_df.add_argument("--snp_dataframe", help="path to dataframe containing snps")
    gen_acdp_df.add_argument("--coverage_dp_object", help="path to coverage DP output object")
    gen_acdp_df.add_argument("--allelic_clusters_object", help="npy file containing allelic dp segs-to-clusters results")
    gen_acdp_df.add_argument("--allelic_draw_index", help="index of ADP draw used for coverage MCMC", type=int, default=-1)
    
    # run acdp clustering 
    ac_dp = subparsers.add_parser("allelic_coverage_dp", help="Run DP clustering on allelic coverage tuples")
    ac_dp.add_argument("--coverage_dp_object", help="path to coverage DP output object")
    ac_dp.add_argument("--acdp_df_path", help="path to acdp dataframe")
    ac_dp.add_argument("--num_samples", type=int, help="number of samples to take")
    ac_dp.add_argument("--cytoband_dataframe", help="path to dataframe containing cytoband information")

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
    output_dir = os.path.realpath(args.output_dir)

    if args.command == "run":
        dask_client = dd.Client(n_workers=int(args.n_workers))

        refs = HapasegReference(
            phased_VCF=args.phased_VCF,
            readbacked_phased_VCF=args.read_backed_phased_VCF,
            allele_counts=args.allele_counts_T,
            allele_counts_N=args.allele_counts_N,
            cytoband_file=args.cytoband_file
        )

        runner = AllelicMCMCRunner(
            refs.allele_counts,
            refs.chromosome_intervals,
            dask_client,
            phase_correct=args.phase_correct
        )

        allelic_segs = runner.run_all()

        # TODO: checkpoint here
        allelic_segs.to_pickle(output_dir + "/allelic_imbalance_segments.pickle")

        # TODO: save per-chromosome plots of raw allelic segmentations

    elif args.command == "load":
        # load from VCF
        refs = HapasegReference(
            phased_VCF=args.phased_VCF,
            readbacked_phased_VCF=args.read_backed_phased_VCF,
            allele_counts=args.allele_counts_T,
            allele_counts_N=args.allele_counts_N,
            cytoband_file=args.cytoband_file
        )

        # create chunks
        t = mut.map_mutations_to_targets(refs.allele_counts, refs.chromosome_intervals, inplace=False)
        groups = t.groupby(t).apply(lambda x: [x.index.min(), x.index.max()]).to_frame(name="bdy")
        groups["ranges"] = groups["bdy"].apply(lambda x: np.r_[x[0]:x[1]:args.chunk_size, x[1]])
        chunks = pd.DataFrame(
            np.vstack([
                np.hstack(np.broadcast_arrays(k, np.c_[y[0:-1], y[1:]]))
                for k, y in groups["ranges"].iteritems()
            ])[:, 1:],
            columns=["arm", "start", "end"]
        )

        # save to disk
        refs.allele_counts.to_pickle(output_dir + "/allele_counts.pickle")
        refs.chromosome_intervals.to_pickle(output_dir + "/chrom_int.pickle")
        chunks.to_csv(output_dir + "/scatter_chunks.tsv", sep="\t", index=False)

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

    elif args.command == "coverage_mcmc":
        cov_mcmc_runner = CoverageMCMCRunner(args.coverage_csv,
                                             args.allelic_clusters_object,
                                             args.SNPs_pickle,
                                             None, # no dask client for now
                                             args.covariate_dir,
                                             args.num_draws,
                                             args.cluster_num,
                                             args.allelic_sample)

        seg_samples, beta, mu_i_samples, filtered_cov_df = cov_mcmc_runner.run()

        #TODO make method for concatenating results from each cluster
        #save_results
        if args.cluster_num is not None and seg_samples is not None:
            # if its a single cluster that was not skipped, save the results to a new coverage dir
            coverage_dir = os.path.join(output_dir, 'coverage_mcmc_clusters')
            figure_dir = os.path.join(coverage_dir, 'figures')

            if not os.path.isdir(coverage_dir):
                os.mkdir(coverage_dir)
            
            if not os.path.isdir(figure_dir):
                os.mkdir(figure_dir)

            with open(os.path.join(coverage_dir,
                                   'cov_mcmc_model_cluster_{}.pickle'.format(args.cluster_num)), 'wb') as f:
                pickle.dump(cov_mcmc_runner.model, f)

            np.savez(os.path.join(coverage_dir, 'cov_mcmc_data_cluster_{}'.format(args.cluster_num)),
                     seg_samples=seg_samples, beta=beta, mu_i_samples=mu_i_samples)
            if args.cluster_num == 0:
                # only need one copy of this
                filtered_cov_df.to_pickle(os.path.join(coverage_dir, 'cov_df.pickle'))
            
            # save visualization
            cov_mcmc_runner.model.visualize_cluster_samples(os.path.join(figure_dir, 'cov_mcmc_cluster_{}_visual'.format(args.cluster_num)))
        else:
            with open(os.path.join(output_dir, 'cov_mcmc_model.pickle'), 'wb') as f:
                pickle.dump(cov_mcmc_runner.model, f)

            np.savez(os.path.join(output_dir, 'cov_mcmc_data'),
                     seg_samples=seg_samples, beta=beta, mu_i_samples=mu_i_samples)
            filtered_cov_df.to_pickle(os.path.join(output_dir, 'cov_df.pickle'))
    
    elif args.command == "collect_cov_mcmc":
        full_segmentation = aggregate_clusters(args.coverage_dir)
        
        #load beta from one of the clusters
        cov_data = np.load(os.path.join(args.coverage_dir, 'cov_mcmc_data_cluster_0.npz'))
        beta = cov_data['beta']

        ## save these results to new aggregated file
        np.savez(os.path.join(args.coverage_dir, 'cov_mcmc_collected_data'), seg_samples=full_segmentation, beta=beta)

    elif args.command == "coverage_dp":
        cov_df = pd.read_pickle(args.f_cov_df)
        mcmc_data = np.load(args.cov_mcmc_data)
        segmentation_samples = mcmc_data['seg_samples']
        beta = mcmc_data['beta']

        cov_dp_runner = Coverage_DP(segmentation_samples, beta, cov_df)

        cov_dp_runner.run_dp(args.num_segmentation_samples, args.num_draws)
        with open(args.output_dir + f"/Cov_DP_model.pickle", "wb") as f:
            pickle.dump(cov_dp_runner, f)
   
        #save visualization
        # first make sure the directory exists
        figure_dir = os.path.join(output_dir, 'coverage_figures')
        if not os.path.isdir(figure_dir):
            os.mkdir(figure_dir)

        cov_dp_runner.visualize_DP_run(args.num_segmentation_samples - 1, os.path.join(figure_dir, 'coverage_draw_{}'.format(args.num_draws -1)))
    
    elif args.command == "generate_acdp_df":
        acdp_df, beta = generate_acdp_df(args.snp_dataframe,
                               args.coverage_dp_object,
                               args.allelic_clusters_object,
                               args.allelic_draw_index)
        acdp_df.to_pickle(os.path.join(output_dir, "acdp_df.pickle"))
 
    elif args.command == "allelic_coverage_dp":
        acdp_df = pd.read_pickle(args.acdp_df_path)
        #may want to switch this to a preferred method of beta loading
        with open(args.coverage_dp_object, "rb") as f:
            cdp_pickle = pickle.load(f)
        beta = cdp_pickle.beta
        acdp = AllelicCoverage_DP(acdp_df, beta, args.cytoband_dataframe)
        acdp.run(args.num_samples)
        acdp.visualize_ACDP(output_dir)
        
        
        with open(os.path.join(output_dir, "acdp_model.pickle"), "wb") as f:
            pickle.dump(acdp, f)

if __name__ == "__main__":
    main()
