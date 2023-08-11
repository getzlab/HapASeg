import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc
import sys
import traceback
import tqdm

from capy import mut, seq

from .load import HapasegSNPs
from .run_allelic_MCMC import AllelicMCMCRunner
from .allelic_MCMC import A_MCMC 

from .allelic_DP import A_DP, DPinstance, load_DP_object_from_outputs
from . import utils as hs_utils

from .coverage_MCMC import Coverage_MCMC_SingleCluster
from .run_coverage_MCMC import CoverageMCMCRunner, aggregate_clusters, aggregate_burnin_files 
from .coverage_DP import Coverage_DP
from .a_cov_DP import generate_acdp_df, AllelicCoverage_DP, AllelicCoverage_DP_runner


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
    scatter.add_argument("--chunk_size", default=10000)
    scatter.add_argument("--phased_VCF", required=True)

    scatter.add_argument("--read_backed_phased_VCF")
    scatter.add_argument("--allele_counts_T", required=True)
    scatter.add_argument("--allele_counts_N", default=None)
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
    amcmc.add_argument("--betahyp", default=-1)

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
    dp.add_argument("--seg_dataframe", required = True)
    dp.add_argument("--ref_fasta", required = True) # TODO: only useful for chrpos->gpos; will be removed when this is passed from load
    dp.add_argument("--cytoband_file", required = True)
    dp.add_argument("--wgs", action="store_true", default=False)
  
    ## coverage MCMC
    coverage_mcmc = subparsers.add_parser("coverage_mcmc",
                                          help="Run TCR segmentation on all allelic imbalance clusters")
    coverage_mcmc.add_argument("--coverage_csv",
                               help="csv file containing '['chr', 'start', 'end', 'covcorr', 'covraw'] data")
    coverage_mcmc.add_argument("--allelic_clusters_object",
                               help="npy file containing allelic dp segs-to-clusters results")
    coverage_mcmc.add_argument("--SNPs_pickle", help="pickled dataframe containing SNPs")
    coverage_mcmc.add_argument("--segmentations_pickle", help="pickled sorteddict containing allelic imbalance segment boundaries", required=True)
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
    preprocess_coverage_mcmc.add_argument("--segmentations_pickle", help="pickled sorteddict containing allelic imbalance segment boundaries", required=True)
    preprocess_coverage_mcmc.add_argument("--repl_pickle", help="pickled dataframe containing replication timing data", required=True)
    preprocess_coverage_mcmc.add_argument("--faire_pickle", help="pickled dataframe containing FAIRE data", required=False)
    preprocess_coverage_mcmc.add_argument("--normal_coverage_csv", help="csv file in the format of the tumor coverage file, but for the normal", required=False)
    preprocess_coverage_mcmc.add_argument("--panel_of_normals", help="path to newline delimited file listing multiple normal coverage CSVs", required=False)
    preprocess_coverage_mcmc.add_argument("--gc_pickle", help="pickled dataframe containing precomputed gc content. This is not required but will speed up runtime if passed", default=None)
    preprocess_coverage_mcmc.add_argument("--allelic_sample", type=int,
                                          help="index of sample clustering from allelic DP to use as seed for segmentation. Will use most likely clustering by default",
                                          default=None)
    preprocess_coverage_mcmc.add_argument("--ref_fasta", help="reference fasta file", required=True)
    preprocess_coverage_mcmc.add_argument("--bin_width", help = "Coverage bin width (for WGS only)", default = 1, type = int)
    preprocess_coverage_mcmc.add_argument("--wgs", help = "If not WGS, expand targets by +-150b to capture more SNPs", action = "store_true")

    ## running coverage mcmc on single cluster for scatter task
    coverage_mcmc_shard = subparsers.add_parser("coverage_mcmc_shard",
                                                help="run coverage mcmc on single ADP cluster")
    coverage_mcmc_shard.add_argument("--preprocess_data", help='path to numpy object containing preprocessed data: covariate matrix (C), global beta, ADP cluster mu\'s, covbin ADP cluster assignments (all_mu), covbin raw coverage values (r)',
                                     required=True)
    coverage_mcmc_shard.add_argument("--allelic_seg_indices", help='path to pickled pandas dataframe containing coverage bin indices for each alleic segment',
                                     required=True)
    coverage_mcmc_shard.add_argument("--allelic_seg_idx", help='which allelic segment to perform coverage segmentation on.',
                                     required=True, type=int)
    coverage_mcmc_shard.add_argument("--num_draws", type=int,
                               help="number of draws to take from coverage segmentation MCMC", default=5)
    coverage_mcmc_shard.add_argument("--bin_width", type=int, default=1, help="size of uniform bins if using. Otherwise 1.")
    coverage_mcmc_shard.add_argument("--burnin_files", type=str, help="txt file containing burnt in segment assignments")

    ## collect coverage MCMC shards
    collect_cov_mcmc = subparsers.add_parser("collect_cov_mcmc", help="collect sharded cov mcmc results")
    collect_cov_mcmc.add_argument("--coverage_dir", help="path to the directory containing the coverage mcmc results")
    collect_cov_mcmc.add_argument("--cov_mcmc_files",
                                  help="path to txt file with each line containing a path to a cov mcmc shard result")
    collect_cov_mcmc.add_argument("--cov_df_pickle",
                                  help="path to cov_df pickle file. Required for using --cov_mcmc_files option")
    collect_cov_mcmc.add_argument("--seg_indices_pickle", help='path to segment indices dataframe pickle', required=True)
    collect_cov_mcmc.add_argument("--bin_width", type=int, help="size of uniform bins if using. otherwise 1")
    collect_cov_mcmc.add_argument("--cytoband_file", required = True)

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
    gen_acdp_df.add_argument("--cov_df_pickle", help="path to cov_df pickle. For running on cov segmentation results directly")
    gen_acdp_df.add_argument("--cov_seg_data",  help="path to collected cov mcmc segmentation data. For running on cov segmentation results directly")
    gen_acdp_df.add_argument("--allelic_clusters_object",
                             help="npy file containing allelic dp segs-to-clusters results")
    gen_acdp_df.add_argument("--allelic_draw_index", help="index of ADP draw used for coverage MCMC", type=int, default=-1)
    gen_acdp_df.add_argument("--bin_width", help="size of uniform bins if using. Otherwise 1", default=1, type=int)
    gen_acdp_df.add_argument("--wgs", action='store_true', default=False, help="wgs mode creates acdp df for only the best draw")

    # run acdp clustering 
    ac_dp = subparsers.add_parser("allelic_coverage_dp", help="Run DP clustering on allelic coverage tuples")
    ac_dp.add_argument("--cov_seg_data", help="path to savez file containing global beta")
    ac_dp.add_argument("--acdp_df_path", help="path to acdp dataframe")
    ac_dp.add_argument("--num_samples", type=int, help="number of samples to take")
    ac_dp.add_argument("--cytoband_file", help="path to cytoband txt file")
    ac_dp.add_argument("--opt_cdp_idx", type=int, help="index of best cdp run")
    ac_dp.add_argument("--wgs", help="flag to determine if sample is whole genome", default=False, action='store_true')
    ac_dp.add_argument("--lnp_data_pickle", help="path to lnp data dictionary", required=True)
    ac_dp.add_argument("--use_single_draw", help="flag to force acdp to only use best draw", default=False, action='store_true')
    ac_dp.add_argument("--warmstart", type=bool, default=True, help="run clustering with warmstart")

    # final summary plot
    fp = subparsers.add_parser("summary_plot", help = "summary plot of HapASeg results")
    fp.add_argument("--snps_pickle", required=True, help="path to all SNPs pickle file")
    fp.add_argument("--adp_results", required=True, help="path to the allelic DP results file")
    fp.add_argument("--segmentations_pickle", required=True, help="path to final ADP segmentations")
    fp.add_argument("--acdp_model", required=True, help="path to acdp model pickle")
    fp.add_argument("--ref_fasta", required=True, help="path to matching reference fasta file")
    fp.add_argument("--cytoband_file", required=True, help="path to cytoband file")
    fp.add_argument("--hapaseg_segfile", required=True, help="path to hapaseg format segfile (can be clustered or unclustered)")
    fp.add_argument("--outdir", required=False, default='./', help="path to save figure")
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

    if args.command == "load_snps":
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
                n_iter=int(args.n_iter),
                betahyp=args.betahyp
            )

        # loading from allelic MCMC results object produced by `hapaseg amcmc`
        else:
            with open(args.amcmc_object, "rb") as f:
                H = pickle.load(f)

            # update some class properties set in the constructor
            H._set_ref_bias(float(args.ref_bias))
            H.n_iter = int(args.n_iter)
            H.quit_after_burnin = args.stop_after_burnin
            H._set_betahyp(args.betahyp)
            if len(H.marg_lik) > H.n_iter:
                H.marg_lik = H.marg_lik[:H.n_iter]

        with open(output_dir + "/amcmc_results.pickle", "wb") as f:
            pickle.dump(H.run(), f)

        try:
            H.visualize(show_CIs = args.betahyp != -1) # only plot SNP CI's if there are few SNPs (not WGS)
            plt.savefig(output_dir + "/figures/MLE_segmentation.png", dpi = 300)
        except Exception:
            print("Error plotting segments; see stack trace for details:")
            print(traceback.format_exc())

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
                g = chunk.P.iloc[st:en].groupby("allele_A")
                x = g[["REF_COUNT", "ALT_COUNT"]].sum()
                x["idx"] = i + j
                x["n_SNP"] = g.size()
                X.append(x)

            j += i + 1

        X = pd.concat(X)

        ## filter segments used to compute reference bias

        # 1. restrict to just segments with both alleles represented;
        # segments with only one allele are not informative when computing ref bias
        g = X.groupby("idx").size() == 2
        X = X.loc[X["idx"].isin(g[g].index)]

        # index by segment and allele
        X = X.set_index([X["idx"], X.index]).drop(columns = "idx")

        # 2. we don't want to use segments with too few supporting SNPs (20)
        tot_SNPs = X.groupby(level = 0)["n_SNP"].sum()
        count_idx = X.groupby(level = 0)["n_SNP"].apply(lambda x : (x > 20).all())

        # 3. don't use LoH segments at ~100% purity in reference bias calculations, since
        # these consistently have f_alt ~ 1, f_ref ~ 0, yielding optimal reference bias of 1
        # segments with very little allelic imbalance density between 0.1 and 0.9 are considered LoH
        a = X.loc[(slice(None), 0), "ALT_COUNT"].droplevel(1) + X.loc[(slice(None), 1), "REF_COUNT"].droplevel(1) + 1
        b = X.loc[(slice(None), 0), "REF_COUNT"].droplevel(1) + X.loc[(slice(None), 1), "ALT_COUNT"].droplevel(1) + 1
        rbdens = s.beta.cdf(0.9, a, b) - s.beta.cdf(0.1, a, b)

        # final index of segments to use
        use_idx = count_idx & (rbdens > 0.01)

        ## perform iterative grid search over range of reference bias values
        refbias_dom = np.linspace(0.8, 1, 10)
        print("Computing reference bias ...", file = sys.stderr)
        for opt_iter in range(3):
            # number of Monte Carlo samples to draw from beta distribution
            # we need fewer samples for more total SNPs
            n_beta_samp = np.r_[10, 100, np.maximum(100, int(1e8/tot_SNPs.sum()))][opt_iter]
            # perform increasingly fine grid searches around neighborhood of previous optimum
            if opt_iter > 0:
                refbias_dom = np.linspace(
                  *refbias_dom[np.minimum(np.argmin(refbias_dif) + np.r_[-2, 2], len(refbias_dom) - 1)], # search +- 2 grid points of previous optimum; clip to rb = 1
                  np.r_[10, 20, 30][opt_iter] # fineness of grid search
                )
            refbias_dif = np.full(len(refbias_dom), np.inf)
            pbar = tqdm.tqdm(enumerate(refbias_dom), total = len(refbias_dom))
            for j, rb in pbar:
                pbar.set_description(f"[{refbias_dom.min():0.2f}:{refbias_dom.max():0.2f}:{len(refbias_dom)}] {refbias_dom[refbias_dif.argmin()]:0.4f} ({n_beta_samp} MC samples)")
                absdif = np.full(use_idx.sum(), np.nan)
                for i, seg in enumerate(tot_SNPs.index[use_idx]):
                    f_A = s.beta.rvs(
                      X.loc[(seg, 0), "ALT_COUNT"] + 1,
                      X.loc[(seg, 0), "REF_COUNT"]*rb + 1,
                      size = n_beta_samp
                    )
                    f_B = s.beta.rvs(
                      X.loc[(seg, 1), "REF_COUNT"]*rb + 1,
                      X.loc[(seg, 1), "ALT_COUNT"] + 1,
                      size = n_beta_samp
                    )

                    absdif[i] = np.abs(f_A - f_B).mean()

                refbias_dif[j] = absdif@tot_SNPs[use_idx]/tot_SNPs[use_idx].sum()

            ref_bias = refbias_dom[np.argmin(refbias_dif)]

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

            A.maj_arr = A.P.iloc[:, A.maj_idx].astype(int).values
            A.min_arr = A.P.iloc[:, A.min_idx].astype(int).values

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
        A = A_DP(args.seg_dataframe, ref_fasta=args.ref_fasta, wgs = args.wgs)

        # run DP
        snps_to_clusters, snps_to_phases, likelihoods = A.run()

        # save DP results
        # SNP assignment/phasing samples, likelihoods of each sample
        np.savez(output_dir + "/allelic_DP_SNP_clusts_and_phase_assignments.npz",
                 snps_to_clusters=snps_to_clusters,
                 snps_to_phases=snps_to_phases,
                 likelihoods=likelihoods
                 )

        # segmentation breakpoints for each sample
        with open(output_dir + "/segmentations.pickle", "wb") as f:
            pickle.dump(A.DP_run.segment_trace, f)

        # full SNP dataframe
        A.SNPs.to_pickle(output_dir + "/all_SNPs.pickle")

        #
        # plot DP results

        # 0. likelihood trace
        A.DP_run.plot_likelihood_trace()
        plt.savefig(output_dir + "/figures/likelihood_trace.png", dpi = 300)

        # 1. SNPs + segments
        f = plt.figure(figsize = [17.56, 5.67])
        hs_utils.plot_chrbdy(args.cytoband_file)
        A.DP_run.visualize_segs(f = f, show_snps = True)
        plt.ylabel("Haplotypic imbalance")
        plt.title("SNPs + allelic segmentation (MAP)")
        plt.savefig(output_dir + "/figures/SNPs.png", dpi = 300)
        plt.close()

        # 2. segments alone
        f = plt.figure(figsize = [17.56, 5.67])
        hs_utils.plot_chrbdy(args.cytoband_file)
        A.DP_run.visualize_segs(f = f, show_snps = False)
        plt.ylabel("Haplotypic imbalance")
        plt.title("Allelic segmentation (posterior)")
        plt.savefig(output_dir + "/figures/segs_only.png", dpi = 300)
        plt.close()

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
        ## perform initial Poisson regression
        cov_mcmc_runner = CoverageMCMCRunner(args.coverage_csv,
                                             args.allelic_clusters_object,
                                             args.SNPs_pickle,
                                             args.segmentations_pickle,
                                             args.ref_fasta,
                                             args.repl_pickle,
                                             args.faire_pickle,
                                             f_GC=args.gc_pickle,
                                             f_Ncov=args.normal_coverage_csv,
                                             f_PoN=args.panel_of_normals,
                                             allelic_sample=args.allelic_sample,
                                             bin_width=args.bin_width,
                                             wgs=args.wgs)
        Pi, r, C, all_mu, global_beta, cov_df, adp_cluster = cov_mcmc_runner.prepare_single_cluster()

        # indices of coverage bins 
        seg_g = cov_df.groupby("seg_idx") # NOTE: seg_idx may not be contiguous if any allelic segments were dropped
        seg_g_idx = pd.Series(seg_g.indices).to_frame(name = "indices")
        seg_g_idx["allelic_cluster"] = seg_g["allelic_cluster"].first()
        seg_g_idx["n_cov_bins"] = seg_g.size()
        
        ## save
        # regression matrices
        np.savez(os.path.join(output_dir, 'preprocess_data'), Pi=Pi, r=r, C=C, all_mu=all_mu,
                 global_beta=global_beta, adp_cluster=adp_cluster)
        # coverage dataframe mapped 
        cov_df.to_pickle(os.path.join(output_dir, 'cov_df.pickle'))
        # allelic segment indices into coverage dataframe
        seg_g_idx.to_pickle(os.path.join(output_dir, 'allelic_seg_groups.pickle'))
        
        with open(os.path.join(output_dir, 'allelic_seg_idxs.txt'), 'w') as f:
            for i in seg_g_idx.index:
                f.write("{}\n".format(i))

    ## run scattered coverage mcmc job using preprocessed data
    elif args.command == "coverage_mcmc_shard":
        # load preprocessed data
        preprocess_data = np.load(args.preprocess_data)

        # extract preprocessed data from this cluster
        Pi = preprocess_data['Pi']
        mu = preprocess_data["all_mu"]#[args.cluster_num]
        beta = preprocess_data["global_beta"]
        c_assignments = np.argmax(Pi, axis=1)
        #cluster_mask = (c_assignments == args.cluster_num)
        r = preprocess_data['r']#[cluster_mask]
        C = preprocess_data['C']#[cluster_mask]

        # load and (weakly) verify allelic segment indices
        seg_g_idx = pd.read_pickle(args.allelic_seg_indices)
        #if len(np.hstack(seg_g_idx["indices"])) != C.shape[0]:
        #    raise ValueError("Size mismatch between allelic segment assignments and coverage bin data!")

        # subset to a single allelic segment
        if args.allelic_seg_idx not in seg_g_idx.index:
            raise ValueError("Allelic segment index out of bounds!")

        seg_indices = seg_g_idx.loc[args.allelic_seg_idx]
        
        mu = mu[seg_g_idx.index.get_loc(args.allelic_seg_idx)]
        C = C[seg_indices["indices"], :]
        r = r[seg_indices["indices"], :]
        
        # run cov MCMC
        cov_mcmc = Coverage_MCMC_SingleCluster(args.num_draws, r, C, mu, beta, args.bin_width)

#        # if we get a range argument well be doing burnin on a subset of the coverage bins
#        if args.range is not None:
#            #parse range from string
#            range_lst = args.range.split('-')
#            st,en = int(range_lst[0]), int(range_lst[1]) 
#            if st > en or st < 0 or en > len(r):
#                raise ValueError("invalid range! got range {} for cluster {} with size {}".format(args.range, args.cluster_num, len(r)))
#            
#            #trim data to our desired range
#            r = r[st:en]
#            C = C[st:en]
#            num_draws = 1
#            
#            # if we're just burning in a subset use different save strings
#            model_save_str = 'cov_mcmc_model_cluster_{}_{}.pickle'.format(args.cluster_num, args.range)
#            data_save_str = 'cov_mcmc_data_cluster_{}_{}'.format(args.cluster_num, args.range)
#            figure_save_str = 'cov_mcmc_cluster_{}_{}_visual'.format(args.cluster_num, args.range)
#            
#        else:
#            #if not in burnin use the specified number of draws
#            num_draws = args.num_draws
#            
#            
#            model_save_str = 'cov_mcmc_model_cluster_{}.pickle'.format(args.cluster_num)
#            data_save_str = 'cov_mcmc_data_cluster_{}'.format(args.cluster_num)
#            figure_save_str = 'cov_mcmc_cluster_{}_visual'.format(args.cluster_num)
#        
#        # run on the specified cluster
#        cov_mcmc = NB_MCMC_SingleCluster(num_draws, r, C, mu, beta, args.cluster_num, args.bin_width)
#        
#        # if we're using burnin results load them now
#        if args.burnin_files is not None:
#            with open(args.burnin_files, 'r') as f:
#                file_list = f.read().splitlines()
#            assignments_arr = aggregate_burnin_files(file_list, args.cluster_num)
#            cov_mcmc.init_burnin(assignments_arr)

        cov_mcmc.run()

        # collect the results
        segment_samples, global_beta, mu_i_samples, ll_samples = cov_mcmc.prepare_results()
        
        model_save_str = 'cov_mcmc_model_allelic_seg_{}.pickle'.format(args.allelic_seg_idx)
        data_save_str = 'cov_mcmc_data_allelic_seg_{}'.format(args.allelic_seg_idx)
        figure_save_str = 'cov_mcmc_allelic_seg_{}_visual'.format(args.allelic_seg_idx)
        
        # save samples
        with open(os.path.join(output_dir, model_save_str), 'wb') as f:
            pickle.dump(cov_mcmc, f)

        np.savez(os.path.join(output_dir, data_save_str),
                 seg_samples=segment_samples, beta=global_beta, mu_i_samples=mu_i_samples, ll_samples = ll_samples)

        # save visualization
        cov_mcmc.visualize_cluster_samples(
            os.path.join(output_dir, figure_save_str))

    elif args.command == "collect_cov_mcmc":
        if args.coverage_dir:
            full_segmentation, mu, beta, ll_samples = aggregate_clusters(coverage_dir=args.coverage_dir, cov_df_pickle=args.cov_df_pickle)

        elif args.cov_mcmc_files:
            if args.cov_df_pickle is None:
                raise ValueError("cov_df_pickle argument required for passing shard file")
            full_segmentation, mu, beta, ll_samples= aggregate_clusters(seg_indices_pickle= args.seg_indices_pickle, f_file_list=args.cov_mcmc_files, cov_df_pickle=args.cov_df_pickle, bin_width=args.bin_width)
        else:
            # need to pass in one or the other
            raise ValueError("must pass in either a directory or a txt file listing mcmc results")

        ## save these results to new aggregated file
        if args.coverage_dir:
            np.savez(os.path.join(args.coverage_dir, 'cov_mcmc_collected_data'), seg_samples=full_segmentation,
                     beta=beta, ll_samples = ll_samples)
        else:
            np.savez(os.path.join(output_dir, 'cov_mcmc_collected_data'), seg_samples=full_segmentation, beta=beta, ll_samples = ll_samples)

        ## visualize
        cov_df = pd.read_pickle(args.cov_df_pickle)
        covar_columns = sorted(cov_df.columns[cov_df.columns.str.contains("(?:^C_.*_l?z$|C_log_len)")])
        emu = np.exp(mu)
        C = np.c_[cov_df[covar_columns]]
        seg_idx = full_segmentation[:, ll_samples.argmax()].astype(int)

        f = plt.figure(figsize = [17.56, 5.67])
        residuals = cov_df["fragcorr"]/np.exp(C@beta).ravel()
        plt.scatter(cov_df["start_g"], residuals, marker = ".", s = 1, c = np.array(["dodgerblue", "orangered"])[seg_idx % 2], alpha = 0.5)
        plt.scatter(cov_df["start_g"], emu[seg_idx].ravel(), marker = ".", s = 1, c = 'k')
        
        plt.xlim([0, cov_df["end_g"].max()])
        plt.ylim([residuals.min(), residuals.max()])
        hs_utils.plot_chrbdy(args.cytoband_file)

        plt.ylabel("Corrected fragment coverage")
        plt.title("Total copy segmentation")
        plt.savefig(output_dir + "/figures/segs.png", dpi = 300)

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
            acdp_df, lnp_data, opt_cdp_idx = generate_acdp_df(args.snp_dataframe,
                                             args.allelic_clusters_object,
                                             cdp_object_path=args.cdp_object,
                                             bin_width=args.bin_width,
                                             ADP_draw_index=args.allelic_draw_index,
                                             wgs=args.wgs)
        
        elif args.cdp_filepaths is not None:
            #all of our dp runs are in one object
            acdp_df, lnp_data, opt_cdp_idx = generate_acdp_df(args.snp_dataframe,
                                             args.allelic_clusters_object,
                                             cdp_scatter_files=args.cdp_filepaths,
                                             bin_width=args.bin_width,
                                             ADP_draw_index=args.allelic_draw_index,
                                             wgs=args.wgs)
        
        # for running directly from cov_mcmc segments
        elif args.cov_df_pickle is not None and args.cov_seg_data is not None:
            acdp_df, lnp_data, opt_cdp_idx = generate_acdp_df(args.snp_dataframe,
                                             args.allelic_clusters_object,
                                             cov_df_path=args.cov_df_pickle,
                                             cov_mcmc_data_path =args.cov_seg_data,
                                             bin_width=args.bin_width,
                                             ADP_draw_index=args.allelic_draw_index,
                                             wgs=args.wgs)
                        
        else:
            raise ValueError("must pass a cdp filepath, list of cdp filepaths or cov_df pickle and mcmc seg file")
           
        acdp_df.to_pickle(os.path.join(output_dir, "acdp_df.pickle"))
        with open('./lnp_data.pickle', 'wb') as f:
            pickle.dump(lnp_data, f)

        with open('./opt_cdp_draw.txt', 'w') as f:
            f.write(str(opt_cdp_idx))
       
    elif args.command == "allelic_coverage_dp":
        acdp_df = pd.read_pickle(args.acdp_df_path)
        mcmc_data = np.load(args.cov_seg_data)
        beta = mcmc_data['beta']
       
        with open(args.lnp_data_pickle, 'rb') as f:
            lnp_data = pickle.load(f) 
        draw_idx = args.opt_cdp_idx if args.use_single_draw else None
        acdp = AllelicCoverage_DP_runner(acdp_df, 
                                  beta, 
                                  args.cytoband_file,
                                  lnp_data,
                                  wgs=args.wgs,
                                  draw_idx=draw_idx,
                                  seed_all_clusters=args.warmstart)
        acdp_combined, opt_purity, opt_k = acdp.run_seperated(args.num_samples)
        print("assigning flagged segments...", flush=True)
        acdp_combined.assign_greylist()

        print("visualizing run")
        
        # save segmentation df
        seg_df = acdp_combined.create_allelic_segs_df()
        seg_df.to_csv('./hapaseg_segfile.txt', sep = '\t', index = False)
    
        absolute_df = acdp_combined.create_allelic_segs_df(absolute_format=True)
        absolute_df.to_csv('./absolute_segfile.txt', sep='\t', index=False)
    
        # save the unclustered segs
        acdp.unclustered_seg_df.to_csv('./hapaseg_skip_acdp_segfile.txt', sep = "\t", index = False)

        # make visualizations
        acdp_combined.visualize_ACDP_clusters(output_dir)

        if args.wgs:
            acdp_combined.visualize_ACDP('./acdp_agg_draws.png', use_cluster_stats=True)
        else:
            acdp_combined.visualize_ACDP('./acdp_agg_draws.png', plot_real_cov=True, use_cluster_stats=True)
        
        if not args.use_single_draw:
            acdp_combined.visualize_ACDP('./acdp_best_cdp_draw.png', use_cluster_stats=True, cdp_draw=int(args.opt_cdp_idx))
            acdp_combined.visualize_ACDP('./acdp_all_draws.png')
        
        # print opt purity and opt k
        with open('./acdp_optimal_fit_params.txt', 'w') as f:
            f.write('purity\tk\n')
            f.write(f'{opt_purity}\t{opt_k}\n')
        
        with open(os.path.join(output_dir, "acdp_model.pickle"), "wb") as f:
            pickle.dump(acdp_combined, f)

    elif args.command == "summary_plot":
        adp_obj = load_DP_object_from_outputs(args.snps_pickle, args.adp_results, args.segmentations_pickle)
        colors = mpl.cm.get_cmap("tab10").colors 
        # adp results window
        f, axs = plt.subplots(3,1, sharex=True)
        adp_obj.visualize_segs(ax=axs[0], show_snps=True)
        plt.sca(axs[0])
        plt.yticks(fontsize=6)
        hs_utils.plot_chrbdy(args.cytoband_file)

        # coverage segmentation results window
        acdp = pd.read_pickle(args.acdp_model)
        cov_df = acdp.cov_df.loc[acdp.cov_df.allele == -1]
        covar_cols = sorted(cov_df.columns[cov_df.columns.str.contains("^C_.*z$|^C_log_len$")])
        C = cov_df[covar_cols].values
        residuals = np.exp(np.log(cov_df.fragcorr) - (C@acdp.beta).flatten())
        
        for i in cov_df.segment_ID.unique():
            sub = cov_df.loc[cov_df.segment_ID == i]
            axs[1].scatter(sub.start_g, residuals[sub.index], color = colors[i % 3], s= 1, alpha=0.2)
            # plot seg mu
            start = sub.iloc[0].start_g
            end = sub.iloc[-1].start_g
            mu_exp = np.exp(sub.iloc[0].cov_DP_mu)
            axs[1].plot((start, end), (mu_exp, mu_exp), linewidth=0.5, color='k')
            axs[1].scatter((start + end)/2, mu_exp, marker='.',  s=1, color='k')
        plt.sca(axs[1])
        plt.yticks(fontsize=6)
        hs_utils.plot_chrbdy(args.cytoband_file)

        # plot acdp segment results window
        seg_df = pd.read_csv(args.hapaseg_segfile, sep = '\t')
        seg_df['start_g'] = seq.chrpos2gpos(seg_df['Chromosome'], seg_df['Start.bp'], ref=args.ref_fasta)
        seg_df['end_g'] = seq.chrpos2gpos(seg_df['Chromosome'], seg_df['End.bp'], ref=args.ref_fasta)
        for i, row in seg_df.iterrows():
            seg_len = row['End.bp'] - row['Start.bp']
            mu_major = mpl.patches.Rectangle((row['start_g'], row['mu.major'] - 2), seg_len, 4,  alpha=0.5, color='r')
            axs[2].add_patch(mu_major)
            ci_major = mpl.patches.Rectangle((row['start_g'], row['mu.major'] - 1.95 * row['sigma.major']), seg_len, 2 * 1.95 * row['sigma.major'], alpha=0.1, color='r')
            axs[2].add_patch(ci_major)

            mu_minor =  mpl.patches.Rectangle((row['start_g'], row['mu.minor'] - 2), seg_len, 4, alpha = 0.5, color='b')
            axs[2].add_patch(mu_minor)
            ci_minor = mpl.patches.Rectangle((row['start_g'], row['mu.minor'] - 1.95 * row['sigma.minor']), seg_len, 2 * 1.95 * row['sigma.minor'], alpha=0.1, color='b')
            axs[2].add_patch(ci_minor)
        
        # for some reason patches won't plot unless we draw to the axis again
        axs[2].scatter([0],[0], s=1, alpha=0) 
        plt.sca(axs[2])
        plt.yticks(fontsize=6)
        hs_utils.plot_chrbdy(args.cytoband_file)
        axs[0].set_title('Allelic DP Clusters', size = 8)
        axs[0].set_ylabel('allelic imbalance', size = 8)
        axs[1].set_title('Coverage Segmentation', size=8)
        axs[1].set_ylabel('corrected coverage', size = 8)
        axs[2].set_title('Allelic Coverage Segments', size=8)
        axs[2].set_ylabel('corrected coverage', size = 8)
        axs[2].set_xlabel('Chromosomes', size=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'hapaseg_summary_plot.png'), dpi=200)

if __name__ == "__main__":
    main()
