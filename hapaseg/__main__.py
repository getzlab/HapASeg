import argparse
import dask.distributed as dd
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats as s
import scipy.special as ss

from capy import mut

from .load import HapasegReference
from .run_allelic_MCMC import AllelicMCMCRunner
from .allelic_MCMC import A_MCMC

def parse_args():
    parser = argparse.ArgumentParser(description = "Call somatic copynumber alterations taking advantage of SNP phasing")

    parser.add_argument("--output_dir", default = ".")

    subparsers = parser.add_subparsers(dest = "command")

    ## run
    standalone = subparsers.add_parser("run", help = "Run HapASeg as a standalone module")
    standalone.add_argument("--n_workers", default = multiprocessing.cpu_count() - 1)
    standalone.add_argument("--n_iter", default = 20000)

    input_group = standalone.add_mutually_exclusive_group(
      required = True
      #"Mutually exclusive inputs",
      #"Hapaseg can either take an already phased VCF annotated with alt/refcounts, or a MuTect callstats file. In the case of the latter, Hapaseg will perform phasing.",
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
    phasing_ref_group.add_argument("--bam", help = "BAM to use for read-backed phasing")
    phasing_ref_group.add_argument("--bai", help = "BAI to use for read-backed phasing")

    # if we are taking a pre-phased VCF, we also may want a read-backed phased VCF
    prephased_group = standalone.add_argument_group( 
      "Inputs if phasing has already been imputed",
    )
    prephased_group.add_argument("--read_backed_phased_VCF", help = "Optional.")
    prephased_group.add_argument("--allele_counts_T", help = "Required.")
    prephased_group.add_argument("--allele_counts_N", help = "Required.")

    ai_seg_params = standalone.add_argument_group(
      "Parameters for allelic imbalance segmentation",
    )
    ai_seg_params.add_argument("--phase_correct", action = "store_true")
    ai_seg_params.add_argument("--misphase_prior", default = "0.001")

    ref_group = standalone.add_argument_group(
      "Required reference files",
    )
    ref_group.add_argument("--cytoband_file", required = True)

    ## load
    scatter = subparsers.add_parser("load", help = "Load in phased VCF")
    scatter.add_argument("--chunk_size", default = 5000) 
    scatter.add_argument("--phased_VCF", required = True)
    scatter.add_argument("--read_backed_phased_VCF")
    scatter.add_argument("--allele_counts_T", required = True)
    scatter.add_argument("--allele_counts_N", required = True)
    scatter.add_argument("--cytoband_file", required = True)

    ## amcmc
    amcmc = subparsers.add_parser("amcmc", help = "Run allelic MCMC on a range of SNPs")
    amcmc.add_argument("--snp_dataframe", required = True)
    amcmc.add_argument("--start", required = True)
    amcmc.add_argument("--end", required = True)
    amcmc.add_argument("--stop_after_burnin", action = "store_true")
    amcmc.add_argument("--ref_bias", default = 1.0)
    amcmc.add_argument("--n_iter", default = 20000)

    ## concat
    concat = subparsers.add_parser("concat", help = "Concatenate burned-in chunks")
    concat.add_argument("--chunks", required = True, nargs = "+")
    concat.add_argument("--scatter_intervals", required = True)

    ## DP (TODO: will include gather step)
    dp = subparsers.add_parser("dp", help = "Run DP clustering on allelic imbalance segments")

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
        dask_client = dd.Client(n_workers = int(args.n_workers))

        refs = HapasegReference(
          phased_VCF = args.phased_VCF,
          readbacked_phased_VCF = args.read_backed_phased_VCF,
          allele_counts = args.allele_counts_T,
          allele_counts_N = args.allele_counts_N,
          cytoband_file = args.cytoband_file
        )

        runner = AllelicMCMCRunner(
          refs.allele_counts,
          refs.chromosome_intervals,
          dask_client,
          phase_correct = args.phase_correct
        )

        allelic_segs = runner.run_all()

        # TODO: checkpoint here
        allelic_segs.to_pickle(output_dir + "/allelic_imbalance_segments.pickle")

        # TODO: save per-chromosome plots of raw allelic segmentations

    elif args.command == "load":
        # load from VCF
        refs = HapasegReference(
          phased_VCF = args.phased_VCF,
          readbacked_phased_VCF = args.read_backed_phased_VCF,
          allele_counts = args.allele_counts_T,
          allele_counts_N = args.allele_counts_N,
          cytoband_file = args.cytoband_file
        )

        # create chunks
        t = mut.map_mutations_to_targets(refs.allele_counts, refs.chromosome_intervals, inplace = False)
        groups = t.groupby(t).apply(lambda x : [x.index.min(), x.index.max()]).to_frame(name = "bdy")
        groups["ranges"] = groups["bdy"].apply(lambda x : np.r_[x[0]:x[1]:args.chunk_size, x[1]])
        chunks = pd.DataFrame(
          np.vstack([
            np.hstack(np.broadcast_arrays(k, np.c_[y[0:-1], y[1:]]))
            for k, y in groups["ranges"].iteritems()
          ])[:, 1:],
          columns = ["arm", "start", "end"]
        )

        # save to disk
        refs.allele_counts.to_pickle(output_dir + "/allele_counts.pickle")
        refs.chromosome_intervals.to_pickle(output_dir + "/chrom_int.pickle")
        chunks.to_csv(output_dir + "/scatter_chunks.tsv", sep = "\t", index = False)

    elif args.command == "amcmc":
        P = pd.read_pickle(args.snp_dataframe)
        H = A_MCMC(
          P.iloc[int(args.start):int(args.end)],
          quit_after_burnin = args.stop_after_burnin,
          ref_bias = float(args.ref_bias),
          n_iter = int(args.n_iter)
        )

        with open(output_dir + "/amcmc_results.pickle", "wb") as f:
            pickle.dump(H.run(), f)

    elif args.command == "concat":
        #
        # load scatter intervals
        intervals = pd.read_csv(args.scatter_intervals, sep = "\t")
        intervals["results"] = None

        if len(intervals) != len(args.chunks):
            raise ValueError("Length mismatch in supplied chunks and interval file!")

        # load results
        R = []
        for chunk_path in args.chunks:
            with open(chunk_path, "rb") as f:
                chunk = pickle.load(f)
            R.append(chunk)
        R = pd.DataFrame({ "results" : R })

        # ensure results are in the correct order
        R["first"] = R["results"].apply(lambda x : x.P.loc[0, "index"])
        R = R.sort_values("first", ignore_index = True)

        # concat with intervals
        R = pd.concat([R, intervals], axis = 1).drop(columns = ["first"])

        #
        # compute reference bias
        X = []
        j = 0
        for chunk in R["results"]:
            bpl = np.array(chunk.breakpoints); bpl = np.c_[bpl[0:-1], bpl[1:]]

            for i, (st, en) in enumerate(bpl):
                x = chunk.P.iloc[st:en].groupby("allele_A")[["MIN_COUNT", "MAJ_COUNT"]].sum()
                x["idx"] = i + j
                X.append(x)

            j += len(chunk.P)

        X = pd.concat(X)
        g = X.groupby("idx").size() == 2
        Y = X.loc[X["idx"].isin(g[g].index)]

        f = np.zeros([len(Y)//2, 100])
        l = np.zeros([len(Y)//2, 2])
        for i, (_, g) in enumerate(Y.groupby("idx")):
            f[i, :] = s.beta.rvs(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1, size = 100)/s.beta.rvs(g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1, size = 100)
            l[i, :] = np.r_[
              ss.betaln(g.loc[0, "MIN_COUNT"] + 1, g.loc[0, "MAJ_COUNT"] + 1),
              ss.betaln(g.loc[1, "MIN_COUNT"] + 1, g.loc[1, "MAJ_COUNT"] + 1),
            ]

        # weight mean by negative log marginal likelihoods
        # take smaller of the two likelihoods to account for power imbalance
        w = np.min(-l, 1, keepdims = True)

        ref_bias = (f*w).sum()/(100*w.sum())

        #
        # concat burned in chunks for each arm
        #for arm, Ra in R.groupby("arm"):

if __name__ == "__main__":
    main()
