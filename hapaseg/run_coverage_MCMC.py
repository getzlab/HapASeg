import numpy as np
import pandas as pd

import os

from capy import mut, seq

from .NB_coverage_MCMC import NB_MCMC_AllClusters, NB_MCMC_SingleCluster


class CoverageMCMCRunner:
    def __init__(self,
                 coverage_csv,
                 f_allelic_clusters,
                 f_SNPs,
                 c,
                 covariate_dir,
                 num_draws=50,
                 cluster_num=None,
                 allelic_sample=None
                 ):
        # TODO: type check sample is int within range
        self.client = c
        self.covariate_dir = covariate_dir
        self.num_draws = num_draws
        self.cluster_num = cluster_num

        self.allelic_clusters = np.load(f_allelic_clusters)
        if allelic_sample:
            self.allelic_sample = allelic_sample
        else:
            # randomly chose sample
            num_samples = self.allelic_clusters["snps_to_clusters"].shape[0]
            self.allelic_sample = np.random.choice(num_samples)

        # for now coverage input is expected to be a csv file with columns: ["chr", "start", "end", "covcorr", "covraw"]
        self.full_cov_df = self.load_coverage(coverage_csv)
        self.load_covariates()
        self.SNPs = self.load_SNPs(f_SNPs)
        self.model = None

    def run(self):
        Pi, r, C, filtered_cov_df = self.assign_clusters()

        if self.cluster_num is None:
            # run coverage mcmc on all clusters
            # assign coverage bins to allelic clusters from the specified allelic sample (if specified; o.w. random choice)
            cov_mcmc = NB_MCMC_AllClusters(self.num_draws, r, C, Pi)
        else:
            # run mcmc on single cluster
            if self.cluster_num > self.Pi.shape[1]:
                # in this case our assigned cluster was trimmed for being garbage so we abort
                return None, None, None, None
            c_assignments = np.argmax(self.Pi, axis=1)
            cluster_mask = (self.c_assignments == self.cluster_num)
            cov_mcmc = NB_MCMC_SingleCluster(self.num_draws, r[cluster_mask], C[cluster_mask], self.cluster_num)

        # either way we run and save results
        cov_mcmc.run()
        self.model = cov_mcmc
        segment_samples, global_beta, mu_i_samples = cov_mcmc.prepare_results()
        return segment_samples, global_beta, mu_i_samples, filtered_cov_df

    @staticmethod
    def load_coverage(coverage_csv):
        Cov = pd.read_csv(coverage_csv, sep="\t", names=["chr", "start", "end", "covcorr", "covraw"],
                          low_memory=False)
        Cov["chr"] = mut.convert_chr(Cov["chr"])
        Cov = Cov.loc[Cov["chr"] != 0]
        Cov["start_g"] = seq.chrpos2gpos(Cov["chr"], Cov["start"])
        Cov["end_g"] = seq.chrpos2gpos(Cov["chr"], Cov["end"])

        return Cov

    def load_SNPs(self, f_snps):
        SNPs = pd.read_pickle(f_snps)
        SNPs["chr"], SNPs["pos"] = seq.gpos2chrpos(SNPs["gpos"])

        SNPs["tidx"] = mut.map_mutations_to_targets(SNPs, self.full_cov_df, inplace=False)
        return SNPs

    def load_covariates(self):
        # TODO make this more flexlable for arbitrary covariate inputs

        # f_covariate_lst = glob.glob(self.covariate_dir + '*.pickle')
        self.full_cov_df["C_len"] = self.full_cov_df["end"] - self.full_cov_df["start"] + 1

        # load repl timing
        F = pd.read_pickle(os.path.join(self.covariate_dir, "GSE137764_H1.hg19_liftover.pickle"))

        # map targets to RT intervals
        tidx = mut.map_mutations_to_targets(self.full_cov_df.rename(columns={"start": "pos"}), F, inplace=False)
        self.full_cov_df.loc[tidx.index, "C_RT"] = F.iloc[tidx, 3:].mean(1).values

        # z-transform
        self.full_cov_df["C_RT_z"] = (lambda x: (x - np.nanmean(x)) / np.nanstd(x))(
            np.log(self.full_cov_df["C_RT"] + 1e-20))

        # load GC content
        B = pd.read_pickle(os.path.join(self.covariate_dir, "GC.pickle"))
        self.full_cov_df = self.full_cov_df.merge(B.rename(columns={"gc": "C_GC"}), left_on=["chr", "start", "end"],
                                                  right_on=["chr", "start", "end"], how="left")
        self.full_cov_df["C_GC_z"] = (lambda x: (x - np.nanmean(x)) / np.nanstd(x))(
            np.log(self.full_cov_df["C_GC"] + 1e-20))

    # use SNP cluster assignments from the given draw assign coverage bins to clusters
    # clusters with snps from different clusters are probabliztically assigned
    # method returns coverage df with only bins that overlap snps
    def assign_clusters(self):
        # generate unique clust assignments
        clust_choice = self.allelic_clusters["snps_to_clusters"][self.allelic_sample]
        clust_u, clust_uj = np.unique(clust_choice, return_inverse=True)
        clust_uj = clust_uj.reshape(clust_choice.shape)

        # assign coverage intervals to clusters
        Cov_clust_probs = np.zeros([len(self.full_cov_df), clust_u.max()])

        for targ, snp_idx in self.SNPs.groupby("tidx").indices.items():
            targ_clust_hist = np.bincount(clust_uj[snp_idx].ravel(), minlength=clust_u.max())

            Cov_clust_probs[int(targ), :] = targ_clust_hist / targ_clust_hist.sum()

        # subset intervals containing SNPs
        overlap_idx = Cov_clust_probs.sum(1) > 0
        Cov_clust_probs_overlap = Cov_clust_probs[overlap_idx, :]

        # prune improbable assignments
        Cov_clust_probs_overlap[Cov_clust_probs_overlap < 0.05] = 0
        Cov_clust_probs_overlap /= Cov_clust_probs_overlap.sum(1)[:, None]
        prune_idx = Cov_clust_probs_overlap.sum(0) > 0
        Cov_clust_probs_overlap = Cov_clust_probs_overlap[:, prune_idx]
        num_pruned_clusters = Cov_clust_probs_overlap.shape[1]
        # subsetting to only targets that overlap SNPs
        Cov_overlap = self.full_cov_df.loc[overlap_idx, :]

        # probabilistically assign each ambiguous coverage bin to a cluster
        amb_mask = np.max(Cov_clust_probs_overlap, 1) != 1
        amb_assgn_probs = Cov_clust_probs_overlap[amb_mask, :]
        new_assgn = np.array([np.random.choice(np.r_[:num_pruned_clusters],
                                               p=amb_assgn_probs[i]) for i in range(len(amb_assgn_probs))])
        new_onehot = np.zeros((new_assgn.size, num_pruned_clusters))
        new_onehot[np.arange(new_assgn.size), new_assgn] = 1

        Pi = Cov_clust_probs_overlap.copy()
        Pi[amb_mask, :] = new_onehot

        r = np.c_[Cov_overlap["covcorr"]]

        # making covariate matrix
        C = np.c_[np.log(Cov_overlap["C_len"]), Cov_overlap["C_RT_z"], Cov_overlap["C_GC_z"]]

        # dropping Nans
        naidx = np.isnan(C[:, 1])
        r = r[~naidx]
        C = C[~naidx]
        Pi = Pi[~naidx]

        Cov_overlap = Cov_overlap.iloc[~naidx]
        Cov_overlap['cluster_assgn'] = np.argmax(Pi, axis=1)

        return Pi, r, C, Cov_overlap
