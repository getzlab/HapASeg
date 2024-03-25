import numpy as np
import pandas as pd
import pickle
import glob
import re
import os
import scipy.special as ss
import sys
import tqdm
from capy import mut, seq
import scipy.stats as stats
from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB

from .coverage_MCMC import Coverage_MCMC_AllClusters, Coverage_MCMC_SingleCluster
from .model_optimizers import PoissonRegression, CovLNP_NR_prior

zt = lambda x : (x - np.nanmean(x))/np.nanstd(x)

class CoverageMCMCRunner:
    def __init__(self,
                 coverage_csv,

                 # allelic segment files
                 f_allelic_clusters,
                 f_SNPs,
                 f_segs,
                 ref_fasta,

                 # covariates
                 f_repl,
                 f_faire,
                 f_GC=None,
                 f_Ncov=None,
                 f_extracov_bed_list=None,

                 num_draws=50,
                 cluster_num=None,
                 allelic_sample=None,
                 bin_width=1,
                 wgs=True,
                 SNP_expansion_radius = None,
                 region_blacklist_bed = None
                 ):

        self.num_draws = num_draws
        self.cluster_num = cluster_num
        self.f_repl = f_repl
        self.f_faire = f_faire
        self.f_GC = f_GC
        self.f_Ncov = f_Ncov
        self.f_extracov_bed_list = f_extracov_bed_list
        self.ref_fasta = ref_fasta
        self.bin_width = bin_width
        self.wgs = wgs
        if SNP_expansion_radius is None:
            self.SNP_expansion_radius = 10000 if not self.wgs else 0
        else:
            self.SNP_expansion_radius = SNP_expansion_radius
        self.region_blacklist_bed = region_blacklist_bed

        # lnp hyperparameters - can make passable by arguments
        self.alpha_prior = 1e-5
        self.beta_prior = 4e-3
        self.lamda = 1e-10

        self.allelic_clusters = np.load(f_allelic_clusters)
        with open(f_segs, "rb") as f:
            self.segmentations = pickle.load(f)
        if allelic_sample is not None:
            self.allelic_sample = allelic_sample
        else:
            self.allelic_sample = np.argmax(self.allelic_clusters["likelihoods"])
        # coverage input is expected to be a df file with columns: ["chr", "start", "end", "covcorr", "covraw"]
        self.full_cov_df = self.load_coverage(coverage_csv)
        self.SNPs = self.load_SNPs(f_SNPs)

        self.model = None

        # combine allelic segments and coverage bins
        self.aseg_cov_df = self.assign_clusters()

        # load and filter covariates
        self.aseg_cov_df = self.load_covariates(self.aseg_cov_df)
        self.aseg_cov_df = self.filter_and_scale_covariates(self.aseg_cov_df)

    def run_all_clusters(self):
        Pi, r, C, filtered_cov_df = self.assign_clusters()
        # run coverage mcmc on all clusters
        # assign coverage bins to allelic clusters from the specified allelic sample (if specified; o.w. random choice)
        cov_mcmc = NB_MCMC_AllClusters(self.num_draws, r, C, Pi)

        # either way we run and save results
        cov_mcmc.run()
        self.model = cov_mcmc
        segment_samples, global_beta, mu_i_samples = cov_mcmc.prepare_results()
        return segment_samples, global_beta, mu_i_samples, filtered_cov_df

    # Do preprocessing for running on each ADP cluster individually
    def prepare_single_cluster(self):
        Pi, r, C, filtered_cov_df = self.make_regressors(self.aseg_cov_df)
        pois_regr = PoissonRegression(r, C, Pi, log_exposure = np.log(self.bin_width))
        all_mu, global_beta = pois_regr.fit()
        
        # filter bins from each segment based on lnp convergence
        full_mask = filter_segments(Pi, r, C, global_beta, all_mu, exposure=np.log(self.bin_width), alpha_prior = self.alpha_prior, beta_prior = self.beta_prior, lamda=self.lamda)
        
        # save these results to a numpy object
        return Pi[full_mask], r[full_mask], C[full_mask], all_mu, global_beta, filtered_cov_df.loc[full_mask], self.allelic_sample

    def load_coverage(self, coverage_csv):
        Cov = pd.read_csv(coverage_csv, sep="\t", names=["chr", "start", "end", "covcorr", "mean_frag_len", "std_frag_len", "num_frags", "tot_reads", "fail_reads"], low_memory=False)
        Cov.loc[Cov['chr'] == 'chrM', 'chr'] = 'chrMT' #change mitocondrial contigs to follow mut conventions
        Cov["chr"] = mut.convert_chr(Cov["chr"])
        Cov = Cov.loc[Cov["chr"] != 0]
        Cov=Cov.reset_index(drop=True)
        Cov["start_g"] = seq.chrpos2gpos(Cov["chr"], Cov["start"], ref = self.ref_fasta)
        Cov["end_g"] = seq.chrpos2gpos(Cov["chr"], Cov["end"], ref = self.ref_fasta)

        # filter bins with no reads
        Cov = Cov.loc[(Cov['tot_reads'] > 0) & (Cov['num_frags'] > 0)]

        # filter bins with an excessive number of failing reads
        Cov["fail_reads_zt"] = zt(Cov["fail_reads"]/Cov["tot_reads"])
        Cov = Cov.loc[Cov["fail_reads_zt"] < 3]

        # midpoint of coverage bins will be used to map bins to covariates
        Cov["midpoint"] = ((Cov["end"] + Cov["start"])/2).astype(int)

        # scale coverage in units of fragments, rather than bases in order to
        # correctly model Poisson noise
        with pd.option_context('mode.chained_assignment', None): # suppress erroneous SettingWithCopyWarning
            Cov["fragcorr"] = np.round(Cov["covcorr"]/Cov["mean_frag_len"].mean())
        
        # remove any fragcorr zero bins
        Cov = Cov.loc[Cov.fragcorr > 0]

        # blacklist regions (if specified)
        if self.region_blacklist_bed is not None:
            B = pd.read_csv(self.region_blacklist_bed, sep = "\t", names = ["chr", "start", "end", "reason"])
            B["chr"] = mut.convert_chr(B["chr"])
            B = B.sort_values(["chr", "start", "end"])

            # if midpoint of coverage bin is in blacklisted region, remove it
            tidx = mut.map_mutations_to_targets(Cov, B, inplace = False, poscol = "midpoint")
            Cov = Cov.drop(Cov.index[tidx.index])

        return Cov.reset_index(drop = True)

    def load_SNPs(self, f_snps):
        SNPs = pd.read_pickle(f_snps)

        # annotate SNPs with allelic clusters
        clust_choice = self.allelic_clusters["snps_to_clusters"][self.allelic_sample]
        clust_u, clust_uj = np.unique(clust_choice, return_inverse=True)
        clust_uj = clust_uj.reshape(clust_choice.shape)
        SNPs["clust_choice"] = clust_uj

        # annotate SNPs with allelic segments
        seg_bdy = np.r_[0, list(self.segmentations[self.allelic_sample].keys()), len(SNPs)]
        seg_bdy = np.c_[seg_bdy[:-1], seg_bdy[1:]]
        SNPs["seg_idx"] = 0
        self.full_cov_df["allelic_seg_overlap"] = -1
        for i, (st, en) in enumerate(seg_bdy):
            SNPs.iloc[st:en, SNPs.columns.get_loc("seg_idx")] = i

            # add allelic segment boundaries to full coverage dataframe
            st_g, en_g = SNPs.iloc[[st, en - 1], SNPs.columns.get_loc("pos_gp")]
            self.full_cov_df.loc[(self.full_cov_df["start_g"] >= st_g) & (self.full_cov_df["end_g"] <= en_g), "allelic_seg_overlap"] = i

        # pad WES targets by +-300b when mapping SNPs to catch flanking coverage
        # 300b == average fragment size
        if not self.wgs:
            # first, map to regular target boundaries
            mut.map_mutations_to_targets(SNPs, self.full_cov_df)

            # map any unmapped SNPs to extended target boundaries, without exceeding Allelic segment Intervals
            AI = self.full_cov_df.groupby("allelic_seg_overlap").agg({ "start_g" : min, "end_g" : max })
            AI.loc[-1, :] = np.nan
            self.full_cov_df["start_pad"] = seq.gpos2chrpos(np.maximum(
              self.full_cov_df["start_g"].values - 300,
              AI.loc[self.full_cov_df["allelic_seg_overlap"], "start_g"].values
            ))[1]
            self.full_cov_df["end_pad"] = seq.gpos2chrpos(np.minimum(
              self.full_cov_df["end_g"].values + 300,
              AI.loc[self.full_cov_df["allelic_seg_overlap"], "end_g"].values
            ))[1]
            tidx_ext = mut.map_mutations_to_targets(SNPs, self.full_cov_df, startcol = "start_pad", endcol = "end_pad", inplace = False).astype(int)
            unmap_idx = SNPs.index.isin(tidx_ext.index) & (SNPs["targ_idx"] == -1)
            SNPs.loc[unmap_idx, "targ_idx"] = tidx_ext.loc[unmap_idx]
        else:
            mut.map_mutations_to_targets(SNPs, self.full_cov_df)
        return SNPs

    def generate_GC(self, cov_df):
        #grab fasta object from seq to avoid rebuilding
        seq.set_reference(self.ref_fasta)
        F = seq._fa.ref_fa_obj
        cov_df['C_GC'] = np.nan
        
        #this indexing assumes 0-indexed start and end cols
        for (i, chrm, start, end) in tqdm.tqdm(cov_df[['chr', 'start','end']].itertuples(), total = len(cov_df)):
            cov_df.iat[i, -1] = F[chrm-1][start:end+1].gc

    def load_covariates(self, cov_df):
        cov_df = cov_df.copy()
        ## Target size
        
        if not self.wgs:
            # we only need bin size if doing exomes
            cov_df["C_log_len"] = np.log(cov_df["end"] - cov_df["start"] + 1)
            # for wgs these will all be the same and we must remove

        ## Fragment length

        # some bins have zero mean fragment length; these bins are bad and should be removed
        cov_df = cov_df.loc[(cov_df.mean_frag_len > 0) & (cov_df.std_frag_len > 0)].reset_index(drop = True)

        cov_df = cov_df.rename(columns = { "mean_frag_len" : "C_frag_len" })

        # generate on 5x and 11x scales
        swv = np.lib.stride_tricks.sliding_window_view
        fl = cov_df["C_frag_len"].values; fl[np.isnan(fl)] = 0
        wt = cov_df["num_frags"].values
        for scale in [5, 11]:
            fl_sw = swv(np.pad(fl, scale//2), scale)
            wt_sw = swv(np.pad(wt, scale//2), scale)
            conv = np.einsum('ij,ij->i', wt_sw, fl_sw)

            cov_df[f"C_frag_len_{scale}x"] = conv/wt_sw.sum(1)

        ### track-based covariates

        ## Replication timing

        # load repl timing
        F = pd.read_pickle(self.f_repl)
        # map targets to RT intervals
        tidx = mut.map_mutations_to_targets(cov_df, F, inplace=False, poscol = "midpoint")
        F = F.loc[tidx].set_index(tidx.index).iloc[:, 3:].rename(columns = lambda x : "C_RT-" + x)
        cov_df = pd.concat([cov_df, F], axis = 1)

        ## GC content

        # load GC content if we have it precomputed, otherwise generate it
        if self.f_GC is not None and os.path.exists(self.f_GC):
            print("Using precomputed GC content", file = sys.stderr)
            B = pd.read_pickle(self.f_GC)
            
            cov_df = cov_df.merge(B.rename(columns={"gc": "C_GC"}), left_on=["chr", "start", "end"],
                                                  right_on=["chr", "start", "end"], how="left")
        else:
            print("Computing GC content", file = sys.stderr)
            self.generate_GC(cov_df)

        ## FAIRE
        if self.f_faire is not None:
            F = pd.read_pickle(self.f_faire)

            # map targets to FAIRE intervals
            tidx = mut.map_mutations_to_targets(cov_df, F, inplace=False, poscol = "midpoint")
            F = F.loc[tidx].set_index(tidx.index).iloc[:, 3:].rename(columns = lambda x : "C_FAIRE-" + x)
            cov_df = pd.concat([cov_df, F], axis = 1)

        ## (panel of) normal coverage
        if self.f_Ncov is not None:
            Ncov = self.load_coverage(self.f_Ncov)
            cov_df = cov_df.merge(
              Ncov.loc[:, ["start_g", "end_g", "covcorr"]],
              left_on = ["start_g", "end_g"],
              right_on = ["start_g", "end_g"],
              how = "left",
              suffixes = (None, "_N")
            )
            cov_df = cov_df.rename(columns = { "covcorr_N" : "C_normcov0" })

        ## extra covariates
        if self.f_extracov_bed_list is not None:
            print("Loading additional covariates ...", file = sys.stderr)
            with open(self.f_extracov_bed_list, "r") as f:
                for i, bed_file in tqdm.tqdm(enumerate(f.readlines())):
                    extracov_df = pd.read_csv(bed_file.rstrip(), sep = "\t", header = None).rename(columns = { 0 : "chr", 1 : "start", 2 : "end" })
                    extracov_df["chr"] = mut.convert_chr(extracov_df["chr"])

                    tidx = mut.map_mutations_to_targets(cov_df, extracov_df, inplace=False, poscol = "midpoint")
                    extracov_df = extracov_df.loc[tidx].set_index(tidx.index).iloc[:, 3:].rename(columns = lambda x : f"C_extracov-{i},{x - 3}")
                    cov_df = pd.concat([cov_df, extracov_df], axis = 1)

        return cov_df

    def filter_and_scale_covariates(self, cov_df):
        Cslice = cov_df.loc[:, cov_df.columns.str.contains("^C_")]

        ## scale covariates

        # FAIRE, fragment length, normal coverage, and any extra covariates get log z-transformed
        lztcols = Cslice.columns.str.contains("FAIRE|frag_len|normcov|extracov")
        cov_df = pd.concat([
          cov_df,
          Cslice.loc[:, lztcols].apply(lambda x : zt(np.log(x + 1))).rename(columns = lambda x : x + "_lz")
        ], axis = 1)

        # RT and GC get z-transformed sans log (they are already proportional to log coverage)
        ztcols = Cslice.columns.str.contains("C_(?:RT|GC)")
        cov_df = pd.concat([
          cov_df,
          Cslice.loc[:, ztcols].apply(zt).rename(columns = lambda x : x + "_z")
        ], axis = 1)

        # GC content follows a roughly quadratic relationship with coverage,
        # but only WGS has bins with high enough GC content that this is noticeable.
        if self.wgs:
            cov_df["C_GC2_z"] = cov_df["C_GC_z"]**2

        # log bin length gets zero-centered (so that we can see how much it deviates from 1 [perfect proportionality to target length])
        if "C_log_len" in cov_df.columns:
            cov_df["C_log_len"] -= cov_df["C_log_len"].mean()

        ## drop faulty covariates that may be all nan with a warning
        all_nan_cols = cov_df.columns[cov_df.isna().all(0)]
        if len(all_nan_cols):
            print("WARNING: detected covarriate with unusable values")
            cov_df = cov_df.drop(all_nan_cols, axis=1)
        ## filter covariates
        Cslice = cov_df.loc[:, cov_df.columns.str.contains("(?:^C_.*z$|C_log_len)")]

        # no NaN covariates
        naidx = Cslice.isna().any(1)

        # coverage outliers
        outlier_mask = find_outliers(cov_df["fragcorr"].values)
 
        # remove covariate outliers (+- 6 sigma)
        z_norm_columns = Cslice.columns.str.contains("^C_.*z$")
        covar_6sig_idx = (Cslice.loc[:, z_norm_columns].abs() < 6).all(axis = 1)
        
        # apply all filters
        cov_df = cov_df.loc[~naidx & ~outlier_mask & covar_6sig_idx]
        
        # remove outlier bins at ends of segments
        outlier_idxs = edge_outliers(cov_df)
        cov_df = cov_df.loc[~cov_df.index.isin(outlier_idxs)]

        # now remove ADP clusters that are too small (along with respective bins)
        small_clusters = cov_df.seg_idx.value_counts()[lambda x: x < 5].index
        small_adp_mask = cov_df.seg_idx.isin(small_clusters)
        cov_df = cov_df.loc[~small_adp_mask]

        return cov_df

    # use SNP cluster assignments from the given draw assign coverage bins to clusters
    # clusters with snps from different clusters are probabilistically assigned
    # method returns coverage df with only bins that overlap snps
    def assign_clusters(self):
        ## assign coverage intervals to allelic clusters and segments
        self.full_cov_df["seg_idx"] = -1
        self.full_cov_df["allelic_cluster"] = -1

        print("Mapping SNPs to targets ...", file = sys.stderr)
        for targ, D in tqdm.tqdm(self.SNPs.groupby("targ_idx")[["clust_choice", "seg_idx"]]):
            if targ == -1: # SNP does not overlap a coverage bin
                continue

            clust_idx = D["clust_choice"].values
            seg_idx = D["seg_idx"].values
            # all SNPs in this coverage bin have to be assigned to the same allelic segment
            if len(seg_idx) == 1 or (seg_idx[0] == seg_idx).all(): # short circuit second condition for efficiency
                self.full_cov_df.at[targ, "seg_idx"] = seg_idx[0]
                self.full_cov_df.at[targ, "allelic_cluster"] = clust_idx[0]
            # otherwise, we don't consider this coverage bin

        ## add allelic counts to each coverage bin

        # fix phases based on the cluster choice
        phases = self.allelic_clusters["snps_to_phases"][self.allelic_sample]
        self.SNPs.loc[:, ["min_ph", "maj_ph"]] = self.SNPs.loc[:, ["min", "maj"]].values[
          np.c_[0:len(self.SNPs)],
          np.c_[[0, 1], [1, 0]][phases.astype(int)]
        ]

        self.full_cov_df = self.full_cov_df.merge(
          self.SNPs.groupby("targ_idx")[["min_ph", "maj_ph"]].sum(),
          left_index = True, right_index = True,
          how = "left"
        ).rename(columns = { "min_ph" : "min_count", "maj_ph" : "maj_count" })

        ## assign coverage bins within <SNP_expansion_radius> of each bin overlapping a SNP to its allelic segment
        # this is to get
        # a) better total copy ratio estimates for whole exomes, by utilizing more bins
        # b) comprehensive total copy ratio for genomes (to include bins that lack SNPs,
        #    to catch focal events that may not have allelic information available)
        if self.SNP_expansion_radius > 0:
            # make sure that SNP radii don't exceed the boundaries of their respective Segment Intervals
            SI = self.SNPs.groupby("seg_idx")["pos_gp"].agg([min, max])
            T = pd.DataFrame({
              "start" : np.maximum(
                self.SNPs["pos_gp"].values - self.SNP_expansion_radius,
                SI.loc[self.SNPs["seg_idx"], "min"].values
              ),
              "end" : np.minimum(
                self.SNPs["pos_gp"].values + self.SNP_expansion_radius,
                SI.loc[self.SNPs["seg_idx"], "max"].values
              ),
              "seg_idx" : self.SNPs["seg_idx"]
            })

            # collapse overlapping intevals
            # index disjoint intervals, so that ...
            djidx = np.r_[-1, np.flatnonzero(T.iloc[1:]["start"].values >= T.iloc[:-1]["end"].values), len(T) - 1]

            # ... overlapping intervals will span all intervals between disjoint ones
            To = pd.DataFrame({
              "chr" : seq.gpos2chrpos(T["start"].iloc[djidx[:-1] + 1].values)[0],
              "start" : seq.gpos2chrpos(T["start"].iloc[djidx[:-1] + 1].values)[1],
              "end" : seq.gpos2chrpos(T["end"].iloc[djidx[1:]].values)[1],
              "seg_idx" : T["seg_idx"].iloc[djidx[:-1] + 1].values
            })
            # TODO: extend intervals to midpoints of gaps?

            # map midpoints of coverage bins to SNPs with radius +- SNP_expansion_radius
            tidx = mut.map_mutations_to_targets(self.full_cov_df, To, inplace = False, poscol = "midpoint")
            tidx = tidx.loc[self.full_cov_df.loc[tidx.index, "seg_idx"] == -1]
            self.full_cov_df.loc[tidx.index, "seg_idx"] = To.loc[tidx, "seg_idx"].values

            # set allelic counts to 0 for these coverage bins, since they don't actually contain SNPs
            self.full_cov_df.loc[tidx.index, "min_count"] = 0
            self.full_cov_df.loc[tidx.index, "maj_count"] = 0

        ## subset to targets containing SNPs
        Cov_overlap = self.full_cov_df.loc[self.full_cov_df["seg_idx"] != -1, :]

        ## add allelic cluster annotations to expanded allelic segments
        acmap = Cov_overlap.loc[
          Cov_overlap["allelic_cluster"] != -1, ["seg_idx", "allelic_cluster"]
        ].drop_duplicates().set_index("seg_idx")

        ## some expanded bins may come from segments without any covered bins. drop these
        Cov_overlap = Cov_overlap.loc[Cov_overlap["seg_idx"].isin(acmap.index)]

        with pd.option_context('mode.chained_assignment', None): # suppress erroneous SettingWithCopyWarning
            Cov_overlap.loc[:, "allelic_cluster"] = acmap.loc[Cov_overlap["seg_idx"]].values

        return Cov_overlap

    def make_regressors(self, cov_df):
        ## making regressor vector/covariate matrix

        # sort by genomic coordinates
        cov_df = cov_df.sort_values("start_g", ignore_index = True)

        # regressor
        r = np.c_[cov_df["fragcorr"]]

        # intercept matrix (one intercept per allelic segment)
        Pi = np.zeros([len(cov_df), cov_df["seg_idx"].max() + 1], dtype = np.float16)
        Pi[np.r_[0:len(cov_df)], cov_df["seg_idx"]] = 1
        Pi = Pi[:, Pi.sum(0) > 0] # prune zero columns in Pi (allelic segments that got totally eliminated)
        
        # remove non z-transformed covar cols
        drop_cols = list(cov_df.columns[cov_df.columns.str.contains("^C_.*") & 
                                       ~cov_df.columns.str.contains("^C_.*z$|^C_frag_len$|^C_log_len")])
        # also remove now useless coverage info
        drop_cols += ['covcorr', 'C_frag_len', 'std_frag_len','num_frags', 'tot_reads', 'fail_reads', 'fail_reads_zt']
        cov_df = cov_df.drop(drop_cols, axis=1)
        
        # covariate matrix
        col_idx = cov_df.columns.str.contains("(?:^C_.*z$|C_log_len)")
        sorted_covar_columns = sorted(cov_df.columns[col_idx])
        C = np.c_[cov_df[sorted_covar_columns]]

        return Pi, r, C, pd.concat([cov_df.loc[:, ~col_idx], cov_df.loc[:, sorted_covar_columns]], axis = 1)

# function tries to optimize the lnp, iteratively removing the most unlikely
# bins until convergence or 5% of bins are thrown out. returns a boolean mask 
# of length n, masking thrown out bins.
def poisson_outlier_filter(r, C, beta, mu_prior=None, exposure=0., alpha_prior=1e-5, beta_prior=4e-3, lamda=1e-10):
    r = r.flatten()
    # set max removals to 5%, if exceeded the segment is thrown out
    max_idxs_to_remove = max(2, int(len(r) / 20))
    # filter based on the corrected coverage, so compute this
    residuals = np.exp(np.log(r) - (C @ beta).flatten())
    pois_log_liks = stats.poisson(mu = residuals.mean()).logpmf(residuals.astype(int))
    idxs_to_remove = np.argsort(pois_log_liks)
    mask = np.ones(len(r), dtype=bool)
    mu_prior=mu_prior if mu_prior is not None else np.log(r[mask]).mean()
    # first try to see if we can converge without removing anything
    try:
        lnp = CovLNP_NR_prior(r[mask,None], beta, C[mask], exposure = exposure, alpha_prior = alpha_prior, beta_prior=beta_prior, mu_prior = mu_prior, lamda=lamda, init_prior = False)
        lnp.fit()
        return mask
    except:
        pass
    
    # if not we try removing one bin at a time util convergence or the threshold
    for idx_del in idxs_to_remove[:max_idxs_to_remove]:
        mask[idx_del] = False
        lnp = CovLNP_NR_prior(r[mask,None], beta, C[mask], exposure = exposure, alpha_prior = alpha_prior, beta_prior=beta_prior, mu_prior = mu_prior, lamda=lamda, init_prior = False)
        try:
            lnp.fit()
            #if we fit properly, then return mask
            return mask
        except:
            continue
    return np.zeros(len(r), dtype=bool)

# runs poisson filtering on each segment, returning a final mask over all bins
def filter_segments(Pi, r, C, beta, mus, exposure=0., alpha_prior = 1e-5, beta_prior=4e-3, lamda=1e-10):
    mask_lst = []
    seg_labels = np.argmax(Pi, 1)
    mu_arr = mus.flatten()
    print("filtering outlier bins from segments...")
    for i, seg_idx in tqdm.tqdm(enumerate(sorted(np.unique(seg_labels)))):
        seg_mask = seg_labels==seg_idx
        mask = poisson_outlier_filter(r[seg_mask, :], C[seg_mask], beta, mu_prior=mu_arr[i], exposure = exposure, alpha_prior = alpha_prior, beta_prior=beta_prior, lamda=lamda)
        mask_lst.append(mask)
    final_mask = np.concatenate(mask_lst)
    print(f"filtered {(~final_mask).sum()} bins")
    return final_mask

#TODO switch to lnp
# function for fitting nb model without covariates
def fit_nb(r):
    endog = r.flatten()
    exog = np.ones(len(r))
    sNB = statsNB(endog, exog)
    res = sNB.fit(disp=0)
    mu = res.params[0]; lepsi = -np.log(res.params[1])
    return mu, lepsi

# function for computing log survivial function values
def scipy_sf(r, mu, epsi):
    r = r.flatten()
    mu= mu.flatten()
    epsi = epsi.flatten()
    exp = np.exp(mu).flatten()
    return stats.nbinom.logsf(r, epsi, (1-(exp / (exp + epsi))))

# function for computing log cdf values
def scipy_cdf(r, mu, epsi):
    r = r.flatten()
    mu= mu.flatten()
    epsi = epsi.flatten()
    exp = np.exp(mu).flatten()
    return stats.nbinom.logcdf(r, epsi, (1-(exp / (exp + epsi))))

# function for finding nb outliers based on input log threshold
def find_outliers(r, thresh=-25):
    mu, lepsi = fit_nb(r)
    logsf = scipy_sf(r, mu, np.exp(lepsi))
    logcdf = scipy_cdf(r, mu, np.exp(lepsi))
    outliers = np.logical_or(logcdf < thresh, logsf < thresh)
    if outliers.sum() > len(r) * 0.05:
        raise ValueError("greater than 5% of bins considered outliers")
    return outliers

# function for sorting file strings by the cluster number rather than alphanumeric
def nat_sort(lst): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(lst, key=alphanum_key)

# find outliers at ends of allelic segments using segment mean +- sigma * segment_std
# as threshold, excluding the first and last two bins. bins are only thrown out 
# if the second to the end bin does not cross the threshold
def edge_outliers(cov_df, sigma=5):
    idx_lst = []
    for idx, seg_df in list(cov_df.groupby('seg_idx')):
        if len(seg_df) < 5:
            # will filter segment anyways
            continue
        f_arr = seg_df.fragcorr.values
        seg_mean = f_arr[2:].mean()
        seg_std = f_arr[2:].std()
        if f_arr[0] < seg_mean - sigma * seg_std or f_arr[0] > seg_mean + sigma * seg_std:
            # check if second bin is also outlier
            if not(f_arr[1] < seg_mean - sigma * seg_std or f_arr[1] > seg_mean + sigma * seg_std):
                idx_lst.append(seg_df.index[0])
        # check end of segment
        seg_mean = f_arr[:-2].mean()
        seg_std = f_arr[:-2].std()
        if f_arr[-1] < seg_mean - sigma * seg_std or f_arr[-1] > seg_mean + sigma * seg_std:
            # check if second to last bin is also outlier
            if not(f_arr[-2] < seg_mean - sigma * seg_std or f_arr[-2] > seg_mean + sigma * seg_std):
                idx_lst.append(seg_df.index[-1])
    return idx_lst

    
# function for collecting coverage mcmc results from each ADP cluster
def aggregate_clusters(seg_indices_pickle=None, coverage_dir=None, f_file_list=None, cov_df_pickle=None, bin_width=1):
    if coverage_dir is None and f_file_list is None:
        raise ValueError("need to pass in either coverage_dir or file_list txt file!")
    if coverage_dir is not None and f_file_list is not None:
        raise ValueError("need to pass in either coverage_dir or file_list txt file!, got both!")

    # get results files from the directory provided or from the file list provided
    if coverage_dir is not None:
        seg_files = nat_sort(glob.glob(os.path.join(coverage_dir, 'cov_mcmc_data_allelic_seg*')))
        cov_df = pd.read_pickle(os.path.join(coverage_dir, 'cov_df.pickle'))
        
    else:
        if cov_df_pickle is None:
            raise ValueError("Need to pass in cov_df file")
        file_ext = os.path.splitext(os.path.basename(f_file_list))[1]
        if file_ext == '.txt':
            # read in files from f_file_list
            read_files = []
            with open(f_file_list, 'r') as f:
                all_lines = f.readlines()
                for l in all_lines:
                    to_add = l.rstrip('\n')
                    if to_add != "nan":
                        read_files.append(to_add)
        
            seg_files = nat_sort(read_files)
        elif file_ext == '.npz':
            # handle case of single input file which is not in a txt list
            seg_files = [f_file_list]
        else:
            raise ValueError(f"Could not process file list {f_file_list}")
    
        seg_idxs = []
        for f in seg_files:
            search = re.search(".*allelic_seg_(\d+).npz.*", f)
            if search:
                seg_idxs.append(int(search.group(1)))
            
        cov_df = pd.read_pickle(cov_df_pickle)
    
    clust_assignments = cov_df['allelic_cluster'].values
    
    seg_data = pd.read_pickle(seg_indices_pickle)
    
    seg_results = {}
    mu_i_results = {}
    ll_results = {} 
    # load data from each cluster
    for seg, data_path in zip(seg_idxs, seg_files):
        cluster_data = np.load(data_path)
        seg_results[seg] = cluster_data['seg_samples']
        mu_i_results[seg] = cluster_data['mu_i_samples']
        ll_results[seg] = cluster_data['ll_samples']

    num_draws = seg_results[seg_idxs[0]].shape[1]
    num_clusters = len(seg_data.allelic_cluster.unique())
    num_segments = len(seg_results)
    
    # now we use these data to fill an overall coverage segmentation array
    coverage_segmentation = np.zeros((len(cov_df), num_draws))
    mu_i_values = np.zeros((len(cov_df), num_draws))

    for d in range(num_draws):
        global_counter = 0
        for seg in seg_idxs:
            seg_indices = seg_data.loc[seg].indices
            coverage_segmentation[seg_indices, d] = seg_results[seg][:,d] + global_counter
            mu_i_values[seg_indices, d] = mu_i_results[seg][:, d]
            global_counter += len(np.unique(seg_results[seg][:,d]))
    
    # generate data to re-compute global beta
    # calculate likelihoods of each sample
    ll_samples_arr = np.zeros((num_segments, num_draws))
    for ID, (seg, ll_arr) in enumerate(ll_results.items()):
        ll_samples_arr[ID] = ll_arr
    
    ll_samples = ll_samples_arr.sum(0)
    MAP_draw = np.nansum(ll_samples_arr, 0).argmax()
    MAP_seg = coverage_segmentation[:, MAP_draw]  
    # refitting beta
    r = np.c_[cov_df["fragcorr"]]
    ## create new intercept matrix using MAP coverage segmentation
    Pi = np.zeros((len(MAP_seg), int(MAP_seg.max()) + 1), dtype=np.float16)
    Pi[range(len(MAP_seg)), MAP_seg.astype(int)] = 1.
    ## generate covars
    covar_columns = sorted(cov_df.columns[cov_df.columns.str.contains("(?:^C_.*_l?z$|C_log_len)")])
    C = np.c_[cov_df[covar_columns]]
    ## do regression
    pois_regr = PoissonRegression(r, C, Pi)
    mu_refit, beta_refit = pois_regr.fit()

    return coverage_segmentation, mu_refit, beta_refit, ll_samples

def aggregate_burnin_files(file_list, cluster_num):
    file_captures = []
    for s in file_list:
        match = re.match(".*cluster_(\d*)_(\d*)-(\d*).npz", s)
        file_captures.append((int(match[1]), int(match[2]), int(match[3]), s))
    files_df = pd.DataFrame(file_captures).sort_values(by=[0,1])
    files_df = files_df.loc[files_df[0] == cluster_num]

    arrs=[]
    prev_max = 0
    prev_en = 0
    for i, _, st, en, path in files_df.itertuples():
        if st != prev_en:
            raise ValueError("missing a burnin file. please check input files")
        data = np.load(path)
        arr = data['seg_samples'][:,0]

        #set cluster assignments to be consistent with the previous clusters
        # we want the last cluster from the previous subset to be merged with the first subset of this interval
        arr += prev_max
        arrs.append(arr)

        prev_en = en
        prev_max = arr.max()
    reconciled_assignments = np.hstack(arrs)
    return reconciled_assignments
