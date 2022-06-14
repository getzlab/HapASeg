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

from .NB_coverage_MCMC import NB_MCMC_AllClusters, NB_MCMC_SingleCluster
from .model_optimizers import PoissonRegression

class CoverageMCMCRunner:
    def __init__(self,
                 coverage_csv,
                 f_allelic_clusters,
                 f_SNPs,
                 f_segs,
                 ref_fasta,
                 f_repl,
                 f_faire,
                 f_GC=None,
                 num_draws=50,
                 cluster_num=None,
                 allelic_sample=None,
                 bin_width=1,
                 ):

        self.num_draws = num_draws
        self.cluster_num = cluster_num
        self.f_repl = f_repl
        self.f_faire = f_faire
        self.f_GC = f_GC
        self.ref_fasta = ref_fasta
        self.bin_width = bin_width

        self.allelic_clusters = np.load(f_allelic_clusters)
        with open(f_segs, "rb") as f:
            self.segmentations = pickle.load(f)
        # coverage input is expected to be a df file with columns: ["chr", "start", "end", "covcorr", "covraw"]
        self.full_cov_df = self.load_coverage(coverage_csv)
        self.load_covariates()
        self.SNPs = self.load_SNPs(f_SNPs)
        
        if allelic_sample is not None:
            self.allelic_sample = allelic_sample
        else:
            self.allelic_sample = np.argmax(self.allelic_clusters["likelihoods"])

        self.model = None

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
        Pi, r, C, filtered_cov_df = self.assign_clusters()
        pois_regr = PoissonRegression(r, C, Pi, log_exposure = np.log(self.bin_width))
        all_mu, global_beta = pois_regr.fit()
        pois_hess = pois_regr.hess()

        # save these results to a numpy object
        return Pi, r, C, all_mu, global_beta, filtered_cov_df, self.allelic_sample, pois_hess

    def load_coverage(self, coverage_csv):
        Cov = pd.read_csv(coverage_csv, sep="\t", names=["chr", "start", "end", "covcorr", "mean_frag_len", "std_frag_len", "num_reads"], low_memory=False)
        Cov.loc[Cov['chr'] == 'chrM', 'chr'] = 'chrMT' #change mitocondrial contigs to follow mut conventions
        Cov["chr"] = mut.convert_chr(Cov["chr"])
        Cov = Cov.loc[Cov["chr"] != 0]
        Cov=Cov.reset_index(drop=True)
        Cov["start_g"] = seq.chrpos2gpos(Cov["chr"], Cov["start"], ref = self.ref_fasta)
        Cov["end_g"] = seq.chrpos2gpos(Cov["chr"], Cov["end"], ref = self.ref_fasta)
        
        return Cov

    def load_SNPs(self, f_snps):
        SNPs = pd.read_pickle(f_snps)
        mut.map_mutations_to_targets(SNPs, self.full_cov_df)
        return SNPs

    def generate_GC(self):
        #grab fasta object from seq to avoid rebuilding
        seq.set_reference(self.ref_fasta)
        F = seq._fa.ref_fa_obj
        self.full_cov_df['C_GC'] = np.nan
        
        #this indexing assumes 0-indexed start and end cols
        for (i, chrm, start, end) in tqdm.tqdm(self.full_cov_df[['chr', 'start','end']].itertuples(), total = len(self.full_cov_df)):
            self.full_cov_df.iat[i, -1] = F[chrm-1][start:end+1].gc
        

    def load_covariates(self):
        zt = lambda x : (x - np.nanmean(x))/np.nanstd(x)

        ## Target size

        # we only need bin size if doing exomes but we can check by looking at the bin lengths
        self.full_cov_df["C_log_len"] = np.log(self.full_cov_df["end"] - self.full_cov_df["start"] + 1)
        self.full_cov_df["C_log_len_z"] = zt(self.full_cov_df["C_log_len"])
            
        # in case we are doing wgs these will all be the same and we must remove
        if (np.diff(self.full_cov_df["C_log_len"]) == 0).all():
            #remove the len col since it will ruin beta fitting
            self.full_cov_df = self.full_cov_df.drop(['C_log_len', 'C_log_len_z'], axis=1)

        ## Fragment length

        # some bins have zero mean fragment length; these bins are bad and should be removed
        self.full_cov_df = self.full_cov_df.loc[(self.full_cov_df.mean_frag_len > 0) & (self.full_cov_df.std_frag_len > 0)].reset_index(drop = True)

        self.full_cov_df = self.full_cov_df.rename(columns = { "mean_frag_len" : "C_frag_len" })
        self.full_cov_df["C_frag_len_z"] = zt(np.log(self.full_cov_df["C_frag_len"]))

        # generate on 5x and 11x scales
        swv = np.lib.stride_tricks.sliding_window_view
        fl = self.full_cov_df["C_frag_len"].values; fl[np.isnan(fl)] = 0
        wt = self.full_cov_df["num_reads"].values
        for scale in [5, 11]:
            fl_sw = swv(np.pad(fl, scale//2), scale)
            wt_sw = swv(np.pad(wt, scale//2), scale)
            conv = np.einsum('ij,ij->i', wt_sw, fl_sw)

            self.full_cov_df[f"C_frag_len_{scale}x"] = conv/wt_sw.sum(1)
            self.full_cov_df[f"C_frag_len_{scale}x_z"] = zt(np.log(self.full_cov_df[f"C_frag_len_{scale}x"]))

        ### track-based covariates
        # use midpoint of coverage bins to map to intervals
        self.full_cov_df["midpoint"] = ((self.full_cov_df["end"] + self.full_cov_df["start"])/2).astype(int)

        ## Replication timing

        # load repl timing
        F = pd.read_pickle(self.f_repl)
        # map targets to RT intervals
        tidx = mut.map_mutations_to_targets(self.full_cov_df, F, inplace=False, poscol = "midpoint")
        self.full_cov_df['C_RT'] = np.nan
        self.full_cov_df.iloc[tidx.index, -1] = F.iloc[tidx, 3:].mean(1).values

        # z-transform
        self.full_cov_df["C_RT_z"] = zt(np.log(self.full_cov_df["C_RT"]))

        ## GC content

        # load GC content if we have it precomputed, otherwise generate it
        if self.f_GC is not None and os.path.exists(self.f_GC):
            print("Using precomputed GC content", file = sys.stderr)
            B = pd.read_pickle(self.f_GC)
            
            self.full_cov_df = self.full_cov_df.merge(B.rename(columns={"gc": "C_GC"}), left_on=["chr", "start", "end"],
                                                  right_on=["chr", "start", "end"], how="left")
        else:
            print("Computing GC content", file = sys.stderr)
            self.generate_GC()
        
        self.full_cov_df["C_GC_z"] = zt(self.full_cov_df["C_GC"])

        ## FAIRE

        if self.f_faire is not None:
            F = pd.read_pickle(self.f_faire)
            # map targets to FAIRE intervals
            tidx = mut.map_mutations_to_targets(self.full_cov_df, F, inplace=False, poscol = "midpoint")
            self.full_cov_df['C_FAIRE'] = np.nan
            self.full_cov_df.iloc[tidx.index, -1] = F.iloc[tidx, -1].values

            # z-transform
            self.full_cov_df["C_FAIRE_z"] = zt(np.log(self.full_cov_df["C_FAIRE"] + 1))

    # use SNP cluster assignments from the given draw assign coverage bins to clusters
    # clusters with snps from different clusters are probabliztically assigned
    # method returns coverage df with only bins that overlap snps
    def assign_clusters(self):
        ## generate unique clust assignments
        clust_choice = self.allelic_clusters["snps_to_clusters"][self.allelic_sample]
        clust_u, clust_uj = np.unique(clust_choice, return_inverse=True)
        clust_uj = clust_uj.reshape(clust_choice.shape)
        self.SNPs["clust_choice"] = clust_uj

        ## assign coverage intervals to allelic clusters and segments 
        # get allelic segment boundaries
        seg_bdy = np.r_[0, list(self.segmentations[self.allelic_sample].keys()), len(self.SNPs)]
        seg_bdy = np.c_[seg_bdy[:-1], seg_bdy[1:]]
        self.SNPs["seg_idx"] = 0
        for i, (st, en) in enumerate(seg_bdy):
            self.SNPs.iloc[st:en, self.SNPs.columns.get_loc("seg_idx")] = i
        seg_max = self.SNPs["seg_idx"].max() + 1

        # assignment probabilities of each coverage interval -> allelic segment
        Cov_clust_probs = np.zeros([len(self.full_cov_df), seg_max])

        # first compute assignment probabilities based on the SNPs within each bin
        # segments just get assigned to the maximum probability
        self.full_cov_df["seg_idx"] = -1
        print("Mapping SNPs to targets ...", file = sys.stderr)
        for targ, D in tqdm.tqdm(self.SNPs.groupby("targ_idx")[["clust_choice", "seg_idx"]]):
            if targ == -1: # SNP does not overlap a coverage bin
                continue
            clust_idx = D["clust_choice"].values
            seg_idx = D["seg_idx"].values
            if len(seg_idx) == 1:
                Cov_clust_probs[targ, seg_idx] = 1.0
                self.full_cov_df.at[targ, "seg_idx"] = seg_idx[0]
            else: 
                targ_clust_hist = np.bincount(seg_idx, minlength = seg_max) 
                Cov_clust_probs[targ, :] = targ_clust_hist / targ_clust_hist.sum()
                self.full_cov_df.at[targ, "seg_idx"] = np.bincount(seg_idx).argmax()

        ## subset to targets containing SNPs
        overlap_idx = Cov_clust_probs.sum(1) > 0
#        # add targets within a 2 targ radius
#        overlap_idx = np.flatnonzero(Cov_clust_probs.sum(1) > 0)[:, None]
#        overlap_idx = overlap_idx + np.c_[-2:3].T
#        overlap_idx = np.sort(np.unique((overlap_idx + np.c_[-2:3].T).ravel()))
        Cov_clust_probs_overlap = Cov_clust_probs[overlap_idx, :]

        # zero out improbable assignments and re-normalilze
        Cov_clust_probs_overlap[Cov_clust_probs_overlap < 0.05] = 0
        Cov_clust_probs_overlap /= Cov_clust_probs_overlap.sum(1)[:, None]
        # prune empty clusters
        prune_idx = Cov_clust_probs_overlap.sum(0) > 0
        Cov_clust_probs_overlap = Cov_clust_probs_overlap[:, prune_idx]
        num_pruned_clusters = Cov_clust_probs_overlap.shape[1]

        ## subsetting to only targets that overlap SNPs
        Cov_overlap = self.full_cov_df.loc[overlap_idx, :]

        ## probabilistically assign each ambiguous coverage bin to a cluster
        # for now we will take maximum instead
        amb_mask = np.max(Cov_clust_probs_overlap, 1) != 1
        amb_assgn_probs = Cov_clust_probs_overlap[amb_mask, :]
        if amb_mask.sum() > 0:
            #new_assgn = np.array([np.random.choice(np.r_[:num_pruned_clusters],
            #                                       p=amb_assgn_probs[i]) for i in range(len(amb_assgn_probs))])
            new_assgn = np.array([np.argmax(amb_assgn_probs[i]) for i in range(len(amb_assgn_probs))])
            new_onehot = np.zeros((new_assgn.size, num_pruned_clusters))
            new_onehot[np.arange(new_assgn.size), new_assgn] = 1
    
            # update with assigned values
            Cov_clust_probs_overlap[amb_mask, :] = new_onehot

        ## downsampling for wgs
#        if len(Cov_clust_probs_overlap) > 20000:
#            downsample_mask = np.random.rand(Cov_clust_probs_overlap.shape[0]) < 0.2
#            Cov_clust_probs_overlap = Cov_clust_probs_overlap[downsample_mask]
#            Cov_overlap = Cov_overlap.iloc[downsample_mask]
    
        # remove clusters with fewer than 4 assigned coverage bins (remove these coverage bins as well)
        bad_clusters = Cov_clust_probs_overlap.sum(0) < 4
        bad_bins = Cov_clust_probs_overlap[:, bad_clusters].any(1) == 1
        filtered = Cov_clust_probs_overlap[~bad_bins, :][:, ~bad_clusters]

        Cov_overlap = Cov_overlap.loc[~bad_bins, :]
        Pi = filtered.copy()
        Cov_overlap['allelic_cluster'] = np.argmax(Pi, axis=1)
       
        r = np.c_[Cov_overlap["covcorr"]]
        
        covar_columns = sorted(Cov_overlap.columns[Cov_overlap.columns.str.contains("^C_.*_z$")])

        ## making covariate matrix
        C = np.c_[Cov_overlap[covar_columns]]

        ## dropping Nans
        naidx = np.isnan(C).any(axis=1)
        # drop zero coverage bins as well (this is to account for a bug in coverage collector) TODO: remove need for this
        naidx = np.logical_or(naidx, (r==0).flatten())
        r = r[~naidx]
        C = C[~naidx]
        Pi = Pi[~naidx]

        Cov_overlap = Cov_overlap.iloc[~naidx]
        
        ## removing coverage outliers
        outlier_mask = find_outliers(r)
        r = r[~outlier_mask]
        C = C[~outlier_mask]
        Pi = Pi[~outlier_mask]
        Cov_overlap = Cov_overlap.iloc[~outlier_mask]

        # some clusters may have been eliminated by this point; prune them from Pi
        Pi = Pi[:, Pi.sum(0) > 0]
 
        ## remove covariate outliers (+- 6 sigma)
        covar_outlier_idx = (Cov_overlap.loc[:, covar_columns].abs() < 6).all(axis = 1)
        Cov_overlap = Cov_overlap.loc[covar_outlier_idx]
        Pi = Pi[covar_outlier_idx, :]
        r = r[covar_outlier_idx]
        C = C[covar_outlier_idx, :]

        return Pi, r, C, Cov_overlap

# function for fitting nb model without covariates
def fit_nb(r):
    endog = r.flatten()
    exog = np.ones(r.shape[0])
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


# function for collecting coverage mcmc results from each ADP segment
def aggregate_adp_segments(allelic_seg_groups_pickle, coverage_dir=None, f_file_list=None, cov_df_pickle=None, bin_width=1):
    S = pd.read_pickle(allelic_seg_groups_pickle)
    S = S.rename_axis(index = "allelic_seg_idx").reset_index()

    if coverage_dir is None and f_file_list is None:
        raise ValueError("need to pass in either coverage_dir or file_list txt file!")
    if coverage_dir is not None and f_file_list is not None:
        raise ValueError("need to pass in either coverage_dir or file_list txt file!, got both!")

    # get results files from the directory provided or from the file list provided
    if coverage_dir is not None:
        adp_seg_files = nat_sort(glob.glob(os.path.join(coverage_dir, 'cov_mcmc_data_*')))
        cov_df = pd.read_pickle(os.path.join(coverage_dir, 'cov_df.pickle'))
        
    else:
        if cov_df_pickle is None:
            raise ValueError("Need to pass in cov_df file")
        # read in files from f_file_list
        read_files = []
        with open(f_file_list, 'r') as f:
            all_lines = f.readlines()
            for l in all_lines:
                to_add = l.rstrip('\n')
                if to_add != "nan":
                    read_files.append(to_add)
        adp_seg_files = nat_sort(read_files)
        cov_df = pd.read_pickle(cov_df_pickle)

    # make sure that number of results shards is consistent with shard indices
    if len(adp_seg_files) != len(S):
        raise ValueError("Number of ADP seg files does not match scatter shards!")

    # load in covMCMC segment boundaries and mu's for each ADP segment
    seg_results = []
    mu_i_results = []
    
    for f in adp_seg_files:
        seg_data = np.load(f)
        seg_results.append(seg_data['seg_samples'])
        mu_i_results.append(seg_data['mu_i_samples'])

    S["seg_results"] = seg_results
    S["mu_i_results"] = mu_i_results

    num_draws = seg_results[0].shape[1]

    # create overall segmentation array
    coverage_segmentation = np.zeros((len(cov_df), num_draws))
    mu_i_values = np.zeros((len(cov_df), num_draws))

    # loop over each cov MCMC draw
    # TODO: only use maximum likelihood draw; CDP should be able to resegment
    for d in range(num_draws):
        n_tot_segs = 0
        # loop over ADP segments
        for _, s in S.iterrows():
            seg_idxs = s["seg_results"][:, d]
            coverage_segmentation[s["indices"], d] = seg_idxs + n_tot_segs
            n_tot_segs += seg_idxs[-1] + 1

            mu_i_values[s["indices"], d] = s["mu_i_results"][:, d]

    # TEMP HACK: for now, only take iteration with fewest number of segments
    sidx = coverage_segmentation[-1, :].argmin()
    coverage_segmentation = coverage_segmentation[:, [sidx]]
    mu_i_values = mu_i_values[:, [sidx]]

    # remove short segments (<200Kb)
    # TODO: remove segments not well-modeled by covariates
    cov_df["cov_seg_idx"] = coverage_segmentation.astype(int)
    long_seg_idx = cov_df.groupby("cov_seg_idx").apply(lambda x : (x.iloc[-1]["end"] - x.iloc[0]["start"]) > 2e5).rename("seg_OK")
    cov_df = cov_df.merge(long_seg_idx, left_on = "cov_seg_idx", right_index = True)

    coverage_segmentation = coverage_segmentation[cov_df["seg_OK"], :]
    mu_i_values = mu_i_values[cov_df["seg_OK"], :]
    cov_df = cov_df.loc[cov_df["seg_OK"]]

    # recompute global beta
    r = np.c_[cov_df["covcorr"]]
    # we'll use the mu_is from the last segmentation sample
    mu_is = mu_i_values[:, [-1]]
    # generate covars
    covar_columns = sorted(cov_df.columns[cov_df.columns.str.contains("^C_.*_z$")])
    C = np.c_[cov_df[covar_columns]]
    # do regression
    pois_regr = PoissonRegression(r, C, np.ones(r.shape), np.log(bin_width) + mu_is)
    mu_refit, beta_refit = pois_regr.fit()
    
    return coverage_segmentation, beta_refit

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
