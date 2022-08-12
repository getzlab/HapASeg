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
                 f_PoN=None,

                 num_draws=50,
                 cluster_num=None,
                 allelic_sample=None,
                 bin_width=1,
                 wgs=True
                 ):

        self.num_draws = num_draws
        self.cluster_num = cluster_num
        self.f_repl = f_repl
        self.f_faire = f_faire
        self.f_GC = f_GC
        self.f_Ncov = f_Ncov
        self.f_PoN = f_PoN
        self.ref_fasta = ref_fasta
        self.bin_width = bin_width
        self.wgs = wgs

        self.allelic_clusters = np.load(f_allelic_clusters)
        with open(f_segs, "rb") as f:
            self.segmentations = pickle.load(f)
        if allelic_sample is not None:
            self.allelic_sample = allelic_sample
        else:
            self.allelic_sample = np.argmax(self.allelic_clusters["likelihoods"])
        # coverage input is expected to be a df file with columns: ["chr", "start", "end", "covcorr", "covraw"]
        self.full_cov_df = self.load_coverage(coverage_csv)
        self.load_covariates()
        self.SNPs = self.load_SNPs(f_SNPs)

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

        # save these results to a numpy object
        return Pi, r, C, all_mu, global_beta, filtered_cov_df, self.allelic_sample

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
        
        return Cov

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
            self.full_cov_df["start_pad"] = seq.gpos2chrpos(np.maximum(
              self.full_cov_df["start_g"].values - 300,
              AI.loc[self.full_cov_df["allelic_seg_overlap"], "start_g"].values
            ))[1]
            self.full_cov_df["end_pad"] = seq.gpos2chrpos(np.minimum(
              self.full_cov_df["end_g"].values + 300,
              AI.loc[self.full_cov_df["allelic_seg_overlap"], "end_g"].values
            ))[1]
            tidx_ext = mut.map_mutations_to_targets(SNPs, self.full_cov_df, startcol = "start_pad", endcol = "end_pad", inplace = False)
            unmap_idx = SNPs.index.isin(tidx_ext.index) & (SNPs["targ_idx"] == -1)
            SNPs.loc[unmap_idx, "targ_idx"] = tidx_ext.loc[unmap_idx]
        else:
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
        # in case we are doing wgs these will all be the same and we must remove
        # since it will ruin beta fitting
        if (np.diff(self.full_cov_df["C_log_len"]) == 0).all():
            self.full_cov_df = self.full_cov_df.drop(['C_log_len'], axis=1)

        ## Fragment length

        # some bins have zero mean fragment length; these bins are bad and should be removed
        self.full_cov_df = self.full_cov_df.loc[(self.full_cov_df.mean_frag_len > 0) & (self.full_cov_df.std_frag_len > 0)].reset_index(drop = True)

        self.full_cov_df = self.full_cov_df.rename(columns = { "mean_frag_len" : "C_frag_len" })
        self.full_cov_df["C_frag_len_z"] = zt(np.log(self.full_cov_df["C_frag_len"]))

        # generate on 5x and 11x scales
        swv = np.lib.stride_tricks.sliding_window_view
        fl = self.full_cov_df["C_frag_len"].values; fl[np.isnan(fl)] = 0
        wt = self.full_cov_df["num_frags"].values
        for scale in [5, 11]:
            fl_sw = swv(np.pad(fl, scale//2), scale)
            wt_sw = swv(np.pad(wt, scale//2), scale)
            conv = np.einsum('ij,ij->i', wt_sw, fl_sw)

            self.full_cov_df[f"C_frag_len_{scale}x"] = conv/wt_sw.sum(1)
            self.full_cov_df[f"C_frag_len_{scale}x_z"] = zt(np.log(self.full_cov_df[f"C_frag_len_{scale}x"]))

        ## Failing read fraction
        # (not used as a covariate, but it makes sense to preprocess it here anyway)
        self.full_cov_df["fail_reads_zt"] = zt(self.full_cov_df["fail_reads"]/self.full_cov_df["tot_reads"])

        ### track-based covariates
        # use midpoint of coverage bins to map to intervals
        self.full_cov_df["midpoint"] = ((self.full_cov_df["end"] + self.full_cov_df["start"])/2).astype(int)

        ## Replication timing

        # load repl timing
        F = pd.read_pickle(self.f_repl)
        # map targets to RT intervals
        tidx = mut.map_mutations_to_targets(self.full_cov_df, F, inplace=False, poscol = "midpoint")
        F = F.loc[tidx].set_index(tidx.index).iloc[:, 3:].rename(columns = lambda x : "C_RT-" + x)
        self.full_cov_df = pd.concat([self.full_cov_df, F], axis = 1)

        # z-transform
        # note that log(RT) \propto coverage, so we merely z-transform without
        # taking any log here
        self.full_cov_df = pd.concat([
          self.full_cov_df,
          self.full_cov_df.loc[:, F.columns].apply(lambda x : zt(x)).rename(columns = lambda x : x + "_z")
        ], axis = 1)

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

        # GC content follows a roughly quadratic relationship with coverage
        self.full_cov_df["C_GC2"] = self.full_cov_df["C_GC"]**2

        ## FAIRE
        if self.f_faire is not None:
            F = pd.read_pickle(self.f_faire)

            # map targets to FAIRE intervals
            tidx = mut.map_mutations_to_targets(self.full_cov_df, F, inplace=False, poscol = "midpoint")
            F = F.loc[tidx].set_index(tidx.index).iloc[:, 3:].rename(columns = lambda x : "C_FAIRE-" + x)
            self.full_cov_df = pd.concat([self.full_cov_df, F], axis = 1)

            # z-transform
            self.full_cov_df = pd.concat([
              self.full_cov_df,
              self.full_cov_df.loc[:, F.columns].apply(lambda x : zt(np.log(x + 1))).rename(columns = lambda x : x + "_z")
            ], axis = 1)

        ## (panel of) normal coverage
        if self.f_Ncov is not None:
            Ncov = self.load_coverage(self.f_Ncov)
            self.full_cov_df = self.full_cov_df.merge(
              Ncov.loc[:, ["start_g", "end_g", "covcorr"]],
              left_on = ["start_g", "end_g"],
              right_on = ["start_g", "end_g"],
              how = "left",
              suffixes = (None, "_N")
            )
            self.full_cov_df = self.full_cov_df.rename(columns = { "covcorr_N" : "C_normcov0" })

        if self.f_PoN is not None:
            raise NotImplementedError("PoNs are not yet supported")

        # z-transform all normal coverage covariates, if they exist
        normcovcols = self.full_cov_df.columns[self.full_cov_df.columns.str.contains(r"^C_normcov\d")]
        if len(normcovcols):
            self.full_cov_df = pd.concat([
              self.full_cov_df,
              self.full_cov_df.loc[:, normcovcols].apply(lambda x : zt(np.log(x + 1))).rename(columns = lambda x : x + "_z"),
            ], axis = 1)

    # use SNP cluster assignments from the given draw assign coverage bins to clusters
    # clusters with snps from different clusters are probabliztically assigned
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

        ## assign coverage bins within 10kb of each bin overlapping a SNP to its allelic segment
        if not self.wgs: # TODO: always expand, and set threshold based on WGS/WES?
            max_dist = 10000

            # make sure that SNP radii don't exceed the boundaries of their respective Segment Intervals
            SI = self.SNPs.groupby("seg_idx")["pos_gp"].agg([min, max])
            T = pd.DataFrame({
              "chr" : self.SNPs["chr"],
              "start" : seq.gpos2chrpos(np.maximum(
                self.SNPs["pos_gp"].values - max_dist,
                SI.loc[self.SNPs["seg_idx"], "min"].values
              ))[1],
              "end" : seq.gpos2chrpos(np.minimum(
                self.SNPs["pos_gp"].values + max_dist,
                SI.loc[self.SNPs["seg_idx"], "max"].values
              ))[1],
              "seg_idx" : self.SNPs["seg_idx"]
            })

            # map midpoints of coverage bins to SNPs with radius +- max_dist
            tidx = mut.map_mutations_to_targets(self.full_cov_df, T, inplace = False, poscol = "midpoint")
            tidx = tidx.loc[self.full_cov_df.loc[tidx.index, "seg_idx"] == -1]
            self.full_cov_df.loc[tidx.index, "seg_idx"] = T.loc[tidx, "seg_idx"].values

            # set allelic counts to 0 for these coverage bins, since they don't actually contain SNPs
            self.full_cov_df.loc[tidx.index, "min_count"] = 0
            self.full_cov_df.loc[tidx.index, "maj_count"] = 0

        ## subset to targets containing SNPs
        Cov_overlap = self.full_cov_df.loc[self.full_cov_df["seg_idx"] != -1, :]

        ## scale coverage in units of fragments, rather than bases in order to
        ## correctly model Poisson noise
        with pd.option_context('mode.chained_assignment', None): # suppress erroneous SettingWithCopyWarning
            Cov_overlap["fragcorr"] = np.round(Cov_overlap["covcorr"]/Cov_overlap["C_frag_len"].mean())

        ## filtering
        # use all z-transformed covariates + non-scaled GC content+GC^2 + target length (if running on exomes)
        covar_columns = sorted(Cov_overlap.columns[Cov_overlap.columns.str.contains("^C_.*_z|^C_GC|^C_log_len$")])
        Cslice = Cov_overlap.loc[:, covar_columns]

        # no NaN covariates
        naidx = Cslice.isna().any(1)

        # remove bins with low quality reads (>3 sigma)
        lowqidx = Cov_overlap["fail_reads_zt"] > 3

        # coverage outliers
        outlier_mask = find_outliers(Cov_overlap["fragcorr"].values)
 
        # remove covariate outliers (+- 6 sigma)
        z_norm_columns = Cslice.columns[Cslice.columns.str.contains("^C_.*_z$")]
        covar_6sig_idx = (Cslice.loc[:, z_norm_columns].abs() < 6).all(axis = 1)

        # apply all filters
        Cov_overlap = Cov_overlap.loc[~naidx & ~lowqidx & ~outlier_mask & covar_6sig_idx]

        ## making regressor vector/covariate matrix

        # sort by genomic coordinates
        Cov_overlap = Cov_overlap.sort_values("start_g", ignore_index = True)

        # regressor
        r = np.c_[Cov_overlap["fragcorr"]]

        # intercept matrix (one intercept per allelic segment)
        Pi = np.zeros([len(Cov_overlap), Cov_overlap["seg_idx"].max() + 1], dtype = np.float16)
        Pi[np.r_[0:len(Cov_overlap)], Cov_overlap["seg_idx"]] = 1
        Pi = Pi[:, Pi.sum(0) > 0] # prune zero columns in Pi (allelic segments that got totally eliminated)

        # covariate matrix
        C = np.c_[Cov_overlap[covar_columns]]

        return Pi, r, C, Cov_overlap

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
        # read in files from f_file_list
        read_files = []
        with open(f_file_list, 'r') as f:
            all_lines = f.readlines()
            for l in all_lines:
                to_add = l.rstrip('\n')
                if to_add != "nan":
                    read_files.append(to_add)
        seg_files = nat_sort(read_files)
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
    r = np.c_[cov_df["covcorr"]]
    # we'll use the mu_is from the last segmentation sample
    mu_is = mu_i_values[:,-1]
    # compute new edogenous targets by subtracking out the mu_i values of the segments
    # along with the bin exposure
    endog = np.exp(np.log(r).flatten() - np.log(bin_width) - mu_is).reshape(-1,1)
    # generate covars
    covar_columns = sorted(cov_df.columns[cov_df.columns.str.contains("^C_.*_z$|^C_log_len$")])
    C = np.c_[cov_df[covar_columns]]
    # do regression
    pois_regr = PoissonRegression(endog, C, np.ones(endog.shape))
    mu_refit, beta_refit = pois_regr.fit()
    
    # calculate likelihoods of each sample
    ll_samples_arr = np.zeros((num_segments, num_draws))
    for seg, ll_arr in ll_results.items():
        ll_samples_arr[seg] = ll_arr
    
    ll_samples = ll_samples_arr.sum(0)

    return coverage_segmentation, beta_refit, ll_samples

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
