import colorama
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc
import pickle
import os
import distinctipy
import tqdm
from kneed import KneeLocator
from scipy.signal import find_peaks

from capy import seq, mut

from.model_optimizers import CovLNP_NR_prior

from .utils import *

colors = mpl.cm.get_cmap("tab20").colors


# method for concatenating dp draws into a single large df to be used by the acdp
def generate_acdp_df(SNP_path, # path to SNP df
                 ADP_path, # path to npz ADP result
                 cdp_object_path=None, # path to CDP runner pickle object
                 cdp_scatter_files=None, #path to CDP scattered pickle obj txt file
                 cov_df_path=None, #cov_df file if using segments directly from cov mcmc
                 cov_mcmc_data_path=None, #segmentation data if using segments from cov_mcmc
                 bin_width=1, #set to uniform bin width for wgs or 1 for exomes
                 ADP_draw_index=-1, # index of ADP draw used in Coverage MCMC
                 wgs=False # if wgs, will only create acdp df for best draw
                 ):

    SNPs = pd.read_pickle(SNP_path)
    ADP_clusters = np.load(ADP_path)
    alpha_prior = 1e-5
    beta_prior=4e-3
    
    # load data
    ## data can come from one of three sources, a CDP object containing all draws
    ## a list of scattered cdp files, or a cov_df and cov_mcmc object for going 
    ## directly from coverage segmentation to acdp
    dp_run_data = []
    if cdp_object_path is not None:
        with open(CDP_path, 'rb') as f:
            dp_pickle = pickle.load(f)
            DP_runs = dp_pickle.DP_runs
        for run_idx, dp_run in enumerate(DP_runs):
            dp_run_data.append((run_idx, dp_run.cov_df, dp_run.bins_to_clusters[-1]))
        beta = dp_pickle.beta
        best_run = np.argmax([dp.ll_history[-1] for dp in DP_runs])
    
        # limit to best draw
        if wgs:
            dp_run_data = [dp_run_data[best_run]]

    elif cdp_scatter_files is not None:
        #load in the file names
        with open(cdp_scatter_files, 'r') as f:
            cdp_file_list = f.read().splitlines()
        #fill in dp objects from the scatter jobs
        DP_lls = []
        dp_run_data = []
        for run_idx, cdp_file in enumerate(cdp_file_list):
             with open(cdp_file, 'rb') as cdp:
                dp_pickle = pickle.load(cdp)
                print("file {} : len dp runs: {}".format(cdp_file, len(dp_pickle.DP_runs)), flush=True)
                dp_run = dp_pickle.DP_runs[0]
                dp_run_data.append((run_idx, dp_run.cov_df, dp_run.bins_to_clusters[-1]))
                DP_lls.append(dp_run.ll_history[-1])
        beta = dp_pickle.beta
        best_run = np.argmax(DP_lls)
        
        if wgs:
            dp_run_data = [dp_run_data[best_run]]

    elif cov_df_path is not None and cov_mcmc_data_path is not None:
        seg_data = np.load(cov_mcmc_data_path)
        beta = seg_data['beta']
        seg_samples = seg_data['seg_samples']
        cov_df = pd.read_pickle(cov_df_path)
        best_run = np.argmax(seg_data['ll_samples'])
       
        # only gather data from best run if in wgs mode
        runs_considered = [best_run] if wgs else range(seg_samples.shape[1]) 
        
        for run in runs_considered:
            cov_df_run = cov_df.copy()
            cov_df_run['segment_ID'] = seg_samples[:, run].astype(int)
            dp_assignments = seg_samples[:,run]
            
            # filter out small segments (<10)            
            gb = cov_df_run.groupby('segment_ID').count().iloc[:,0]
            small_segs = gb.loc[gb < 10].index
            dp_assignments = dp_assignments[~cov_df_run.segment_ID.isin(small_segs)]
            cov_df_run = cov_df_run.loc[~cov_df_run.segment_ID.isin(small_segs)]

            dp_run_data.append((run, cov_df_run, dp_assignments))
    else:
        raise ValueError("must pass in either a cdp object path or a txt file containing a list of object paths")
    
    # save the hessians for bootstrapping in a dictionary
    data = {}
    # currently uses the last sample from every DP draw
    draw_dfs = []
    for draw_num, cov_df_run, dp_assignments in dp_run_data:
        print('concatenating dp run ', draw_num, flush=True)
        a_cov_seg_df = cov_df_run

        # columns containing z transformed covariates to use
        covar_cols = sorted(a_cov_seg_df.columns[a_cov_seg_df.columns.str.contains("(?:^C_.*z$|C_log_len)")])

        # add dp cluster annotations
        a_cov_seg_df['cov_DP_cluster'] = -1
        
        if cov_df_path is None:
        # use the segs to clusts data
            segs_to_clusts = dp_assignments
            for seg in range(len(segs_to_clusts)):
                a_cov_seg_df.loc[a_cov_seg_df['segment_ID'] == seg, 'cov_DP_cluster'] = segs_to_clusts[seg]
        
        else:
            # each segment becomes its own DP cluster
            a_cov_seg_df['cov_DP_cluster'] = a_cov_seg_df['segment_ID']

        # remove segments that were blacklisted in this draw
        a_cov_seg_df = a_cov_seg_df.loc[a_cov_seg_df['cov_DP_cluster'] != -1]

        # adding cluster mus and sigmas to df for each tuple
        # falls back to CDP mu and sigma if the tuple is too small (less than 10 coverage bins)
        a_cov_seg_df['cov_DP_mu'] = 0
        a_cov_seg_df['cov_DP_sigma'] = 0

        for adp, cdp in tqdm.tqdm(a_cov_seg_df.groupby(['allelic_cluster', 'cov_DP_cluster']).indices):
            acdp_clust = a_cov_seg_df.loc[
                (a_cov_seg_df.cov_DP_cluster == cdp) & (a_cov_seg_df.allelic_cluster == adp)]
            if len(acdp_clust) < 10:
                acdp_clust = a_cov_seg_df.loc[a_cov_seg_df.cov_DP_cluster == cdp]
            r = acdp_clust.fragcorr.values
            C = np.c_[acdp_clust[covar_cols]]
            lnp = CovLNP_NR_prior(r[:,None], beta, C, exposure = np.log(bin_width), alpha_prior = alpha_prior, beta_prior=beta_prior, mu_prior= r.mean(), lamda=1e-10, init_prior = False)
            res = lnp.fit(ret_hess=True)
            mu = res[0]
            a_cov_seg_df.loc[
                (a_cov_seg_df.cov_DP_cluster == cdp) & (a_cov_seg_df.allelic_cluster == adp), 'cov_DP_mu'] = mu
            #H = sNB.hessian(res.params)
            # variance of the mu posterior is taken as the inverse of the hessian component for mu
            sigma_hinv = np.linalg.inv(res[2])[1,1]
            
            #try propogating through sigma with proper change of variable
            mu_sigma = np.exp(res[1] - sigma_hinv**2)
            a_cov_seg_df.loc[(a_cov_seg_df.cov_DP_cluster == cdp) & (
                        a_cov_seg_df.allelic_cluster == adp), 'cov_DP_sigma'] = mu_sigma
            res = (res[0], res[1], res[2])
            data[(adp, cdp, draw_num)] = res
        # duplicate segments to account for second allele
        num_bins = len(a_cov_seg_df)
        a_cov_seg_df = a_cov_seg_df.reset_index(drop=True)
        a_cov_seg_df = a_cov_seg_df.append(a_cov_seg_df)
        a_cov_seg_df = a_cov_seg_df.reset_index(drop=True)
        a_cov_seg_df['allele'] = 0
        allele_col_idx = a_cov_seg_df.columns.get_loc('allele')

        # minor
        a_cov_seg_df.iloc[:num_bins, allele_col_idx] = -1
        # major
        a_cov_seg_df.iloc[num_bins:, allele_col_idx] = 1

        a_cov_seg_df['dp_draw'] = draw_num
        
        #draw_dfs.append(a_cov_seg_df.drop(covar_cols, axis=1))
        draw_dfs.append(a_cov_seg_df)

    print('completed ACDP dataframe generation')
    # return the acdp df, the hessians dictionary, and the index of the best DP run
    return pd.concat(draw_dfs), data,  best_run

def generate_unclustered_segs(filename, acdp_df, lnp_data, opt_idx):
    acdp_single = acdp_df.loc[acdp_df.dp_draw == opt_idx]
    num_bins = (acdp_single.allele==1).sum()
    row_idxs = [acdp_single.columns.get_loc(i) for i in ['chr', 'start', 'end', 'allelic_cluster', 'segment_ID', 'cov_DP_cluster', 'min_count', 'maj_count']]
    min_df = acdp_single.iloc[num_bins:, row_idxs]

    segs_df = pd.DataFrame.from_records(min_df.groupby(['chr', 'segment_ID']).apply(lambda x: {'chr': x.chr.values[0],
                                                               'start': x.start.min(), 
                                                                'end': x.end.max(), 
                                                                'allelic_cluster':x.allelic_cluster.values[0], 
                                                                'segment_ID': x.segment_ID.values[0],
                                                                'cov_DP_cluster':x.cov_DP_cluster.values[0], 
                                                                'min_count':x.min_count.sum(),
                                                                'maj_count':x.maj_count.sum()}).values)
    
    segs_df.loc[:, ['mu.major', 'mu.minor', 'sigma.major', 'sigma.minor']] = np.nan
    for i, row in segs_df.iterrows():
        lnp_res = lnp_data[(row[3], row[5], 2)]
        a,b = row[[6,7]]
        norm_samples = np.random.multivariate_normal(mean=(lnp_res[0], lnp_res[1]), cov = np.linalg.inv(-lnp_res[2]), size = 10000)
        r_maj = np.array(s.poisson.rvs(np.exp(s.norm.rvs(norm_samples[:,0], np.exp(norm_samples[:,1]))) * s.beta.rvs(a,b, size = 10000)))
        mu_maj, sigma_maj = r_maj.mean(), r_maj.std()
        r_min = np.array(s.poisson.rvs(np.exp(s.norm.rvs(norm_samples[:,0], np.exp(norm_samples[:,1]))) * s.beta.rvs(b,a, size = 10000)))
        mu_min, sigma_min = r_min.mean(), r_min.std()
        segs_df.loc[i, ['mu.major', 'mu.minor', 'sigma.major', 'sigma.minor']] = mu_maj, mu_min, sigma_maj, sigma_min
    segs_df = segs_df.rename(columns={'chr':'Chromosome', 'start':'Start.bp', 'end':'End.bp'})     
    segs_df.to_csv(filename, sep='\t', index=False)

class AllelicCoverage_DP_runner:
    def __init__(self, cov_df, beta, cytoband_file, lnp_data, wgs=False, draw_idx=None, seed_all_clusters=True):
        self.cov_df = cov_df
        self.beta = beta
        self.cytoband_file = cytoband_file
        self.seed_all_clusters = seed_all_clusters
        self.wgs = wgs
        self.draw_idx = draw_idx
        self.default_draw = draw_idx if draw_idx is not None else 0
        self.lnp_data = lnp_data

        # if draw_idx passed then only use those draws
        if draw_idx is not None:
            self.cov_df = self.cov_df.loc[self.cov_df.dp_draw == draw_idx]
        
        self.num_segments = len(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        self.segment_r_list = [None] * self.num_segments
        self.segment_V_list = np.zeros(self.num_segments)
        self.segment_counts = np.zeros(self.num_segments, dtype=int)
        self.segment_allele = np.zeros(self.num_segments, dtype=int)
        self.cluster_assignments = np.ones(self.num_segments, dtype=int) * -1
        self.segment_sums = np.zeros(self.num_segments)        
        self.segment_ssd = np.zeros(self.num_segments)
        self.greylist_segments = sc.SortedSet({})

        self._init_segments()
   
        single_allele = self.cov_df 
        self.acdp = None

    # initialize each segment object with its data
    def _init_segments(self):
        # keep a table of reads for each allelic cluster to fallback on if the tuple has too few bins (<10)
        fallback_counts = sc.SortedDict({})
        # keep track of acdp segment assignments for each bin
        self.cov_df.loc[:, 'acdp_segID'] = -1
        self.cov_df.loc[:, 'acdp_segID'] = self.cov_df['acdp_segID'].astype(int)

        for ID, (name, grouped) in enumerate(
                self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw'])):
            self.cov_df.loc[grouped.index, 'acdp_segID'] = ID

            mu = grouped['cov_DP_mu'].values[0]
            sigma = grouped['cov_DP_sigma'].values[0]
            group_len = len(grouped)
            if group_len > 10:
                major, minor = (grouped['maj_count'].sum(), grouped['min_count'].sum())
            else:
                ADP_clust = grouped['allelic_cluster'].values[0]
                if ADP_clust in fallback_counts:
                    major, minor = fallback_counts[ADP_clust]
                else:
                    filt = self.cov_df.loc[self.cov_df.allelic_cluster == ADP_clust]
                    major, minor = filt['maj_count'].sum(), filt['min_count'].sum()
                    fallback_counts[ADP_clust] = (major, minor)

            allele = grouped.allele.values[0]
            self.segment_allele[ID] = allele
            if group_len < 3:
                self.greylist_segments.add(ID)

            if allele == -1:
                a = minor
                b = major
            else:
                a = major
                b = minor
            # lin scale
            #r = np.array(np.exp(s.norm.rvs(mu, np.sqrt(sigma), size=group_len)) * s.beta.rvs(a, b, size=group_len))
            #logscale
            #r = np.array(s.norm.rvs(mu, np.sqrt(sigma), size=group_len) + np.log(s.beta.rvs(a, b, size=group_len)))
            # experiment 1
            #r = np.array(np.exp(s.norm.rvs(mu + np.log( s.beta.rvs(a, b, size=group_len)), np.sqrt(sigma), size=group_len)))
            # lnp
            #r = np.array(s.poisson.rvs(np.exp(s.norm.rvs(mu, sigma, size=group_len)) * s.beta.rvs(a, b, size=group_len)))
            
            # using joint
            lnp_res = self.lnp_data[(name[0], name[1], name[3])]
            
            norm_samples = np.random.multivariate_normal(mean=(lnp_res[0], lnp_res[1]), cov = np.linalg.inv(-lnp_res[2]), size = group_len)
            #r = np.array(s.poisson.rvs(np.exp(norm_samples[:,0]) * s.beta.rvs(a,b, size = group_len)))
            r = np.array(s.poisson.rvs(np.exp(s.norm.rvs(norm_samples[:,0], np.exp(norm_samples[:,1]))) * s.beta.rvs(a,b, size = group_len)))
            
            # linscale
            # V = (np.exp(s.norm.rvs(mu, np.sqrt(sigma), size=10000)) * s.beta.rvs(a, b, size=10000)).var()
            # logscale
            # V = (s.norm.rvs(mu, np.sqrt(sigma), size=10000) + np.log(s.beta.rvs(a, b, size=10000))).var()
            # experiment 1
            #V = (np.exp(s.norm.rvs(mu + np.log(s.beta.rvs(a, b, size=10000)), np.sqrt(sigma), size=10000))).var()
            #lnp
            #V = (np.array(s.poisson.rvs(np.exp(s.norm.rvs(mu, sigma, size=10000)) * s.beta.rvs(a, b, size=10000)))).var()
            
            # using joint
            norm_samples = np.random.multivariate_normal(mean=(lnp_res[0], lnp_res[1]), cov = np.linalg.inv(-lnp_res[2]), size = 10000)
            V = np.array(s.poisson.rvs(np.exp(s.norm.rvs(norm_samples[:,0], np.exp(norm_samples[:,1]))) * s.beta.rvs(a,b, size = 10000))).var()
            #V = np.array(s.poisson.rvs(np.exp(norm_samples[:,0]) * s.beta.rvs(a,b, size = 10000))).var()
            #V = np.array(np.exp(s.norm.rvs(lnp_res[0] + np.log(s.beta.rvs(a,b, size = 10000)), norm_samples[:,0].std()))).var()

            self.segment_V_list[ID] = V
            self.segment_r_list[ID] = r
            self.segment_counts[ID] = group_len
            self.segment_sums[ID] = r.sum()
            self.segment_ssd[ID] = ((r - r.mean())**2).sum()

        # go back through segments and greylist ones with high variance
        #greylist_mask = np.ones(self.num_segments, dtype=bool)
        #greylist_mask[self.greylist_segments] = False
        #cutoff = np.median(self.segment_V_list[greylist_mask]) * 20
        #self.alpha_0 = self.segment_counts[greylist_mask].mean()
        #self.beta_0 = self.alpha_0 / 2 * self.segment_V_list[greylist_mask].mean()
        self.alpha_0 = 1e-4
        self.beta_0 = 1e-4

        self.loggamma_alpha_0 = ss.loggamma(self.alpha_0)
        self.log_beta_0 = np.log(self.beta_0)

        #print(cutoff)
        #for i in set(range(self.num_segments)) - self.greylist_segments:
        #    if self.segment_V_list[i] > cutoff:
        #        self.greylist_segments.add(i)
        
        # new method for greylisting segments checks if mean varaince relation
        # holds for segments
        means = np.array([self.segment_r_list[s].mean() for s in range(self.num_segments)])
        slope, intercept = s.linregress(means, np.sqrt(self.segment_V_list))[:2]
        median_mean, median_std = np.median(means), np.median(np.sqrt(self.segment_V_list))
        boundary_intercept = (2.5 * median_std - (median_mean * slope + intercept)) + intercept
        self.greylist_segments.update(np.r_[:self.num_segments][(means * slope + boundary_intercept < np.sqrt(self.segment_V_list))])
     
        #self.seg_count_norm = self.segment_counts.mean() / 25

    # fit comb to allelic imbalances followed by allelic coverage levels to
    # classify segments as confidently clonal/subclonal. Run acdp seperately
    # on each group and merge for final results
    def run_seperated(self, n_iter):
        # allelic imbalance comb fitting
        ## subset to single draw and major allele 
        single_allele_df = self.cov_df.loc[(self.cov_df.dp_draw == self.default_draw) & (self.cov_df.allele == -1)]
        ## aggregate het counts across allelic DP segments
        snp_segments_df = single_allele_df.groupby('segment_ID').apply(lambda x:
                                          pd.Series({'weight': len(x),
                                                     'min_count':x['min_count'].sum(),
                                                     'maj_count':x['maj_count'].sum(),
                                                     'f':x['min_count'].sum() / (x['min_count'].sum() + x['maj_count'].sum())}))
        ## instantiate beta distribution at each segment imbalance level
        beta_dict = {c: s.beta(r.min_count, r.maj_count) for c, r in snp_segments_df.iterrows()}

        ## compute allelic levels given purity, copy number and ploidy
        def _imb(alpha, na, tau):
            return (1 + alpha*(na - 1))/(2*(1 - alpha) + tau*alpha)
        
        ## we deploy a grid search over alpha (purity) values
        alpha = np.linspace(0., 1., 10000)
        ## generate clonal imbalance levels. By default do this up to ploidy 7
        clonal_imbs = {}
        for tau in range(1, 8):
            for na in range(1, tau + 1):
                if na < tau/2:
                    continue
                clonal_imbs[f"{na}/{tau}"] = _imb(alpha, na, tau)
        clonal_arr = np.array(list(clonal_imbs.values()))
        
        purity_res = np.array([beta.logpdf(clonal_arr) for beta in beta_dict.values()]).transpose((2,1,0))
        
        opt_purity_idx = purity_res.max(1).sum(1).argmax()
        opt_purity = alpha[opt_purity_idx]
        
        ## check to see if there are multiple potential purities; warn if so
        opt_states = purity_res.max(1).sum(1)
        purity_liks = np.exp(opt_states[find_peaks(opt_states, prominence=10, distance = 10)[0]] - opt_states.max())
        if (purity_liks > 0.01).sum() > 1:
            print(f"WARNING: multiple purities possible, using optimal purity {opt_purity}", flush=True)
        else:
            print(f"Optimal purity is {opt_purity}", flush = True)
        
        ## now take the segments most likely to be clonal by their beta likelihoods 
        ## to avoid setting a hard theshold we take the elbow point as threshold
        opt_max = purity_res[opt_purity_idx].max(0)
        total_weight = snp_segments_df.weight.sum()
        sorted_liks = np.array(sorted(zip(opt_max, snp_segments_df.weight)))
        sorted_cum_weights = np.cumsum(sorted_liks[:,1]) / total_weight
        kneedle = KneeLocator(sorted_cum_weights, sorted_liks[:,0], S=2.0, online=True)
        clonal_lik_threshold = sorted_liks[np.where(sorted_cum_weights == kneedle.elbow)[0][0]][0]

        clonal_segIDs = np.array(list(beta_dict.keys()))[purity_res.max(1)[opt_purity_idx] > clonal_lik_threshold]
        clonal_segs = self.cov_df.loc[self.cov_df.segment_ID.isin(clonal_segIDs)].acdp_segID.unique()
        
        ## run acdp clustering on segs that pass the threshold
        print("Clustering first pass clonal segments...", flush=True)
        acdp_clonal_first = self.run(1, segs_to_use = clonal_segs)
       
        # fit allelic coverage comb
 
        ## use this first pass clustering to fit constant for allelic coverage comb
        clonal_datapoints = [np.hstack([acdp_clonal_first.segment_r_list[s] for s in acdp_clonal_first.cluster_dict[c]]) for c in acdp_clonal_first.cluster_dict]
        ## grid search max is set to twice the median
        max_k = 2 * int(np.median(np.hstack(clonal_datapoints)))
        ks = np.r_[0:max_k:0.5]
        # precompute the mus and sigmas
        consts = np.array([ks * ((1-opt_purity) + a * opt_purity) for a in range(0,6)])
        sigmas = np.array([seg_datapoints.std() for seg_datapoints in clonal_datapoints])
        print("fitting allelic coverage comb...")
        # compute gaussian pdfs for each clonal cluster, at allelic copy 1 thorough 6 and all scale factors
        covcomb_res = np.array([[[(-np.log(sigmas[i]) - np.log(np.sqrt(2* np.pi)) - ((clonal_datapoints[i] - consts[ac][k]**2) / (2 * sigmas[i]**2))).sum(0) for k in range(len(ks))] for ac in range (0,6)] for i in range(len(sigmas))])
        
        opt_k = ks[covcomb_res.max(1).sum(0).argmax()]
        opt_comb = covcomb_res.max(1).sum(0)
        comb_liks = np.exp(opt_comb[find_peaks(opt_comb, prominence=10, distance = 10)[0]] - opt_comb.max())
        if (comb_liks > 0.01).sum() > 1:
            print(f"WARNING: multiple k values possible, using optimal k= {opt_k}", flush=True)
        else:
            print(f"Optimal k value is {opt_k}", flush=True)

        # filter out non-clonal clusters that may have clonal f due to degeneracy
        ## we also filter out small clusters (less than 1 percent of genomic mass)
        opt_covcomb = covcomb_res[:,:, covcomb_res.max(1).sum(0).argmax()] # na liks for optimal k
        opt_nA = opt_covcomb.argmax(1)

        max_clonal_datapoint = np.hstack(clonal_datapoints).max()
        clonal_sizes = np.array([len(s) for s in clonal_datapoints])
        clonal_fraction = clonal_sizes / clonal_sizes.sum()

        clonal_clusters = []
        for i, c in enumerate(acdp_clonal_first.cluster_dict):
            loglik = opt_covcomb[i, opt_nA[i]]
            null_lik = np.log(1/(clonal_datapoints[i].max() - clonal_datapoints[i].min())) * clonal_sizes[i]
            if loglik > null_lik and clonal_fraction[i] > 0.01:
                clonal_clusters.append(c)
        ## this gives us our set of likely clonal clusters with cn < 6
        
        # re-assign segments based on fitted comb
        ## we now see if any segments originally binned as subclonal wish to join a clonal cluster
        ## compute target distributions for the clonal clusters
        target_clonal_datapoints = [(c, np.hstack([self.segment_r_list[s] for s in acdp_clonal_first.cluster_dict[c]])) for c in clonal_clusters]
        target_normals = {tc_data[0] : s.norm(tc_data[1].mean(), tc_data[1].std()) for tc_data in target_clonal_datapoints}
        ## find the clonal comb levels to set null uniform dist limits
        ## for every segment that was originally classified as subclonal (by allelic imbalance) we compare to each clonal distribution and the null dist
        subclonal_single_segs = np.array(list(beta_dict.keys()))[purity_res.max(1)[opt_purity_idx] <= clonal_lik_threshold]
        subclonal_segs = self.cov_df.loc[self.cov_df.segment_ID.isin(subclonal_single_segs)].acdp_segID.unique()
        subclonal_segs_data = [(s, self.segment_r_list[s]) for s in subclonal_segs]

        clonal_classified = []
        subclonal_classified = []
        for seg, seg_datapoints in subclonal_segs_data:
            up, low = seg_datapoints.max(), seg_datapoints.min()
            seg_res = {c:norm.logpdf(seg_datapoints).sum() for c, norm in target_normals.items()}
            seg_res[-1] = np.log(1/ (up-low)) * len(seg_datapoints)
            assgn = list(seg_res.keys())[np.array(list(seg_res.values())).argmax()]
            if assgn == -1:
                subclonal_classified.append(seg)
            else:
                clonal_classified.append(seg)
       
        # run acdp clustering on each set of segments
        clonal_segs_final = np.r_[np.hstack([list(acdp_clonal_first.cluster_dict[c]) for c in clonal_clusters]), clonal_classified].astype(int)
        subclonal_segs_final = np.array(list(set(range(self.num_segments)) - set(clonal_segs_final))).astype(int)
        print("clustering final clonal segments...", flush=True)        
        acdp_clonal = self.run(1, segs_to_use = clonal_segs_final)
        print("clustering final subclonal segments...", flush=True)        
        acdp_subclonal = self.run(1, segs_to_use = subclonal_segs_final)
        
        # merge acdp results from the two classes into a final acdp result
        acdp_combined = AllelicCoverage_DP(self)
        ## create set of unique cluster labels
        unique_cluster_labels = {c:v for c,v in zip(*np.unique(np.r_[list(acdp_clonal.cluster_dict.keys()), list(acdp_subclonal.cluster_dict.keys())], return_index=True))}
        combined_cluster_dict = {**{unique_cluster_labels[c]:v for c,v in acdp_clonal.cluster_dict.items()}, **{unique_cluster_labels[c]:v for c,v in acdp_subclonal.cluster_dict.items()}}
        print("Merging cluster outputs...", flush=True)  
        ## update fields with combined data
        acdp_combined.cluster_dict = sc.SortedDict(combined_cluster_dict)
        acdp_combined.next_cluster_index = len(acdp_combined.cluster_dict)
        acdp_combined.cluster_counts = sc.SortedDict({c:acdp_combined.segment_counts[list(acdp_combined.cluster_dict[c])].sum() for c in acdp_combined.cluster_dict})
        acdp_combined.cluster_sums = sc.SortedDict({c: v.sum() for c, v in [(c, np.hstack([acdp_combined.segment_r_list[s] for s in acdp_combined.cluster_dict[c]])) for c in acdp_combined.cluster_dict]})
        acdp_combined.cluster_ssd = sc.SortedDict({c: v.var() * len(v) for c, v in [(c, np.hstack([acdp_combined.segment_r_list[s] for s in acdp_combined.cluster_dict[c]])) for c in acdp_combined.cluster_dict]})
        for c in acdp_combined.cluster_dict:
            acdp_combined.cluster_assignments[list(acdp_combined.cluster_dict[c])] = c
        acdp_combined.cluster_MLs = sc.SortedDict({c:acdp_combined._ML_cluster_from_list(acdp_combined.cluster_dict[c]) for c in acdp_combined.cluster_dict})    
        
        acdp_combined.prepare_df()
            
        return acdp_combined
         
    def run(self, n_iter, segs_to_use=None, return_res=False):
        acdp = AllelicCoverage_DP(self, segs_to_use)
        acdp_res = acdp.run(n_iter)
        self.acdp = acdp
        if return_res:
            return acdp, acdp_res
        else:
            return acdp

class AllelicCoverage_DP:
    def __init__(self, runner_obj, segs_to_use=None, assign_greylist=False):
        # load data from top-level runner
        self.cov_df = runner_obj.cov_df
        self.cytoband_file = runner_obj.cytoband_file
        self.seed_all_clusters = runner_obj.seed_all_clusters
        self.wgs = runner_obj.wgs
        self.draw_idx = runner_obj.draw_idx
        self.default_draw = runner_obj.default_draw
        self.num_segments = runner_obj.num_segments
        self.segment_r_list = runner_obj.segment_r_list
        self.segment_V_list = runner_obj.segment_V_list
        self.segment_counts = runner_obj.segment_counts
        self.segment_allele = runner_obj.segment_allele
        self.cluster_assignments = runner_obj.cluster_assignments
        self.segment_sums = runner_obj.segment_sums
        self.segment_ssd = runner_obj.segment_ssd
        self.greylist_segments = runner_obj.greylist_segments
        self.beta = runner_obj.beta
        self.to_assign_greylist = assign_greylist
        
        if segs_to_use is None:
            segs_to_use = sc.SortedSet(range(self.num_segments))
        self.segments = sc.SortedSet(segs_to_use)

        self.cluster_counts = sc.SortedDict({})
        self.unassigned_segs = sc.SortedSet(self.segments)
        self.cluster_dict = sc.SortedDict({})
        self.cluster_MLs = sc.SortedDict({})
        self.cluster_sums = sc.SortedDict({})
        self.cluster_ssd = sc.SortedDict({}) # sum of squared deviations
        
        self.ML_total_history = []
        self.DP_total_history = []
        self.MLDP_total_history = []

        # inverse gamma hyper parameter default values -- will be set later based on tuples
        self.alpha_0 = 100
        self.beta_0 = 30
        self.kappa_0 = 1e-6
        self.loggamma_alpha_0 =0
        self.log_beta_0 = 0
        self.half_log2pi = np.log(2*np.pi) / 2
        self.seg_count_norm = 5. 

        self._init_clusters(segs_to_use)

        # containers for saving the MCMC trace_cov_dp
        self.clusters_to_segs = []
        self.bins_to_clusters = []
        self.draw_indices = []

        self.alpha = 0.5
        
    def _init_clusters(self, segs_to_use):
        [self.unassigned_segs.discard(s) for s in self.greylist_segments]
        if not self.seed_all_clusters:
            first = (self.segments - self.greylist_segments)[0]
            clusterID = 0
            self.cluster_counts[0] = self.segment_counts[0]
            self.unassigned_segs.discard(0)
            self.cluster_dict[0] = sc.SortedSet([first])
            self.cluster_MLs[0] = self._ML_cluster_from_list([first])
            self.cluster_assignments[first] = 0
            self.cluster_sums[0] = self.segment_sums[first]
            self.cluster_ssd[0] = self.segment_ssd[first]
            #next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = 1
        else:
            for i in self.segments - self.greylist_segments:
                self.cluster_counts[i] = self.segment_counts[i]
                self.unassigned_segs.discard(i)
                self.cluster_dict[i] = sc.SortedSet([i])
                self.cluster_MLs[i] = self._ML_cluster_from_list([i])
                self.cluster_assignments[i] = i
                self.cluster_sums[i] = self.segment_sums[i]
                self.cluster_ssd[i] = self.segment_ssd[i]
            #next cluster index is the next unused cluster index (i.e. not used by prior cluster or current)
            self.next_cluster_index = i+1

        # datapoint array generation methods

        # generate cluster from list of segment IDs
    def _cluster_gen_from_list(self, cluster_list):
        r_lst = []
        for s in cluster_list:
            r_lst.append(self.segment_r_list[s])
        r = np.hstack(r_lst)
        return r

    def _cluster_gen_add_one(self, clusterID, segID):
        return np.concatenate([self.cluster_datapoints[clusterID], self.segment_r_list[segID]], axis=0)

    # assumes the datapoints are ordered by segment ID
    def _cluster_gen_remove_one(self, clusterID, segID):
        cur = self.cluster_datapoints[clusterID]
        segs = self.cluster_dict[clusterID]
        seg_ind = segs.index(segID)
        st = self.segment_counts[segs][:seg_ind].sum()        
        en = st + self.segment_counts[segID]
        return np.concatenate([cur[:st], cur[en:]], axis=0)

    def _cluster_gen_merge(self, clust_A, clust_B):
        return np.concatenate([self.cluster_datapoints[clust_A], self.cluster_datapoints[clust_B]], axis=0)

    def _ML_cluster_direct(self, n, r_mean, ssd):
        return self.ML_normalgamma(n, r_mean, ssd)

    def _ML_cluster_from_list(self, cluster_list):
        r = self._cluster_gen_from_list(cluster_list)
        n = len(r)
        ssd = n * r.var()
        return self.ML_normalgamma(n, r.mean(), ssd)

    def _ML_cluster_add_one(self, clusterID, segID):
        mn, mu_mn, ssd = self._ssd_cluster_add_one(clusterID, segID)
        
        return self.ML_normalgamma(mn, mu_mn, ssd)

    def _ML_cluster_remove_one(self, clusterID, segID):
        m, mu_m, ssd = self._ssd_cluster_remove_one(clusterID, segID) 
        
        return self.ML_normalgamma(m, mu_m, ssd)

    def _ML_cluster_merge(self, clust_A, clust_B):
        mn, mu_mn, ssd = self._ssd_cluster_merge(clust_A, clust_B)
        return self.ML_normalgamma(mn, mu_mn, ssd)
    
    # worker function for normal-gamma distribution log Marginal Likelihood
    def ML_normalgamma(self, n, x_mean, ssd):
        # for now x_mean is the same as mu0
        mu0 = x_mean

        mu_n = (self.kappa_0*mu0 + n * x_mean) / (self.kappa_0 + n)
        kappa_n = self.kappa_0 + n
        alpha_n = self.alpha_0 + n/2
        beta_n = self.beta_0 + 0.5 * ssd + self.kappa_0 * n * (x_mean - mu0)**2 / 2*(self.kappa_0 + n)

        return ss.loggamma(alpha_n) - self.loggamma_alpha_0 + self.alpha_0 * self.log_beta_0 - alpha_n * np.log(beta_n) + np.log(self.kappa_0 / kappa_n) / 2 - n * self.half_log2pi


# utility methods for ssd and mean calculations
    def _ssd_cluster_add_one(self, clusterID, segID):
        n = self.cluster_counts[clusterID]
        m = self.segment_counts[segID]
        mn = m + n
        
        sum_n = self.cluster_sums[clusterID]
        sum_m = self.segment_sums[segID]
        mu_mn = (sum_n + sum_m) / (mn)
        
        mu_n = sum_n / n
        mu_m = sum_m / m
        ssd = self.cluster_ssd[clusterID] + self.segment_ssd[segID] + m * n / (m + n) * (mu_m - mu_n)**2
        return mn, mu_mn, ssd
    
    def _ssd_cluster_merge(self, clust_A, clust_B):
        
        n = self.cluster_counts[clust_A]
        m = self.cluster_counts[clust_B]
        mn = m + n
        sum_n = self.cluster_sums[clust_A]
        sum_m = self.cluster_sums[clust_B]
        mu_mn = (sum_n + sum_m) / (mn)
        
        mu_n = sum_n / n
        mu_m = sum_m / m
        
        ssd = self.cluster_ssd[clust_A] + self.cluster_ssd[clust_B] + m * n / (m + n) * (mu_m - mu_n)**2
        return mn, mu_mn, ssd

    def _ssd_cluster_remove_one(self, clusterID, segID):
        mn = self.cluster_counts[clusterID]
        mn = self.cluster_counts[clusterID]
        n = self.segment_counts[segID]
        m = mn - n

        sum_mn = self.cluster_sums[clusterID]
        sum_n = self.segment_sums[segID]

        mu_m = (sum_mn - sum_n) / m
        mu_n = self.segment_sums[segID] / n 
        
        #S_m = S_mn - S_n - mn/(m+n) / (mu_m - mu_n)^2
        ssd = self.cluster_ssd[clusterID] - self.segment_ssd[segID] - (m * n) / (m + n) * (mu_m - mu_n)**2
        return m, mu_m, ssd

    def save_ML_total(self):
        ML_tot = np.r_[self.cluster_MLs.values()].sum()
        self.ML_total_history.append(ML_tot)

        num_clusts = len(self.cluster_dict)
        N = sum(self.cluster_counts.values())
        DP_tot = num_clusts * np.log(self.alpha) + sum(
            [ss.gammaln(na) for na in self.cluster_counts.values()]) - ss.gammaln(self.alpha + N) + ss.gammaln(
            self.alpha)
        self.DP_total_history.append(DP_tot)
        self.MLDP_total_history.append(ML_tot + DP_tot)

    # functions for computing the DP priors for each cluster action

    def DP_merge_prior(self, cur_cluster):
        cur_index = self.cluster_counts.index(cur_cluster)
        cluster_vals = np.array(self.cluster_counts.values())
        ##normalize cluster counts
        cluster_vals = cluster_vals / self.seg_count_norm
        N = cluster_vals.sum()
        M = cluster_vals[cur_index]
        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            if i != cur_index:
                prior_results[i] = ss.loggamma(M + nc) + ss.loggamma(N + self.alpha - M) - (
                            ss.loggamma(nc) + ss.loggamma(N + self.alpha))
            else:
                # the prior prob of remaining in the current cluster is the same as for joining a new cluster
                prior_results[i] = ss.gammaln(M) + np.log(self.alpha) + ss.gammaln(N + self.alpha - M) - ss.gammaln(
                    N + self.alpha)
        return prior_results

    def DP_tuple_split_prior(self, seg_id):
        cur_cluster = self.cluster_assignments[seg_id]
        seg_size = self.segment_counts[seg_id]
        cluster_vals = np.array(self.cluster_counts.values())

        if cur_cluster > -1:
            # exclude the points were considering moving from the dp calculation
            # if the tuple was already in a cluster
            cur_index = self.cluster_counts.index(cur_cluster)
            cluster_vals[cur_index] -= seg_size
        
        ##normalize cluster counts
        cluster_vals = cluster_vals / self.seg_count_norm
        seg_size = seg_size / self.seg_count_norm
 
        N = cluster_vals.sum()

        loggamma_N_alpha = ss.loggamma(N + self.alpha)
        loggamma_N_alpha_seg = ss.loggamma(N + self.alpha - seg_size)
        prior_results = np.zeros(len(cluster_vals))
        for i, nc in enumerate(cluster_vals):
            prior_results[i] = ss.loggamma(seg_size + nc) + loggamma_N_alpha_seg - (ss.loggamma(nc) + loggamma_N_alpha)

        # the prior prob of starting a new cluster
        prior_new = ss.gammaln(seg_size) + np.log(self.alpha) + loggamma_N_alpha_seg - loggamma_N_alpha
        return np.r_[prior_results, prior_new]

    # since were taking the ratio we can remove the final two terms:
    # ss.gammaln(self.alpha + N - M) - ss.gammaln(self.alpha + N)
    # from both split and stay
    def DP_split_prior(self, split_A_segs, split_B_segs):
        n_a = self.segment_counts[split_A_segs].sum()
        n_b = self.segment_counts[split_B_segs].sum()
        ##normalize segment_counts
        n_a = n_a / self.seg_count_norm
        n_b = n_b / self.seg_count_norm
        M = n_a + n_b
        split = 2 * np.log(self.alpha) + ss.gammaln(n_a) + ss.gammaln(n_b)
        stay = np.log(self.alpha) + ss.gammaln(M)
        return split - stay

    # old method for assigning greylisted segments after the clustering is complete
    # ML greylist assignemnts were prone to assigning high variance segments 
    # to the highest variance cluster. moved to likelihood
    def assign_greylist_MLs(self):
        # keep a copy of this since it will remain static
        ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])
        # make a deep copy of the cluster dict since we dont want assignment of greylisted clusters to affect subsequent assignments
        greylist_added_dict = sc.SortedDict({k: v.copy() for k, v in self.cluster_dict.items()})

        for segID in self.greylist_segments:

            # compute ML of every cluster if S joins
            ML_BC = np.array([self._ML_cluster_add_one(k, segID) for k in self.cluster_counts.keys()])

            # likelihood ratios of S joining each other cluster S -> Ck
            ML_rat = ML_BC - ML_C

            # currently we do not support prior draws here
            # construct transition probability distribution and draw from it
            log_count_prior = self.DP_tuple_split_prior(segID)[:-1]
            MLs_max = (ML_rat + log_count_prior).max()
            choice_p = np.exp(ML_rat + log_count_prior - MLs_max) / np.exp(
                ML_rat + log_count_prior - MLs_max).sum()
            choice_idx = np.random.choice(
                np.r_[0:len(ML_rat)],
                p=choice_p
            )
            choice = np.r_[self.cluster_dict.keys()][choice_idx]
            choice = int(choice)
            greylist_added_dict[choice].add(segID)

        # now set cluster dict to the one with the greylisted items assigned
        self.cluster_dict = greylist_added_dict

    # assignes greylisted segments to clusters or discards based on gaussian
    # likelyhoods
    def assign_greylist(self):
        clusters = np.array(list(self.cluster_dict.keys()))
        norms = {c: s.norm(seg.mean(), seg.std()) for c, seg in [(c, np.hstack([self.segment_r_list[i] for i in self.cluster_dict[c]])) for c in clusters]}
        for segID in self.greylist_segments:
            r = self.segment_r_list[segID]
    
            log_liks = np.r_[[norm.logpdf(r).sum() for c, norm in norms.items()]]
            null_lik = np.log(1 /(r.max() - r.min())) * len(r)
            clusterID = clusters[log_liks.argmax()]
            if log_liks.max() > null_lik:
                print(f'segment {segID} assigned to cluster {clusterID}')
                # we found a good match for this segment and we can add it to the cluster
                self.cluster_dict[clusterID].add(segID)
                self.cluster_assignments[segID] = clusterID
                self.cluster_counts[clusterID] += len(r)
                self.cluster_sums[clusterID] += r.sum()
                self.cluster_MLs[clusterID] = self._ML_cluster_add_one(clusterID, segID)
                self.cluster_ssd[clusterID] = self._ssd_cluster_add_one(clusterID, segID)
                continue
            else:
                if len(r) > 25:
                    print(f'segment {segID} given new cluster {self.next_cluster_index}')
                    # if we have a farily large segment we let it form its own cluster
                    self.cluster_dict[self.next_cluster_index] = sc.SortedSet([segID])
                    self.cluster_assignments[segID] = self.next_cluster_index
                    self.cluster_counts[self.next_cluster_index] = len(r)
                    self.cluster_sums[self.next_cluster_index] = r.sum()
                    self.cluster_ssd[self.next_cluster_index] = len(r) * r.var()
                    self.cluster_MLs[self.next_cluster_index] = self._ML_cluster_direct(len(r), r.mean(), len(r) * r.var())
                    self.next_cluster_index += 1
                else:
                    # discard the segment by not assigning it a cluster
                    print(f'segment {segID} discarded')
                    continue

    # save current state as a draw in mcmc run
    def take_draw(self, n_it):
        self.bins_to_clusters.append(self.cluster_assignments.copy())
        self.clusters_to_segs.append(self.cluster_dict.copy())
        self.draw_indices.append(n_it)

    def run(self, n_iter):

        burned_in = False
        all_assigned = False

        n_it = 0
        n_it_last = 0

        white_segments = self.segments - self.greylist_segments

        while len(self.bins_to_clusters) < n_iter:

            self.save_ML_total()
            
            # check if only one segment, if so skip
            if len(white_segments) < 2:
                self.take_draw(n_it)
                n_it_last = n_it
                n_it += 1
                continue
            
            
            # status update
            if not n_it % 250:
                print("{} iterations; num_clusters: {}; ML:{}".format(n_it, len(self.cluster_dict.keys()), self.MLDP_total_history[-1]), flush=True)

            # start couting for burn in
            if not n_it % 100:
                # self.cdict_history.append(self.cluster_dict.copy())
                if not all_assigned and (self.cluster_assignments[white_segments] > -1).all():
                    all_assigned = True
                    n_it_last = n_it
                # burn in after n_seg / n_clust iteration
                if not burned_in and all_assigned and n_it - n_it_last > max(2000, len(self.segments)):
                    if np.diff(np.r_[self.MLDP_total_history[-2000:]]).mean() <= 0:
                        print('burnin', flush=True)
                        burned_in = True
                        n_it_last = n_it
            
            # pick either a segment or a cluster at random
            # pick segment
            if np.random.rand() < 0.5:
                if len(self.unassigned_segs) > 0 and len(
                        self.unassigned_segs) / len(self.segments) < 0.1 and np.random.rand() < 0.5:
                    segID = np.random.choice(self.unassigned_segs)
                else:
                    segID = np.random.choice(white_segments)
                # get cluster assignment of S
                clustID = self.cluster_assignments[segID]
                # compute ML of AB = Cs (cached)
                if clustID == -1:
                    ML_AB = 0
                else:
                    ML_AB = self.cluster_MLs[clustID]

                # compute ML of A = Cs - S
                if clustID == -1:
                    ML_A = 0
                # if cluster is empty without S ML is also 0
                elif len(self.cluster_dict[clustID]) == 1:
                    ML_A = 0
                else:
                    ML_A = self._ML_cluster_remove_one(clustID, segID)
                # compute ML of S on its own
                ML_S = self._ML_cluster_from_list([segID])

                # compute ML of every other cluster C = Ck, k != s (cached)
                # for now were also allowing it to chose to stay in current cluster

                ML_C = np.array([ML for (ID, ML) in self.cluster_MLs.items()])

                # compute ML of every cluster if S joins
                ML_BC = np.array([self._ML_cluster_add_one(k, segID) if k!= clustID else ML_AB for k in self.cluster_counts.keys()])

                # likelihood ratios of S joining each other cluster S -> Ck
                ML_rat_BC = ML_A + ML_BC - (ML_AB + ML_C)

                # if cluster is unassigned we set the ML ratio to 1 for staying in its own cluster
                if clustID > -1:
                    ML_rat_BC[list(self.cluster_counts.keys()).index(clustID)] = 0

                # compute ML of S starting a new cluster
                ML_new = ML_A + ML_S - ML_AB

                ML_rat = np.r_[ML_rat_BC, ML_new]

                # currently we do not support prior draws here
                # construct transition probability distribution and draw from it
                log_count_prior = self.DP_tuple_split_prior(segID)
                MLs_max = (ML_rat + log_count_prior).max()
                choice_p = np.exp(ML_rat + log_count_prior - MLs_max) / np.exp(
                    ML_rat + log_count_prior - MLs_max).sum()

                if np.isnan(choice_p.sum()):
                    print('skipping iteration {} due to nan. picked segment {}'.format(n_it, segID), flush=True)
                    #print(np.where(np.isnan(log_count_prior)))
                    n_it += 1
                    continue
                choice_idx = np.random.choice(
                    np.r_[0:len(ML_rat)],
                    p=choice_p
                )

                # last = brand new, -1, -2, -3, ... = -(prior clust index) - 1
                choice = np.r_[self.cluster_counts.keys(), self.next_cluster_index][choice_idx]
                choice = int(choice)
                # if choice == next_cluster_idx then start a brand new cluster with this new index
                # if choice < 0 then we create a new cluster with the index of the old cluster
                if choice == self.next_cluster_index:
                    if clustID > -1:
                        # if the segment used to occupy a cluster by itself, do nothing if its a new cluster
                        if len(self.cluster_dict[clustID]) == 1:
                            n_it += 1
                            continue
                        else:
                            # otherwise seg was previously assigned so remove it from previous cluster
                            self.cluster_ssd[clustID] = self._ssd_cluster_remove_one(clustID, segID)[2]
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
                            self.cluster_sums[clustID] -= self.segment_sums[segID]
                    else:
                        # if it wasn't previously assigned we need to remove it from the unassigned list
                        self.unassigned_segs.discard(segID)

                    # create new cluster with next available index and add segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_counts[choice] = self.segment_counts[segID]
                    self.cluster_dict[choice] = sc.SortedSet([segID])
                    self.cluster_MLs[choice] = ML_S
                    self.cluster_sums[choice] = self.segment_sums[segID]
                    self.cluster_ssd[choice] = self.segment_ssd[segID]
                    self.next_cluster_index += 1
                else:
                    # if remaining in same cluster check if we should take a draw then continue
                    if clustID == choice:
                        if burned_in and n_it - n_it_last > len(self.segments):
                            self.take_draw(n_it)
                            n_it_last = n_it
                        n_it += 1
                        continue

                    # joining existing cluster

                    # update new cluster with additional segment
                    self.cluster_assignments[segID] = choice
                    self.cluster_ssd[choice] = self._ssd_cluster_add_one(choice, segID)[2]
                    self.cluster_counts[choice] += self.segment_counts[segID]
                    self.cluster_dict[choice].add(segID)
                    self.cluster_MLs[choice] = ML_BC[list(self.cluster_counts.keys()).index(choice)]
                    self.cluster_sums[choice] += self.segment_sums[segID]
                    # if seg was previously assigned we need to update its previous cluster
                    if clustID > -1:
                        # if segment was previously alone in cluster, that cluster will be destroyed
                        if len(self.cluster_dict[clustID]) == 1:
                            del self.cluster_counts[clustID]
                            del self.cluster_dict[clustID]
                            del self.cluster_MLs[clustID]
                            del self.cluster_sums[clustID]
                            del self.cluster_ssd[clustID]
                        else:
                            # otherwise update former cluster
                            self.cluster_ssd[clustID] = self._ssd_cluster_remove_one(clustID, segID)[2]
                            self.cluster_counts[clustID] -= self.segment_counts[segID]
                            self.cluster_dict[clustID].discard(segID)
                            self.cluster_MLs[clustID] = ML_A
                            self.cluster_sums[clustID] -= self.segment_sums[segID]
                    else:
                        self.unassigned_segs.discard(segID)

            # pick cluster to merge or split
            else:
                # it only makes sense to try joining two clusters if there are at least two of them!
                if len(self.cluster_counts) < 2:
                    n_it += 1
                    continue

                clust_pick = np.random.choice(self.cluster_dict.keys())
                clust_pick_segs = np.r_[self.cluster_dict[clust_pick]].astype(int)

                # half the time we'll propose splitting this cluster
                if np.random.rand() < 0.5:
                    # if theres only one tuple then we cant split
                    if len(clust_pick_segs) == 1:
                        n_it += 1
                        continue

                    # find the best place to split these tuples based on their datapoint means
                    seg_means = self.segment_sums[clust_pick_segs] / self.segment_counts[clust_pick_segs]
                    sort_indices = np.argsort(seg_means)
                    sorted_vals = seg_means[sort_indices]

                    tot_list = []
                    stay_ml = self.cluster_MLs[clust_pick]

                    sorted_segs = clust_pick_segs[sort_indices]
                    sorted_lens = self.segment_counts[sorted_segs]
                    n_A = sorted_lens[0]
                    n_B = sorted_lens[1:].sum()
                    
                    sorted_sums = self.segment_sums[sorted_segs]
                    sum_A = sorted_sums[0]
                    sum_B = sorted_sums[1:].sum()
                    
                    sorted_ssds = self.segment_ssd[sorted_segs]
                    ssd_A = sorted_ssds[0]
                    ssd_B = self._ssd_cluster_remove_one(clust_pick, sorted_segs[0])[2]
                    
                    cached_ssds = []
                    search_inds = np.r_[1:len(sorted_vals)]
                    for i in search_inds:
                        A_list = sorted_segs[:i]
                        B_list = sorted_segs[i:]

                        mu_A = sum_A / n_A
                        mu_B = sum_B/ n_B
                        
                        ML_A =  self._ML_cluster_direct(n_A, mu_A, ssd_A)
                        ML_B = self._ML_cluster_direct(n_B, mu_B, ssd_B)
                        ML_rat = ML_A + ML_B - stay_ml
                        
                        dp_prior_rat = self.DP_split_prior(A_list, B_list)
                        ML_tot = ML_rat + dp_prior_rat
                        tot_list.append(ML_tot)
                        cached_ssds.append((ssd_A, ssd_B))
                        
                        #update running statistics if there are more to compute
                        if i < len(sorted_vals) - 1 :
                            # S is the tuple we're moving from B to A
                            len_S = sorted_lens[i]
                            sum_S = sorted_sums[i]
                            mu_S = sum_S / len_S
                            
                            ssd_S = self.segment_ssd[sorted_segs[i]]
                            ssd_A = ssd_A + ssd_S + (n_A * len_S) / (n_A + len_S) * (mu_A - mu_S)**2
                            n_newB = n_B - len_S
                            mu_newB = (sum_B - sum_S)/ n_newB
                
                            ssd_B = ssd_B - ssd_S - (n_newB * len_S) / (n_newB + len_S) * (mu_newB - mu_S)**2
                            n_A += len_S
                            n_B -= len_S
                            sum_A += sum_S
                            sum_B -= sum_S
                        
                        
                    tot_list = np.array(tot_list)
                    tot_max = tot_list.max()
                    choice_p = np.exp(tot_list - tot_max) / np.exp(tot_list - tot_max).sum()
                    split_ind = np.random.choice(len(tot_list), p=choice_p)

                    A_list = sorted_segs[:split_ind + 1]
                    B_list = sorted_segs[split_ind + 1:]
                    ssd_A, ssd_B = cached_ssds[split_ind]
                    # add ML ratios to get the likelihood of splitting
                    ML_tot = tot_list[split_ind]

                    # we split with probability equal to this likelihood
                    # to avoid overflow with large positive lls
                    if ML_tot >= 0:
                        split_prob = 1
                    else:
                        split_prob = np.exp(ML_tot)
                    if np.random.rand() < split_prob:
                        # split these clusters

                        # update cluster pick to include only segments from list A
                        self.cluster_counts[clust_pick] = sum(self.segment_counts[A_list])
                        self.cluster_dict[clust_pick] = sc.SortedSet(A_list)
                        self.cluster_MLs[clust_pick] = self._ML_cluster_from_list(A_list)
                        self.cluster_sums[clust_pick] = self.segment_sums[A_list].sum()
                        self.cluster_ssd[clust_pick] = ssd_A
                        # create new cluster with next available index and add segments from list B
                        self.cluster_assignments[B_list] = self.next_cluster_index
                        self.cluster_counts[self.next_cluster_index] = sum(self.segment_counts[B_list])
                        self.cluster_dict[self.next_cluster_index] = sc.SortedSet(B_list)
                        self.cluster_MLs[self.next_cluster_index] = self._ML_cluster_from_list(B_list)
                        self.cluster_sums[self.next_cluster_index] = self.segment_sums[B_list].sum()
                        self.cluster_ssd[self.next_cluster_index] = ssd_B
                        self.next_cluster_index += 1

                # otherwise we'll propose a merge
                else:
                    # get ML of this cluster merged with each of the other existing clusters

                    ML_join = [self._ML_cluster_merge(i, clust_pick) if i != clust_pick else
                               self.cluster_MLs[i] for i in self.cluster_dict.keys()]
                    # we need to compare this ML with the ML of leaving the target cluster and the picked cluster on their own
                    ML_split = np.array(self.cluster_MLs.values()) + self.cluster_MLs[clust_pick]
                    ML_split[self.cluster_MLs.keys().index(clust_pick)] = self.cluster_MLs[clust_pick]
                    ML_rat = np.array(ML_join) - ML_split

                    count_prior = np.r_[self.DP_merge_prior(clust_pick)]

                    # construct transition probability distribution and draw from it
                    MLs_max = (ML_rat + count_prior).max()
                    choice_p = np.exp(ML_rat + count_prior - MLs_max) / np.exp(
                        ML_rat + count_prior - MLs_max).sum()

                    if np.isnan(choice_p.sum()):
                        print("skipping iteration {} due to nan".format(n_it), flush=True)
                        #print(ML_rat)
                        #print(count_prior)
                        #print(choice_p)
                        n_it += 1
                        continue

                    choice_idx = np.random.choice(
                        np.r_[0:len(ML_rat)],
                        p=choice_p
                    )

                    choice = np.r_[self.cluster_counts.keys()][choice_idx]
                    choice = int(choice)

                    if choice != clust_pick:
                        # we need to merge clust_pick and choice_idx which we do by merging to the cluster with more
                        # segments
                        tup = (clust_pick, choice)
                        larger_cluster = np.argmax([self.cluster_counts[clust_pick], self.cluster_counts[choice]])
                        merged_ID = tup[larger_cluster]
                        vacatingID = tup[int(not larger_cluster)]

                        # move vacatingID to newclustID
                        # update new cluster with additional segments
                        vacating_segs = self.cluster_dict[vacatingID]
                        self.cluster_assignments[vacating_segs] = merged_ID
                        self.cluster_ssd[merged_ID] = self._ssd_cluster_merge(merged_ID, vacatingID)[2]
                        self.cluster_counts[merged_ID] += self.segment_counts[vacating_segs].sum()
                        self.cluster_dict[merged_ID] = self.cluster_dict[merged_ID].union(vacating_segs)
                        self.cluster_MLs[merged_ID] = ML_join[choice_idx]
                        self.cluster_sums[merged_ID] += self.cluster_sums[vacatingID]

                        # delete last cluster
                        del self.cluster_counts[vacatingID]
                        del self.cluster_dict[vacatingID]
                        del self.cluster_MLs[vacatingID]
                        del self.cluster_sums[vacatingID]
                        del self.cluster_ssd[vacatingID]
                
            if burned_in and n_it - n_it_last > len(self.segments):
                self.take_draw(n_it)
                n_it_last = n_it

            n_it += 1

        #assign the greylisted segments
        if self.to_assign_greylist:
            self.assign_greylist()
        
        ## add cluster information to the dataframe
        self.prepare_df()
        # return the clusters from the last draw and the counts
        return self.clusters_to_segs
    
    #helper functions for plotting
    def _get_tuple_intervals(self, x_ind, min_len=5):
        """Find runs of consecutive indices of length at least min_len"""
        run_intervals = []
        prev_idx = 0
        prev_val = 0
        run_len = 0
        
        for i, v in enumerate(x_ind):
            if v == prev_val + 1:
                run_len += 1
            else:
                if run_len >= min_len:
                    run_intervals += [prev_idx, i-1]
                prev_idx = i
                run_len = 1
        
            prev_val = v
        
        # check if there is a unfinished run
        if run_len >= min_len:
            run_intervals += [prev_idx, i-1]
        return np.array(run_intervals).reshape(-1,2)
    
    def _get_seg_terr(self, df):
        # find the genomic territory spanned by some coverage bins
        return (df.end_g - df.start_g).sum()
    
    def _get_clust_terr(self, seg_list, full_df):
        # find the genomic territory spanned by a acdp cluster
        return sum([self._get_seg_terr(full_df[seg][1]) for seg in seg_list])
    
    def _get_color_palette(self, num_colors):
        base_colors = np.array([
              [0.368417, 0.506779, 0.709798],
              [0.880722, 0.611041, 0.142051],
              [0.560181, 0.691569, 0.194885],
              [0.922526, 0.385626, 0.209179],
              [0.528488, 0.470624, 0.701351],
              [0.772079, 0.431554, 0.102387],
              [0.363898, 0.618501, 0.782349],
              [1, 0.75, 0],
              [0.647624, 0.37816, 0.614037],
              [0.571589, 0.586483, 0.],
              [0.915, 0.3325, 0.2125],
              [0.400822, 0.522007, 0.85],
              [0.972829, 0.621644, 0.073362],
              [0.736783, 0.358, 0.503027],
              [0.280264, 0.715, 0.429209]
            ])
        extra_colors = np.array(
              distinctipy.distinctipy.get_colors(
                num_colors - base_colors.shape[0],
                exclude_colors = [list(x) for x in np.r_[np.c_[0, 0, 0], np.c_[1, 1, 1], np.c_[0.5, 0.5, 0.5], np.c_[1, 0, 1], base_colors]],
            rng = 1234
              )
            )
        return np.r_[base_colors, extra_colors if extra_colors.size > 0 else np.empty([0, 3])]

    def _get_cluster_colors(self):
        num_clusters = len(self.cluster_dict.keys())
        
        full_df = list(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        #get argsorting in descending order (hence the negative sign)
        si = np.argsort(np.argsort([- self._get_clust_terr(self.cluster_dict[c], full_df) for c in self.cluster_dict.keys()]))
        palette = self._get_color_palette(num_clusters)
        
        return palette[si]
    
    def _get_cdp_colors(self, cdp_draw=0):
        chosen_draw = self.cov_df.loc[(self.cov_df.dp_draw == cdp_draw) & (self.cov_df.allele==1)]
        num_clusters = len(chosen_draw.cov_DP_cluster.unique())
        palette = self._get_color_palette(num_clusters)
        
        return {c : palette[i] for i, c in enumerate(chosen_draw.cov_DP_cluster.value_counts().index)}

    def _get_adp_colors(self):
        chosen_draw = self.cov_df.loc[(self.cov_df.dp_draw == self.default_draw) & (self.cov_df.allele==1)]
        num_clusters = len(chosen_draw.allelic_cluster.unique())
        palette = self._get_color_palette(num_clusters)
        
        return {c : palette[i] for i, c in enumerate(chosen_draw.allelic_cluster.value_counts().index)}
    
    # helper function for plotting coverage bins at their centers
    def _scatter_apply(self, x, _minor, _major):
        _f = np.zeros(len(x))
        _f[x.allele == -1] = _minor / (_minor + _major)
        _f[x.allele == 1] = _major / (_minor + _major)
        centers = x.start_g.values + (x.end_g.values - x.start_g.values) / 2
        return centers, _f
    
    def _get_real_cov(self, df):
        covar_cols = sorted(df.columns[df.columns.str.contains("(?:^C_.*z$|C_log_len)")])
        C = np.c_[df[covar_cols]]
        return np.exp(np.log(df.fragcorr.values) - (C @ self.beta).flatten())

    def precompute_cluster_params(self, cdp_draw=None):
        
        clust_data = {}
        if cdp_draw is None:
            full_df = list(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        else:
            full_df = list(self.cov_df.loc[self.cov_df.dp_draw==cdp_draw].groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))
        for i, c in enumerate(self.cluster_dict.keys()):
            clust_r = []
            clust_terr = 0
            for tup in self.cluster_dict[c]:
                clust_r.append(self.segment_r_list[tup])
                clust_terr += self._get_seg_terr(full_df[tup][1])
            clust_r_arr = np.concatenate(clust_r)
            clust_data[c] = {'mean': clust_r_arr.mean(), 'std': clust_r_arr.std(), 'terr':clust_terr, 'datapoints':clust_r_arr}
        clust_terrs = np.array([clust_data[c]['terr'] for c in clust_data.keys()])
        clust_ratios = clust_terrs / clust_terrs.sum()
        for i, c in enumerate(clust_data.keys()):
            clust_data[c]['terr_fraction'] = clust_ratios[i]
        return clust_data

    # by default uses last sample
    def visualize_ACDP(self, 
                   save_path, 
                   use_cluster_stats = False, 
                   plot_hist=True, 
                   plot_real_cov=False, 
                   plot_SNP_imbalance=False,
                   cdp_draw=None):
    
        if self.draw_idx is not None:
            # override to use the only draw used
            cdp_draw = self.draw_idx
        # precompute the fallback ADP counts
        ADP_dict = {}
        for ADP, group in self.cov_df.loc[self.cov_df.dp_draw == self.default_draw].groupby('allelic_cluster'):
            ADP_dict[ADP] = (group['maj_count'].sum(), group['min_count'].sum())
        
        # set up canvas according to options
        if plot_hist:
            fig = plt.figure(6, figsize=[22, 7])
            plt.clf()
            ax_g = fig.add_axes([0,0,0.85,1])
            ax_hist = fig.add_axes([0.855,0,0.145,1])
            ax_hist.tick_params(axis="y", labelleft=False, left=False, right=True, labelright=True)
        else:
            fig = plt.figure(6, figsize=[22,7])
            plt.clf()
            ax_g = plt.gca()
        
        # to plot the cdp colors agnostic of yscale we create a new transform
        cdp_trans = mpl.transforms.blended_transform_factory(ax_g.transData, ax_g.transAxes)
        
        if plot_SNP_imbalance:
            #make a twin axis for the allelic imbalance
            ax_g2 = ax_g.twinx()
            ax_g2.set_ylabel('allelic imbalance')
        
        cluster_stats = self.precompute_cluster_params(cdp_draw)
        
        cluster_colors = self._get_cluster_colors()
        cdp_colors = self._get_cdp_colors() if cdp_draw is None else self._get_cdp_colors(cdp_draw)
        if plot_SNP_imbalance or self.wgs:
            adp_colors = self._get_adp_colors()
        
        full_df = list(self.cov_df.groupby(['allelic_cluster', 'cov_DP_cluster', 'allele', 'dp_draw']))

        max_acov = 0
        for i, c in enumerate(self.cluster_dict.keys()):
            cluster_real_data = []
            for tup in self.cluster_dict[c]:
                tup_data = full_df[tup]
                
                quad = tup_data[0]
                x = tup_data[1]
                
                # if we're only plotting a single cdp cluster we can skip irrelevant tuples
                if cdp_draw is not None and quad[3] != cdp_draw:
                    continue

                adp = x['allelic_cluster'].values[0]
                if len(x) > 10:
                    major, minor = x['maj_count'].sum(), x['min_count'].sum()
                else:
                    major, minor = ADP_dict[adp]

                # scatter plot for each of the coverage bins in the segment
                locs, f = self._scatter_apply(x, minor, major)
                y = np.exp(x.cov_DP_mu)
                acov_levels = f*y
                
                 #plot shrinkage estimates in order to see small segments
                ax_g.scatter(
                    locs,
                    acov_levels,
                    color=np.array(cluster_colors)[i],
                    marker='.',
                    alpha=0.15,
                    s=8
                )
                
                #keep track of the largest allelic coverage level seen so far
                max_acov = max(max_acov, max(acov_levels))
                
                #save real coverage values from the first cdp draw or chosen draw if given
                if quad[3] == 0 or (cdp_draw is not None and quad[3]==cdp_draw):
                    real_data = f * self._get_real_cov(x)
                    cluster_real_data.append(real_data)
                    
                    # plot real coverage data as a grey scatter
                    if plot_real_cov:
                        ax_g.scatter(
                            locs,
                            real_data,
                            color='gray',
                        marker='.',
                        alpha=0.1,
                        s=4
                        )
                    # plot the allelic imbalance of each coverage bin
                    if plot_SNP_imbalance:
                        ax_g2.scatter(
                            locs,
                            x['min_count']/ (x['min_count'] + x['maj_count']),
                            color=adp_colors[quad[0]],
                            marker='.',
                            alpha=0.1,
                            s=8
                        )
                    
                    #plot CDP cluster assignments
                    lc = mpl.collections.LineCollection([[(l, 0), (l, 0.01)] for l in locs], color = cdp_colors[quad[1]], transform=cdp_trans)
                    ax_g.add_collection(lc)
                
                if not self.wgs:
                    #plot patches for each segment within the tuple
                    seg_intervals = self._get_tuple_intervals(x.index)
                    for intv in seg_intervals:
                        #plot patch with centered at cluster mean with height equal to cluster 95% CI
                        if use_cluster_stats:
                            # use overall cluster statistics
                            tup_mean = cluster_stats[c]['mean']
                            tup_std = cluster_stats[c]['std']
                        else:
                            # use statistics from the tuple only
                            tup_mean = self.segment_r_list[tup].mean()
                            tup_std = np.sqrt(self.segment_V_list[tup])
                        
                        tup_width = x.iloc[intv[1]].end_g - x.iloc[intv[0]].start_g
                        tup_allele = x.iloc[0].allele
                        
                       # draw the acdp segment
                        ax_g.add_patch(mpl.patches.Rectangle(
                          (x.iloc[intv[0]].start_g, tup_mean - 1.95 * tup_std),
                          tup_width,
                          np.maximum(0, 2 * 1.95 * tup_std),
                          facecolor = cluster_colors[i],
                          fill = True, alpha=0.5 if cdp_draw is None else 0.8,
                          edgecolor = 'b' if tup_allele > 0 else 'r', # color edges according to allele
                          linewidth = 1 if tup_width > 1000000 else 0.5,
                          ls = (0, (0,5,5,0)) if tup_allele > 0 else (0, (5,0,0,5)) # color edges with alternating pattern according to allele
                        ))
                        
                        # show cdp cluster at bottom of plot
                        if quad[3] == 0 or (cdp_draw is not None and quad[3]==cdp_draw):
                            ax_g.add_patch(mpl.patches.Rectangle(
                              (x.iloc[intv[0]].start_g, 0),
                              tup_width,
                              0.01,
                              transform=cdp_trans,
                              facecolor = cdp_colors[quad[1]],
                              fill = True, alpha=1,
                            ))
            if self.wgs:
                #only plot for first dp cluster for now
                # plot patches for each cluster based only on the allleic cluster and allele
                cdp_draw_idx = 0 if cdp_draw is None else cdp_draw
                all_cluster_bins = pd.concat([full_df[tup][1] for tup in self.cluster_dict[c] if full_df[tup][0][3]==cdp_draw_idx])
                all_cluster_bins = all_cluster_bins.sort_index()
                # plot patches for each contiguous segment within a adp, allele tuple
                for label, bins in all_cluster_bins.groupby(['allelic_cluster', 'allele']):
                    seg_intervals = self._get_tuple_intervals(bins.index)
                    for intv in seg_intervals:
                        #only use cluster stats since individual tuples are short
                        tup_mean = cluster_stats[c]['mean']
                        tup_std = cluster_stats[c]['std']
                        tup_width = bins.iloc[intv[1]].end_g - bins.iloc[intv[0]].start_g
                        tup_allele = bins.iloc[0].allele
                        
                        
                        # draw the acdp segment
                        ax_g.add_patch(mpl.patches.Rectangle(
                          (bins.iloc[intv[0]].start_g, tup_mean - 1.95 * tup_std),
                          tup_width,
                          np.maximum(0, 2 * 1.95 * tup_std),
                          facecolor = cluster_colors[i],
                          fill = True, alpha=0.5 if cdp_draw is None else 0.8,
                          edgecolor = 'b' if tup_allele > 0 else 'r', # color edges according to allele
                          linewidth = 1 if tup_width > 1000000 else 0.5,
                          ls = (0, (0,5,5,0)) if tup_allele > 0 else (0, (5,0,0,5)) # color edges with alternating pattern according to allele
                        ))
                        
                        # show adp cluster at bottom of plot
                        ax_g.add_patch(mpl.patches.Rectangle(
                          (bins.iloc[intv[0]].start_g, 0),
                          tup_width,
                          0.01,
                          transform=cdp_trans,
                          facecolor = adp_colors[label[0]],
                          fill = True, alpha=1,
                        ))
         
            #save real data for histogram if there were any in the draw(s) of interest
            cluster_stats[c]['real_data'] = np.concatenate(cluster_real_data) if len(cluster_real_data) > 0 else []
            
            # draw cluster average coverage level for large clusters
            if cluster_stats[c]['terr_fraction'] > 0.05:
                ax_g.axhline(cluster_stats[c]['mean'], color=cluster_colors[i], alpha=0.3, linewidth=2)
    
        #round_max_acov = 25*int(np.ceil(max_acov / 25))
        round_max_acov = 25 * int(np.ceil(max([d['mean'] + 4 * d['std'] for d in cluster_stats.values()]) / 25))
        
        #now that we know the maximum allelic coverage value we can set our bins and plot the histogram
        if plot_hist:
            real = []
            hist_bin_width = 1
            for i, c in enumerate(self.cluster_dict.keys()):
                ax_hist.hist(cluster_stats[c]['datapoints'], bins = np.r_[:round_max_acov:hist_bin_width], alpha = 0.5, orientation='horizontal', color= cluster_colors[i])
                real.append(cluster_stats[c]['real_data'])
            ax_hist2=ax_hist.twiny()
            ax_hist2.hist(np.concatenate(real), bins = np.r_[:round_max_acov:hist_bin_width], alpha = 0.1, orientation='horizontal', color= 'k')
        
        plt.sca(ax_g)
        plot_chrbdy(self.cytoband_file)

        plt.xlabel("Genomic position")
        plt.ylabel("Coverage of major/minor alleles")
        
        # find last chrom to plot using cytoband file
        cb = parse_cytoband(self.cytoband_file)
        ends = cb.loc[cb.start!=0]
        last_chr_idx = np.searchsorted(np.cumsum(ends.end), self.cov_df.end_g.max()) + 1
        last_g = ends.end[:last_chr_idx].sum()

        ax_g.set_xlim((0.0, last_g))
        ax_g.set_ylim([0, round_max_acov])
        
        if plot_hist:
            ax_hist.set_ylim([0, round_max_acov])
        
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
 
    def visualize_ACDP_clusters(self, save_path):
        #plot individual tuples within clusters
        rs = []
        rmax=0
        for c in self.cluster_dict:
            rs.append(
                (np.array([np.array(self.segment_r_list[i]).mean() for i in self.cluster_dict[c]]).mean(), c))
            rmax = max(rmax, np.array([np.array(self.segment_r_list[i]).max() for i in self.cluster_dict[c]]).max())
        f, ax = plt.subplots(1, figsize=[19.2, 10])
        #plt.clf()
        counter = 0
        cc = 0
        for c in [t[1] for t in sorted(rs)]:
            c0 = counter
            vals = [np.array(self.segment_r_list[i]) for i in self.cluster_dict[c]]

            for arr in vals:
                ax.scatter(np.repeat(counter, len(arr)), arr, marker='.', alpha=0.05, s=4)
                counter += 1
            ax.add_patch(mpl.patches.Rectangle((c0, 0), counter - c0, rmax, fill=True, alpha=0.10,
                                               color=colors[cc % len(colors)]))
            if self.cluster_counts[c] > 2000:
                ax.text(c0 + (counter - c0) / 2, 0, '{}'.format(c), horizontalalignment='center')
            cc += 1

        plt.savefig(os.path.join(save_path, 'acdp_tuples_plot.png'), dpi=300)

        #simple clusters plot
        f, ax = plt.subplots(1, figsize=[19.2, 10])
        #plt.clf()

        counter = 0
        for c in [t[1] for t in sorted(rs)]:
            vals = [np.array(self.segment_r_list[i]).mean() for i in self.cluster_dict[c]]
            ax.scatter(np.r_[counter:counter + len(vals)], vals, 
                    alpha = [0.6 if s in self.greylist_segments else 1 for s in self.cluster_dict[c]])
            counter += len(vals)

        plt.savefig(os.path.join(save_path, 'acdp_clusters_plot.png'), dpi=300)

    # add allelic cluster information to the acdp df
    def prepare_df(self):
        cluster_stats = self.precompute_cluster_params()
        self.cov_df.loc[:, 'acdp_cluster'] = -1
        self.cov_df.loc[:, 'cluster_mu'] = np.nan
        self.cov_df.loc[:, 'cluster_sigma'] = np.nan
        for c in self.cluster_dict:
            for seg in self.cluster_dict[c]:
                self.cov_df.loc[self.cov_df.acdp_segID == seg, ['acdp_cluster', 'cluster_mu', 'cluster_sigma']] = c, cluster_stats[c]['mean'], cluster_stats[c]['std']

    # create allelic copy state segment dataframe in allelic capseg format
    # TODO: will only use the default/selected draw. Need to figure out how to handle inconsistant draws
    def create_allelic_segs_df(self):
        ## subset to defualt draw
        df = self.cov_df.loc[self.cov_df.dp_draw == self.default_draw]
        ## acdp_df is in format major allele bins (in genomic order), minor allele bins (in order)
        ## and hence we can split halfway
        num_bins = (df.allele == 1).sum()
        
        ## subset to relevant columns and convert to numpy for speed
        row_idxs = [df.columns.get_loc(i) for i in ['chr', 'start', 'end', 'acdp_cluster', 'cluster_mu', 'cluster_sigma']]
        maj_vals = df.iloc[:num_bins, row_idxs].values
        min_vals = df.iloc[num_bins:, row_idxs].values
        
        collapsed = []
        maj_chr, cur_start, cur_end, maj_cluster, maj_mu, maj_sigma = maj_vals[0]
        _, _, _, min_cluster, min_mu, min_sigma = min_vals[0]
        for i in range(num_bins):
            cur_chr, cur_maj_cluster = maj_vals[i, [0, 3]]
            cur_min_cluster = min_vals[i, 3]
            
            if cur_chr != maj_chr or cur_maj_cluster != maj_cluster or cur_min_cluster != min_cluster:
                collapsed.append((maj_chr, cur_start, cur_end, maj_mu, maj_sigma, min_mu, min_sigma))
                maj_chr, cur_start, cur_end, maj_cluster, maj_mu, maj_sigma = maj_vals[i]
                _, _, _, min_cluster, min_mu, min_sigma = min_vals[i]
            else:
                cur_end = maj_vals[i, 2]
        collapsed.append((maj_chr, cur_start, cur_end, maj_mu, maj_sigma, min_mu, min_sigma))
        return pd.DataFrame(collapsed, columns = ["Chromosome", 'Start.bp', 'End.bp', 'mu.major', 'sigma.major', 'mu.minor', 'sigma.minor']).astype({'Chromosome':int, 'Start.bp':int, 'End.bp':int, 'mu.major':float, 'sigma.major':float, 'mu.minor':float, 'sigma.minor':float})

