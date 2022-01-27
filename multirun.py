import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as stats
import scipy.special as ss
import sortedcontainers as sc
import os
import tqdm
import pickle

from statsmodels.discrete.discrete_model import NegativeBinomial as statsNB

os.environ["CAPY_REF_FA"] = "/home/opriebe/data/ref/hg19/Homo_sapiens_assembly19.fasta"
import hapaseg.coverage_MCMC as mcmc_cov
import hapaseg.NB_coverage_MCMC as nb_cov
from capy import mut, seq

import hapaseg.coverage_DP as dp_cov
import hapaseg.a_cov_DP as dp_a_cov

colors = mpl.cm.get_cmap("tab20").colors

multidraw_df = pd.read_pickle('exome_results/acdp_df.pickle')

mcmc_data = np.load('exome_results/coverage_mcmc_clusters/cov_mcmc_collected_data.npz')
beta = mcmc_data['beta']

allelic_segs = pd.read_pickle("exome/6_C1D1_META.allelic_segs.auto_ref_correct.overdispersion92.no_phase_correct.pickle")
chrbdy = allelic_segs.dropna().loc[:, ["start", "end"]]
chr_ends = chrbdy.loc[chrbdy["start"] != 0, "end"].cumsum()

f_clust, axs = plt.subplots(4,3, figsize = (30,30))
ax_lst_clust = axs.flatten()
best_a_cov_dp = None
MLs = []
best_total_ML = -1e30
models = []
for run in range(12):
    a_cov_dp = dp_a_cov.Run_Cov_DP(multidraw_df.copy(), beta, coverage_prior=True, seed_all_clusters=False)
    a_cov_dp.run(200)
    counter=0
    for c in a_cov_dp.cluster_dict:
        vals = [np.array(a_cov_dp.segment_r_list[i]).mean() for i in a_cov_dp.cluster_dict[c]]
        ax_lst_clust[run].scatter(np.r_[counter:counter+len(vals)], vals)
        counter+= len(vals)
    models.append(a_cov_dp)
    total_ML = a_cov_dp.DP_total_history[-1] + a_cov_dp.ML_total_history[-1]
    MLs.append(total_ML)
    if total_ML > best_total_ML:
        best_total_ML = total_ML
        best_a_cov_dp = a_cov_dp
    ax_lst_clust[run].set_title('total_ML:{}'.format(np.around(total_ML,2)))

plt.savefig('./multirun_res_cold_200.png')
with open("./saved_multirun_cold_200.pickle", "wb") as f:
    pickle.dump(models, f)
