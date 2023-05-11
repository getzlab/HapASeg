import pandas as pd
import numpy as np
from cnv_suite.simulate.cnv_profile import CNV_Profile
from cnv_suite.visualize.plot_cnv_profile import *
from capy import mut
import tqdm
from capy.num import interval_remap
import scipy.stats as stats

num_profiles = 50

# variables to sample over are:
## number of arm level events
## number of focal events
## prob of whole chromosome event vs arm-level
## ratio clonal

params = []

for i in tqdm.tqdm(range(num_profiles)):
    arm_n = np.random.choice(np.arange(20,300))
    focal_n = np.random.choice(np.arange(20, 500))
    num_subclones = np.random.choice(3)

    arm_num = np.random.binomial(arm_n, 0.25)
    focal_num = np.random.binomial(focal_n, 0.1)
    p_whole = np.random.beta(2,2)
    ratio_clonal = np.random.beta(4, 1) if num_subclones > 0 else 1.0
    num_micro_focal = (np.random.geometric(0.3) -1) * (num_subclones + 1)

    cnv_profile = CNV_Profile(num_subclones=num_subclones, ref_build="hg38")
    cnv_profile.add_cnv_events(arm_num, focal_num, num_micro_focal, p_whole, ratio_clonal, p_arm_del=0.51, p_focal_del=0.49,
                           median_focal_length = 1.8e6, chromothripsis=False, wgd=False)

    cnv_profile.calculate_profiles()

    fig = plt.figure(figsize=(12,8))
    ax = plt.gca()
    plot_acr_static(cnv_profile.cnv_profile_df, ax = ax, csize=cnv_profile.csize)

    # compute variance    
    df = cnv_profile.cnv_profile_df
    df['lens'] = df['End.bp'] - df['Start.bp']
    mus = np.r_[df['mu.major'].values, df['mu.minor'].values]
    lens = np.r_[df['lens'].values, df['lens'].values]
    
    bins, _ = np.histogram(mus, bins = np.r_[0:20:0.01])
    entropy = stats.entropy(bins)

    len_bins, _ = np.histogram(lens, bins = np.r_[0:250e6:1e5])
    len_entropy = stats.entropy(len_bins) / 1.5
    total_entropy = entropy + len_entropy

    # add var to plot
    ax.set_title(f'profile_{i} entropy: {np.around(entropy, 3)}  len_entropy: {np.around(len_entropy, 3)}  sum: {np.around(total_entropy, 3)}') 
    
    label = f'benchmarking_profiles/benchmarking_profile_{int(10000 * total_entropy)}_{i}'
    plt.savefig('./' + label + '.png')
    cnv_profile.to_pickle('./' + label + '.pickle')
    
    # save all params
    params.append((label, arm_num, focal_num, p_whole, ratio_clonal, entropy, len_entropy, entropy + len_entropy))

param_df = pd.DataFrame(params, columns = ['label', 'arm_num', 'focal_num', 'p_whole', 'ratio_clonal', 'entropy', 'len_entropy', 'sum_entropy'])
param_df.to_csv('./benchmarking_profiles/params.txt', sep='\t', index=True)

