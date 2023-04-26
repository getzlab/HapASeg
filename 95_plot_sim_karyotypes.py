import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cnv_suite.visualize.plot_cnv_profile import *
import glob
import scipy.stats as stats
import re

profiles = glob.glob('/home/opriebe/data/cnv_sim/benchmarking/sim_samples/benchmarking_profiles/benchmarking_profile_*.pickle')
for prof in profiles:
    i = re.search(r'.*/benchmarking_profile_\d+_(\d+).pickle', prof).groups()[0]
    cnv_profile = pd.read_pickle(prof)
    fig = plt.figure(figsize=(12,8))
    ax = plt.gca()
    plot_acr_static(cnv_profile.cnv_profile_df, ax = ax, segment_colors = '', csize=cnv_profile.csize)
    
    # compute entropy
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
    ax.set_title(f'profile_{i} entropy: {np.around(entropy, 3)}  len_entropy: {np.around(len_entropy, 3)}  sum_entropy: {np.around(total_entropy, 3)}') 
    
    label = f'benchmarking_profile_{int(10000 * total_entropy)}_{i}'
    plt.savefig('./final_figures/simulated_profiles/' + label + '.png')
    
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(140, 80))
grid = ImageGrid(fig, 111,  
                 nrows_ncols=(10, 5),
                 axes_pad=0.1,  # pad between axes in inch.
                 )
imgs = glob.glob('./final_figures/simulated_profiles/benchmarking_profile_*.png')
for ax, img in zip(grid, imgs):
    im = plt.imread(img)
    ax.imshow(im)
    ax.axis('off')
plt.tight_layout()
plt.savefig('./final_figures/simulated_profiles/consolidated_profiles.png', bbox_inches='tight')
