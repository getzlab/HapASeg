import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import matplotlib as mpl

def interval_remap(c0, s0, e0, s1, e1):
    return (c0 - s0)/(e0 - s0)*(e1 - s1) + s1

def load_mad_results_df(res):
    mad_results = []
    for flow in tqdm.tqdm(list(res.keys())):
        sample_type, complexity, purity = re.search(r"(.*)_benchmarking_profile_(\d+)_\d+_(.*)", flow).groups()
        downstream_res = [s for s in list(res[flow].keys()) if 'Downstream' in s]
        for ds in downstream_res:
            method = ''.join([s for s in re.search(r"Downstream_(.*)_Analysis(_unclustered)?", ds).groups() if s is not None])
            if res[flow][ds] is not None:
                mad_path = res[flow][ds].results.outputs.comparison_results[0]
                # we also want to extract the inferred purity for hapaseg runs
                if 'HapASeg' in method:
                    purity_path = res[flow]['Hapaseg_run_acdp'].results.outputs.acdp_optimal_fit_params[0]
                    inf_purity = float(open(purity_path, 'r').readlines()[1].split('\t')[0])
                else:
                    inf_purity = -1
                mad_results.append((sample_type, complexity, purity, inf_purity, method, float(open(mad_path, 'r').read().split()[5])))
            else:
                print("could not load results for ", flow, ds)
    results_df =  pd.DataFrame(mad_results, columns = ['sample', 'complexity', 'purity', 'inferred_purity', 'method', 'mad_score'])
    
    results_df.loc[:, "purity"] = results_df['purity'].astype(float)
    results_df.loc[:, 'complexity'] = results_df['complexity'].astype(float)
    results_df.loc[:, 'complexity'] = results_df['complexity'] / 10000
    results_df['complexity_size'] = interval_remap(results_df['complexity'], results_df['complexity'].min(), results_df['complexity'].max(), 16,128)
    return results_df

# old way of gathering results
#results_files = glob.glob('/mnt/nfs/benchmarking_dir/*_benchmarking_profile*/Downstream*/jobs/0/workspace/*comparison_results.txt')
#results_tups = [(f, *re.search(r".*(FF.*)_benchmarking_profile_(\d+)_\d+_(.*).Downstream_(.*)_Analysis.*", f).groups()) for f in results_files]
#mad_results = [float(open(tups[0], 'r').read().split()[5]) for tups in results_tups]
#results_df = pd.DataFrame(results_tups, columns = ['path', 'sample', 'complexity', 'purity', 'method']) 

res = pd.read_pickle('./benchmarking/final_benchmakring_results.pickle')
results_df = load_mad_results_df(res)


## plot clustered vs unclustered
samples = ['FF', 'FFPE_CH1022', 'FFPE_CH1032', 'exome_TWIST']
sample_dict = {'FF': 'Fresh Frozen PCR-Free', 'FFPE_CH1032':'FFPE High Quality', 'FFPE_CH1022': 'FFPE Degraded', 'exome_TWIST':'Fresh Frozen WES'}
methods = ['HapASeg', 'HapASeg_unclustered']
fig, axs = plt.subplots(4,1, figsize=(6,14), sharex=True)
ax_lst = axs.flatten()
colors = plt.get_cmap("tab10")
color_dict = dict(zip(methods, range(len(methods))))
marker_dict = dict(zip(methods, ['o', 'x']))
for i, sample in enumerate(samples):
    ax = ax_lst[i]
    ax.set_yscale('log')
    ax.set_ylabel('MAD')
    ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    for method in methods:
        res = results_df.loc[(results_df['method'] == method) & (results_df['sample']==sample), ['purity', 'mad_score', 'complexity_size']].sort_values(by='purity').values
        jit = np.random.rand(len(res[:,0]))/ 20 - 0.025
        ax.scatter(res[:,0] + jit, res[:,1], color = colors(color_dict[method]), label=method, alpha = 0.3, marker = "o", s = res[:,2])

    for method in methods:
        means = results_df.loc[(results_df['method'] == method) & (results_df['sample']==sample), ['purity', 'mad_score',]].sort_values(by='purity').groupby('purity').mean()
        ax.scatter(means.index, means.values, marker=marker_dict[method], color='k')
    ax.set_title(f'{sample_dict[sample]}')
ax.set_xlabel('purity')
plt.legend()
plt.tight_layout()
plt.savefig('./final_figures/1_clustered_vs_unclustered.pdf')

## purity vs inferred purity
fig, axs = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)
ax_lst = axs.flatten()
cmap = plt.get_cmap('viridis')
cnorm = mpl.colors.Normalize(vmin=results_df['complexity'].min(), vmax=results_df['complexity'].max())
sm = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
for i, sample in enumerate(samples):
    ax = ax_lst[i]
    ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    data = results_df.loc[(results_df['sample'] == sample) & (results_df['method'] =='HapASeg'), ['purity', 'inferred_purity', 'complexity']].values
    jit = np.random.rand(len(data[:,0]))/ 20 - 0.025
    ax.scatter(data[:,0] + jit, data[:,1], c=data[:,2], cmap=cmap, alpha=0.2)
    means = results_df.loc[(results_df['sample'] == sample) & (results_df['method'] =='HapASeg'), ['purity', 'inferred_purity']].groupby('purity').mean()
    ax.scatter(means.index, means.values, marker='x', color='k')
    ax.plot([0.1,0.9],[0.1,0.9], color='k', alpha=0.3)
    ax.set_title(f'{sample_dict[sample]}')
    if i > 1: ax.set_xlabel('purity')
    if not i % 2 :ax.set_ylabel('inferred purity')
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.85, top=0.95)
cax = plt.axes([0.9, 0.05, 0.03, 0.9])
plt.colorbar(sm, cax=cax, label='complexity')
plt.savefig('./final_figures/2_benchmarking_inferred_purities.pdf')


## bubble plots

# set unclustered theshold to 0.3 in wgs and 0.2 in WES
optimal_df = results_df.copy()
wgs_samples = ['FF', 'FFPE_CH1022', 'FFPE_CH1032']
optimal_df.loc[(optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity < 0.3) & (optimal_df['method'] == 'HapASeg_unclustered'), 'method'] = 'HapASeg_optimal'
optimal_df.loc[(optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity >= 0.3) & (optimal_df['method'] == 'HapASeg'), 'method'] = 'HapASeg_optimal'
optimal_df.loc[(~optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity < 0.2) & (optimal_df['method'] == 'HapASeg_unclustered'), 'method'] = 'HapASeg_optimal'
optimal_df.loc[(~optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity >= 0.2) & (optimal_df['method'] == 'HapASeg'), 'method'] = 'HapASeg_optimal'

# rename
opt_methods = ['Facets', 'GATK', 'Hatchet', 'ASCAT', 'HapASeg_optimal']
optimal_df = optimal_df.loc[optimal_df['method'].isin(opt_methods)]
optimal_df.loc[optimal_df['method'] == 'HapASeg_optimal', 'method'] = 'HapASeg'

methods = ['HapASeg', 'Facets', 'GATK', 'Hatchet', 'ASCAT']
colors = plt.get_cmap("tab10")
color_dict = dict(zip(methods, range(len(methods))))

for sample in samples:
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylabel('MAD')
    ax.set_xlabel('purity')
    ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    for method in methods:
        res = optimal_df.loc[(optimal_df['method'] == method) & (optimal_df['sample']==sample), ['purity', 'mad_score', 'complexity_size']].sort_values(by='purity').values
        jit = np.random.rand(len(res[:,0]))/ 20 - 0.025 
        ax.scatter(res[:,0] + jit, res[:,1], color = colors(color_dict[method]), label=method, alpha = 0.3, marker = "o", s = res[:,2])
    plt.title(f'{sample_dict[sample]}')
    plt.legend()
    plt.savefig(f'./final_figures/3_{sample}_benchmarking_bubble_plot.pdf')

