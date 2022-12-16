import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def interval_remap(c0, s0, e0, s1, e1):
    return (c0 - s0)/(e0 - s0)*(e1 - s1) + s1

results_files = glob.glob('/mnt/nfs/benchmarking_dir/*_benchmarking_profile*/Downstream*/jobs/0/workspace/*comparison_results.txt')
results_tups = [(f, *re.search(r".*(FF.*)_benchmarking_profile_(\d+)_\d+_(.*).Downstream_(.*)_Analysis.*", f).groups()) for f in results_files]
mad_results = [float(open(tups[0], 'r').read().split()[5]) for tups in results_tups]

results_df = pd.DataFrame(results_tups, columns = ['path', 'sample', 'complexity', 'purity', 'method']) 
results_df.purity = results_df.purity.astype(float)
results_df.loc[:, 'complexity'] = results_df['complexity'].astype(float)

results_df.loc[:, 'complexity'] = results_df['complexity'] / 10000
results_df['mad_score'] = mad_results
results_df['complexity_size'] = interval_remap(results_df['complexity'], results_df['complexity'].min(), results_df['complexity'].max(), 16,128)

samples = ['FF', 'FFPE_CH1022', 'FFPE_CH1032']
sample_dict = {'FF': 'Fresh Frozen PCR-Free', 'FFPE_CH1032':'FFPE High Quality', 'FFPE_CH1022': 'FFPE Degraded'}
methods = ['HapASeg', 'Facets', 'GATK', 'Hatchet', 'ASCAT']

for sample in samples:
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylabel('MAD')
    ax.set_xlabel('purity')
    ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    colors = plt.get_cmap("tab10")
    color_dict = dict(zip(methods, range(len(methods))))
    for method in results_df.method.unique():
        res = results_df.loc[(results_df['method'] == method) & (results_df['sample']==sample), ['purity', 'mad_score', 'complexity_size']].sort_values(by='purity').values
        jit = np.random.rand(len(res[:,0]))/ 20 - 0.025 
        ax.scatter(res[:,0] + jit, res[:,1], color = colors(color_dict[method]), label=method, alpha = 0.3, marker = "o", s = res[:,2])
    plt.title(f'{sample_dict[sample]}')
    plt.legend()
    plt.savefig(f'./{sample}_benchmarking_bubble_plot.pdf')
