import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

results_files = glob.glob('/mnt/nfs/workspace/benchmarking_profile_*_paired/Downstream_*/jobs/0/workspace/benchmarking_profile_*_comparison_results.txt')
results_tups = [(f, *re.search(r"benchmarking_profile_\d+_\d+_(easy|medium|hard)_(.*)_paired.Downstream_(.*)_Analysis.*", f).groups()) for f in results_files]
mad_results = [float(open(tups[0], 'r').read().split()[5]) for tups in results_tups]

pd.DataFrame(list(zip(results_tups, mad_results)))
pd.DataFrame(results_tups)
results_df = pd.DataFrame(results_tups, columns = ['path', 'mode', 'purity', 'method'])
results_df.purity = results_df.purity.astype(float)
results_df['mad_score'] = mad_results

fig = plt.figure()
ax = plt.gca()
ax.set_yscale('log')
colors = plt.get_cmap("tab10")
color_dict = dict(zip(results_df.method.unique(), range(len(results_df.method.unique()))))
sizes_dict = {'hard':48, 'medium': 24, 'easy':16}
for method in results_df.method.unique():
    for mode in ['easy', 'medium', 'hard']:
        res = results_df.loc[(results_df['mode'] == mode) & (results_df['method'] == method), ['purity', 'mad_score']].sort_values(by='purity').values
        if mode == 'medium':
            ax.scatter(res[:,0], res[:,1], color = colors(color_dict[method]), label=method, alpha = 0.4, marker = "o", s = sizes_dict[mode])
        else:
            ax.scatter(res[:,0], res[:,1], color = colors(color_dict[method]), alpha = 0.4, marker = "o", s = sizes_dict[mode])
plt.legend()
plt.xlabel('purity')
plt.ylabel('MAD')

def extract_segment_results(df_path, method, complexity, purity):
    df = pd.read_csv(df_path, sep='\t')
    trim_df = df.loc[~df.unique, ['length_overlap', 'major_AD', 'minor_AD', 'major_ccf', 'minor_ccf']]
    trim_df['method'] = method
    trim_df['purity'] = purity
    trim_df['complexity'] = complexity
    return trim_df

segfiles = glob.glob('/mnt/nfs/workspace/benchmarking_profile_*_paired/Downstream_*/jobs/0/workspace/benchmarking_profile_*_comparison_segfile.tsv')
segfile_tups = [(f, *re.search(r"benchmarking_profile_\d+_\d+_(easy|medium|hard)_(.*)_paired.Downstream_(.*)_Analysis.*", f).groups()) for f in segfiles]
res_dfs = [extract_segment_results(df_path, method, complexity, float(purity)) for df_path, complexity, purity, method in segfile_tups]
seg_results_df = pd.concat(res_dfs).reset_index(drop=True)

minor_df = seg_results_df.loc[:, ['length_overlap', 'minor_AD', 'minor_ccf', 'method', 'purity', 'complexity']].rename({'minor_AD':'AD', 'minor_ccf':'ccf'}, axis=1)
minor_df['allele'] = 'minor'

major_df = seg_results_df.loc[:, ['length_overlap', 'major_AD', 'major_ccf', 'method', 'purity', 'complexity']].rename({'major_AD':'AD', 'major_ccf':'ccf'}, axis=1)
major_df['allele'] = 'major'

seg_results_df = pd.concat([minor_df, major_df]).reset_index(drop=True)
length_bins = [1] + [2e4 * 5**i for i in range(0,7)] + [1e9]
ccf_cutoff = 0.7


