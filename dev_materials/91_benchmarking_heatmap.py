import glob
import sys
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import tqdm

def extract_segment_results(df_path, sample_name, method, complexity, purity, inf_purity):
    df = pd.read_csv(df_path, sep='\t')
    trim_df = df.loc[~df.unique, ['length_overlap', 'major_AD', 'minor_AD', 'major_ccf', 'minor_ccf', 'gt_event_length']]
    trim_df['method'] = method
    trim_df['purity'] = purity
    trim_df['sample'] = sample_name
    trim_df['complexity'] = complexity
    trim_df['inferred_purity'] = inf_purity
    return trim_df

def create_results_array(seg_results_df, sample_names, methods, purities, length_bins, ccf_bins):
    results_arr = np.zeros(shape=(len(sample_names), len(methods), len(purities), len(ccf_bins) -1, len(length_bins) - 1))
    counts_arr = np.zeros(shape=(len(sample_names), len(methods), len(purities), len(ccf_bins) -1, len(length_bins) - 1))
    for sample_i, sample in enumerate(sample_names):
        sample_mask = seg_results_df['sample'] == sample
        for method_i, method in enumerate(methods):
            method_mask = seg_results_df['method'] == method
            for purity_i, purity in enumerate(purities):
                purity_mask = np.isclose(seg_results_df.purity, purity)
                for len_i in range(0, len(length_bins) - 1):
                    len_mask = (seg_results_df['gt_event_length'] > length_bins[len_i]) & (seg_results_df['gt_event_length'] <= length_bins[len_i+ 1])
                    for ccf_i in range(0, len(ccf_bins) - 1):
                        df_res = seg_results_df.loc[lambda x: sample_mask &
                                                              method_mask &
                                                              purity_mask &
                                                              len_mask &
                                                              (x.ccf > ccf_bins[ccf_i]) & (x.ccf <= ccf_bins[ccf_i + 1])]
                        if len(df_res) == 0:
                            continue
                        else:
                            weighted_avg = np.average(df_res.AD.values, weights=df_res['length_overlap'].values)
                            results_arr[sample_i, method_i, purity_i, ccf_i, len_i] = weighted_avg
                            counts_arr[sample_i, method_i, purity_i, ccf_i, len_i] = len(df_res)
    return results_arr, counts_arr

# old way of gathering results
#segfiles = glob.glob('/mnt/nfs/benchmarking_dir/*_benchmarking_profile_*/Downstream_*/jobs/0/workspace/*_comparison_segfile.tsv')
#segfile_tups = [(f, *re.search(r".*(FF.*)_benchmarking_profile_(\d+)_\d+_(.*).Downstream_(.*)_Analysis.*", f).groups()) for f in segfiles]
#res_dfs = [extract_segment_results(df_path, sample_name, method, int(complexity), float(purity)) for df_path, sample_name, complexity, purity, method in segfile_tups]
#seg_results_df = pd.concat(res_dfs).reset_index(drop=True)
#seg_results_df.loc[:, 'complexity'] = seg_results_df['complexity'] / 10000
#minor_df = seg_results_df.loc[:, ['length_overlap', 'gt_event_length','sample', 'minor_AD', 'minor_ccf', 'method', 'purity', 'complexity']].rename({'minor_AD':'AD', 'minor_ccf':'ccf'}, axis=1)
#minor_df['allele'] = 'minor'
#major_df = seg_results_df.loc[:, ['length_overlap','gt_event_length',  'sample', 'major_AD', 'major_ccf', 'method', 'purity', 'complexity']].rename({'major_AD':'AD', 'major_ccf':'ccf'}, axis=1)
#major_df['allele'] = 'major'

def load_seg_results_df(res):
    res_dfs = []
    for flow in tqdm.tqdm(list(res.keys())):
        sample_type, complexity, purity = re.search(r"(.*)_benchmarking_profile_(\d+)_\d+_(.*)", flow).groups()
        downstream_res = [s for s in list(res[flow].keys()) if 'Downstream' in s]
        for ds in downstream_res:
            method = ''.join([s for s in re.search(r"Downstream_(.*)_Analysis(_unclustered)?", ds).groups() if s is not None])
            if res[flow][ds] is not None:
                segfile_path = res[flow][ds].results.outputs.comparison_segfile[0]
                # we also want to extract the inferred purity for hapaseg runs
                if 'HapASeg' in method:
                    purity_path = res[flow]['Hapaseg_run_acdp'].results.outputs.acdp_optimal_fit_params[0]
                    inf_purity = float(open(purity_path, 'r').readlines()[1].split('\t')[0])
                else:
                    inf_purity = -1
                res_dfs.append(extract_segment_results(segfile_path, sample_type, method, int(complexity), float(purity), inf_purity))
            else:
                print("could not load results for ", flow, ds)
    seg_results_df = pd.concat(res_dfs).reset_index(drop=True)
    seg_results_df.loc[:, 'complexity'] = seg_results_df['complexity'] / 10000
    minor_df = seg_results_df.loc[:, ['length_overlap', 'gt_event_length','sample', 'minor_AD', 'minor_ccf', 'method', 'purity', 'inferred_purity', 'complexity']].rename({'minor_AD':'AD', 'minor_ccf':'ccf'}, axis=1)
    minor_df['allele'] = 'minor'

    major_df = seg_results_df.loc[:, ['length_overlap','gt_event_length',  'sample', 'major_AD', 'major_ccf', 'method', 'purity', 'inferred_purity', 'complexity']].rename({'major_AD':'AD', 'major_ccf':'ccf'}, axis=1)
    major_df['allele'] = 'major'
    seg_results_df = pd.concat([minor_df, major_df]).reset_index(drop=True)
    return seg_results_df

def main():
    try:
        flow_results = sys.argv[1]
        print(f"attempting to load flow results from {flow_results}")
        res = pd.read_pickle(flow_results)
    except:
        print("could not load results pickle")
        return 
   
    ## filtering out simulated tumor that is too simple for proper comparison
    res = {key : val for key, val in res.items() if 'benchmarking_profile_37856' not in key}
 
    seg_results_df = load_seg_results_df(res)

    # we set unclustered theshold to 0.3 in wgs and 0.2 in WES
    optimal_df = seg_results_df.copy()
    wgs_samples = ['FF', 'FFPE_CH1022', 'FFPE_CH1032']
    optimal_df.loc[(optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity < 0.3) & (optimal_df['method'] == 'HapASeg_unclustered'), 'method'] = 'HapASeg_optimal'
    optimal_df.loc[(optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity >= 0.3) & (optimal_df['method'] == 'HapASeg'), 'method'] = 'HapASeg_optimal'
    optimal_df.loc[(~optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity < 0.2) & (optimal_df['method'] == 'HapASeg_unclustered'), 'method'] = 'HapASeg_optimal'
    optimal_df.loc[(~optimal_df['sample'].isin(wgs_samples)) & (optimal_df.inferred_purity >= 0.2) & (optimal_df['method'] == 'HapASeg'), 'method'] = 'HapASeg_optimal'

    # rename
    opt_methods = ['Facets', 'GATK', 'Hatchet', 'ASCAT', 'HapASeg_optimal']
    optimal_df = optimal_df.loc[optimal_df['method'].isin(opt_methods)]
    optimal_df.loc[optimal_df['method'] == 'HapASeg_optimal', 'method'] = 'HapASeg'

    length_bins = [1] + [5e4 * 10**i for i in range(0,5)]
    ccf_bins = [0,0.6,.99999,1]
    methods = ['HapASeg', 'ASCAT', 'GATK', 'Facets', 'Hatchet']
    purities = np.r_[0.1:1:0.1]
    samples = ['FF', 'FFPE_CH1022', 'FFPE_CH1032']   
 
    results_arr, counts_arr = create_results_array(optimal_df, samples, methods, purities, length_bins, ccf_bins)
    #np.savez('heatmap_array_results.npz', results_arr=results_arr, counts_arr=counts_arr)
    
    # reorder axis so high quality comes before degraded
    tmp = results_arr[1].copy()
    results_arr[1] = results_arr[2]
    results_arr[2] = tmp
    

    plt.set_cmap('turbo')
    fig = plt.figure(figsize=(10,18))
    grid = AxesGrid(fig, (0.05, 0.06, 0.85, 0.85),
                    nrows_ncols=(len(methods),len(samples)),
                    axes_pad=0.1,
                    share_all=False,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_pad=0.6    
                )
    vmin = 0.1
    vmax = 100
    nlb = len(length_bins) - 1

    for row in range(len(methods)):
        for col in range(len(samples)):
            ax = grid[row * len(samples) + col]
            im = ax.imshow(results_arr[col, row].reshape(9,15).T,  norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
            ax.plot([-0.5,8.5],[nlb - 0.5, nlb - 0.5],linewidth=1.2,color='k')
            ax.plot([-0.5,8.5],[2*nlb - 0.5, 2 * nlb - 0.5],linewidth=1.2,color='k')
            if col == 0:
                ax.set_ylabel(methods[row])
                ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['50kb', '500kb', '5Mb', '50Mb', '500Mb', '50kb', '500kb', '5Mb', '50Mb', '500Mb', '50kb', '500kb', '5Mb', '50Mb', '500Mb'], rotation=25, fontsize=4, fontfamily='sans-serif')
            if col ==2:
                ax2 = ax.secondary_yaxis('right')
                ax2.set_yticks([2,7,12], ['low ccf', 'high ccf', 'clonal'], fontsize=7)
                ax2.yaxis.set_ticks_position('none')

    grid[12].set_xticks([0,1,2,3,4,5,6,7,8], [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize=4, rotation=90, fontfamily='sans-serif')
    grid[13].set_xticks([0,1,2,3,4,5,6,7,8], [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize=4, rotation=90, fontfamily='sans-serif')
    grid[14].set_xticks([0,1,2,3,4,5,6,7,8], [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize=4, rotation=90, fontfamily='sans-serif')
    grid.cbar_axes[0].yaxis.set_ticks_position('right')
    grid.cbar_axes[0].colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)), label='AAD Score')
    grid[0].set_title('Fresh Frozen', rotation=30, fontsize=10, ha='left')
    grid[1].set_title('FFPE Quality', rotation=30, fontsize=10, ha='left')
    grid[2].set_title('FFPE Degraded', rotation=30, fontsize=10, ha='left')
    grid[-2].set_title('purity', y=-0.2, fontsize=10)
    plt.savefig('./final_figures/4_WGS_heatmap_results_plot.pdf')

    ## for a seperate WES heatmap
    samples = ['exome_TWIST']
    exome_results_arr, exome_counts_arr = create_results_array(optimal_df, samples, methods, purities, length_bins, ccf_bins)

    plt.set_cmap('turbo')
    fig = plt.figure(figsize=(10,18))
    grid = AxesGrid(fig, (0.05, 0.06, 0.85, 0.85),
                    nrows_ncols=(len(methods),len(samples)),
                    axes_pad=0.01,
                    share_all=False,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_pad=0.6
                    )
    vmin = 0.1
    vmax = 100
    nlb = len(length_bins) - 1

    for row in range(len(methods)):
        for col in range(len(samples)):
            ax = grid[row * len(samples) + col]
            im = ax.imshow(results_arr[col, row].reshape(9,15).T,  norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
            ax.plot([-0.5,8.5],[nlb - 0.5, nlb - 0.5],linewidth=1.2,color='k')
            ax.plot([-0.5,8.5],[2*nlb - 0.5, 2 * nlb - 0.5],linewidth=1.2,color='k')
            
            ax.set_ylabel(methods[row])
            ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['50kb', '500kb', '5Mb', '50Mb', '500Mb', '50kb', '500kb', '5Mb', '50Mb', '500Mb', '50kb', '500kb', '5Mb', '50Mb', '500Mb'], rotation=25, fontsize=4, fontfamily='sans-serif')
            ax2 = ax.secondary_yaxis('right')
            ax2.set_yticks([2,7,12], ['low ccf', 'high ccf', 'clonal'], fontsize=7)
            ax2.yaxis.set_ticks_position('none')    

    grid[4].set_xticks([0,1,2,3,4,5,6,7,8], [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize=4, rotation=90, fontfamily='sans-serif')
    grid[4].set_title('purity', y=-0.2, fontsize=10)
    grid[0].set_title("FF WES", rotation=30, fontsize=8)
    grid.cbar_axes[0].yaxis.set_ticks_position('right')
    grid.cbar_axes[0].colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)), label='AAD Score')
    plt.savefig('./final_figures/4_WES_heatmap_results_plot.pdf')

if __name__ == "__main__":
    main()
