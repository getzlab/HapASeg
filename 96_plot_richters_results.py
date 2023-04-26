import pandas as pd
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

def order_results(paths):
    ascat = [s for s in paths if 'ASCAT' in s]
    facets = [s for s in paths if 'Facets' in s]
    gatk = [s for s in paths if 'GATK' in s]
    hapaseg = [s for s in paths if 'HapASeg' in s]
    hatchet = [s for s in paths if 'Hatchet' in s]
    return [ascat[0] if len(ascat) else '', facets[0] if len(facets) else '', gatk[0] if len(gatk) else '', hapaseg[0] if len(hapaseg) else '', hatchet[0] if len(hatchet) else '']

# plot the 5 method results for each sample in stacked image 
samples = sorted([os.path.basename(s).split('_')[-1] for s in glob.glob('/mnt/nfs/benchmarking_dir/SP_Richters_Richters_CNA_pipeline_CH10**')])

for sample in samples:
    print(sample)
    paths = glob.glob(f'/mnt/nfs/benchmarking_dir/SP_Richters_Richters_CNA_pipeline_{sample}/Standard_*/jobs/0/workspace/Richters_CNA_pipeline_*_seg_plot.png')
    results = order_results(paths)
    fig = plt.figure(figsize=(14,8*5.4))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 1),axes_pad=0.1,)
    for ax, img in zip(grid, results):
        if img == '':
            # we dont have the actual results, so plot blank image
            im = plt.imread([r for r in results if r != ''][0])
            im[:,:,3] = 0.
            ax.imshow(im)
            ax.axis('off')
            continue
        im = plt.imread(img)
        ax.imshow(im)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'./final_figures/richters_plots/{sample}_results.png', bbox_inches='tight')

# plot all of the images together in one consolidated image
fig = plt.figure(figsize=(14.4 * len(samples), 8*5.4))
grid = ImageGrid(fig, 111, nrows_ncols=(5, len(samples)),axes_pad=0.1,)
N = len(samples)
for i, sample in enumerate(samples):
    paths = glob.glob(f'/mnt/nfs/benchmarking_dir/SP_Richters_Richters_CNA_pipeline_{sample}/Standard_*/jobs/0/workspace/Richters_CNA_pipeline_*_seg_plot.png')
    results = order_results(paths)
    for j, img in enumerate(results):
        ax = grid[N*j + i]
        if img == '':
            # we dont have the actual results, so plot blank image
            im = plt.imread([r for r in results if r != ''][0])
            im[:,:,3] = 0.
            ax.imshow(im)
            ax.axis('off')
            continue
        im = plt.imread(img)
        ax.imshow(im)
        ax.axis('off')
plt.tight_layout()
plt.savefig(f'./final_figures/richters_plots/consolidated_results.png', bbox_inches='tight')
