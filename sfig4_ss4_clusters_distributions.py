from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from clusters.clusters import (pretty_time,
                               clusters_categorise,
                               import_cluster_excel,
                               derive_pocket_number,
                               derive_granule_number,
                               weighted_average,
                               )

from clusters.clusters import GROUPER_CHLP, GROUPER_CLUSTERS, GROUPER_REPLICATES, GROUPER_CONDITIONS

from settings import font

plt.rc('font', **font)
plt.rc('pdf', fonttype=42)
plt.rc('lines', linewidth=8)

matplotlib.rcParams['axes.linewidth'] = .5

project_path = Path('/Users/leoburgy/Dropbox/buergy_ncomms/data/sfig4/')
genotype = 'ss4'
night = 16
time_points = [0, .25, .5, 8]

# Define parameters for histograms
bin_max = 16
binsize = 1
cutoff = 1
barcolor = '#2250d9'
bin_partition = np.arange(0, bin_max + 1, binsize)
tick_partition = np.arange(0, bin_max + 1, bin_max // (bin_max / (4 * binsize)))


def main():
    clusters = import_cluster_excel(project_path / 'clusters_ss4.xlsx', day_points=[0, .25, .5, 8])

    granule_number = derive_granule_number(clusters, GROUPER_CHLP)
    granule_number.to_excel(project_path / 'granules_chloroplasts.xlsx')

    # Weight between replicates
    mean_granule_number = granule_number.groupby(GROUPER_REPLICATES).mean()[['granule_number']]
    granule_number_weighted = weighted_average(mean_granule_number, to_average_column='granule_number',
                                               weight_column='chloroplast')

    # Derive the pocket number
    pocket_number = derive_pocket_number(clusters, GROUPER_CHLP)
    pocket_number_weighted = weighted_average(pocket_number, to_average_column='pocket_number',
                                              weight_column='chloroplast')

    pocket_granules_mean_weighted = pd.concat([granule_number_weighted, pocket_number_weighted], axis=1)
    pocket_granules_mean_weighted.to_excel(project_path / f'mean_pocket_granule_number_{genotype}.xlsx')

    chloroplast_number = clusters.groupby(GROUPER_REPLICATES).count()['chloroplast']
    chloroplast_number.to_excel(project_path / f'chloroplasts_examined_{genotype}.xlsx')

    # Refactor weighting of cluster sizes
    grouped_category = clusters_categorise(df=clusters, grouper=GROUPER_CLUSTERS, bin_partition=bin_partition)

    # Plot distributions of each parameters
    fig, axs = plt.subplots(nrows=3, ncols=len(time_points), figsize=(5, 3.5), sharey=False)

    a = axs[0, :]
    for i, t in enumerate(sorted(granule_number.light.unique())):
        sub = granule_number.loc[granule_number['light'] == t]
        a[i].hist(sub['granule_number'],
                  bins=bin_partition,
                  color=barcolor,
                  rwidth=.8,
                  density=True)

        if i == 1:
            a[i].set_xlabel('# granules / chloroplast')

        title = pretty_time(t)
        a[i].set_title(title)

    b = axs[1, :]
    for i, t in enumerate(sorted(granule_number.light.unique())):
        sub = pocket_number.loc[pocket_number['light'] == t]
        b[i].hist(sub['pocket_number'],
                  bins=bin_partition,
                  color=barcolor,
                  rwidth=.8,
                  density=True)

        if i == 1:
            b[i].set_xlabel('# pockets / chloroplast')

    c = axs[2, :]
    for i, t in enumerate(sorted(grouped_category.light.unique())):
        sub = grouped_category.loc[grouped_category['light'] == t]

        c[i].bar(sub['cluster_size'],
                 sub['granules_in_category'] / sub['granules_in_category'].sum() / binsize,  # normalise
                 color=barcolor,
                 align='edge',
                 width=.8)

        if i == 1:
            c[i].set_xlabel("# granules in cluster category")

    for i, ax in enumerate(axs.ravel()):
        ax.set_xlim(0, bin_max)
        ax.set_xticks(tick_partition)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', width=.5)
        if i % len(time_points) == 0:
            ax.set_ylabel('Frequency')

    fig.tight_layout(h_pad=3)
    fig.savefig(project_path / f'cluster_size_{genotype}_N{night}h_by{binsize}_cutoff{cutoff}_max{bin_max}.pdf')


if __name__ == '__main__':
    main()
