from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from clusters.clusters import (
    clusters_categorise,
    import_cluster_excel,
    derive_pocket_number,
    derive_granule_number,
    derive_chloroplasts_examined,
    summary,
    plot_histogram,
    category_plot,
)

from clusters.clusters import GROUPER_CHLP, GROUPER_CLUSTERS, GROUPER_REPLICATES

from settings import font

plt.rc('font', **font)
plt.rc('pdf', fonttype=42)
plt.rc('lines', linewidth=8)

matplotlib.rcParams['axes.linewidth'] = .5

# Define parameters for histograms
bin_max = 16
bin_size = 1
cutoff = 1
bar_color = '#2250d9'
bin_partition = np.arange(0, bin_max + 1, bin_size)
tick_partition = np.arange(0, bin_max + 1, bin_max // (bin_max / (4 * bin_size)))


def main():
    project_path = Path('/Users/leoburgy/Dropbox/buergy_ncomms/data/sfig4/')
    genotype = 'ss4'
    night = 16
    time_points = [0, .25, .5, 8]

    clusters = import_cluster_excel(project_path / f'clusters_{genotype}.xlsx', day_points=[0, .25, .5, 8])

    chloroplasts_examined = derive_chloroplasts_examined(clusters, GROUPER_CHLP, GROUPER_REPLICATES)
    chloroplasts_examined.to_excel(project_path / f'chloroplasts_examined_{genotype}.xlsx')

    granule_number = derive_granule_number(clusters, GROUPER_CHLP)
    pocket_number = derive_pocket_number(clusters, GROUPER_CHLP)
    grouped_category = clusters_categorise(df=clusters, grouper=GROUPER_CLUSTERS, bin_partition=bin_partition)

    pocket_granules_mean_weighted = summary(granule_number, pocket_number, chloroplasts_examined)
    pocket_granules_mean_weighted.to_excel(project_path / f'mean_pocket_granule_number_{genotype}.xlsx')

    # Plot distributions of each quantity
    fig, axs = plt.subplots(nrows=3, ncols=len(time_points), figsize=(5, 3.5), sharey=False)

    a = axs[0, :]
    b = axs[1, :]
    c = axs[2, :]

    plot_histogram(granule_number, 'granule_number', x_label='# granules / chloroplast', ax=a, bar_color=bar_color,
                   bin_partition=bin_partition, title=True)
    plot_histogram(pocket_number, 'pocket_number', x_label='# pockets / chloroplast', ax=b, bar_color=bar_color,
                   bin_partition=bin_partition, title=False)
    category_plot(grouped_category, x_label="# granules in cluster category", ax=c, bar_color=bar_color,
                  bin_size=bin_size, title=False)

    for i, ax in enumerate(axs.ravel()):
        ax.set_xlim(0, bin_max)
        ax.set_xticks(tick_partition)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', width=.5)
        if i % len(time_points) == 0:
            ax.set_ylabel('Frequency')

    fig.tight_layout(h_pad=3)
    fig.savefig(project_path / f'cluster_size_{genotype}_N{night}h_by{bin_size}_cutoff{cutoff}_max{bin_max}.pdf')


if __name__ == '__main__':
    main()
