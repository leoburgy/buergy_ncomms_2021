from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica Neue',
        'weight': 'light',
        'size': 6}

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

grouper = 'genotype night light replicate chloroplast'.split(' ')


def pocket_score(x):
    if (len(x) == 1) & (x['cluster_size'] == 0).all():
        return x.sum()
    else:
        return x.count()


def pretty_time(time):
    if time < 1:
        return f"+ {int(60 * time)} min"
    else:
        return f"+ {int(time)} h"


test_df = pd.DataFrame({'genotype': ['wt', 'wt', 'wt', 'wt', 'wt', 'wt'],
                        'night': [16, 16, 16, 16, 16, 16],
                        'light': [0, 0, 0, 0, 8, 8],
                        'replicate': [1, 1, 1, 2, 1, 1],
                        'stack': [1, 1, 1, 1, 2, 2],
                        'chloroplast': [1, 2, 2, 1, 1, 1],
                        'cluster_size': [0, 8, 8, 2, 3, 1]})

test_granule_number = test_df.groupby(grouper).sum()['cluster_size']
assert list(test_granule_number) == [0, 16, 2, 4]

test_pocket_number = test_df.groupby(grouper).apply(pocket_score)['cluster_size']
assert list(test_pocket_number) == [0, 2, 1, 2]

test_df['granules_in_category'] = test_df['cluster_size']
test_grouped = test_df.groupby('genotype night light cluster_size'.split(' ')).sum()[
    ['granules_in_category']].reset_index()
test_grouped['binned'] = pd.cut(test_grouped['cluster_size'], bin_partition, labels=bin_partition[:-1], right=False)

assert list(test_grouped['binned']) == [0, 2, 8, 1, 3]
assert list(test_grouped['granules_in_category']) == [0, 2, 16, 1, 3]

test_sub = test_grouped.loc[test_grouped['light'] == 8]
plt.bar(test_sub['binned'], test_sub['granules_in_category'] / test_sub['granules_in_category'].sum() / binsize,
        color=barcolor, align='center', width=1)
plt.xticks(bin_partition)
plt.xlim(0, bin_max)
plt.savefig('/Users/leoburgy/Desktop/test.pdf')

test_grouped = test_grouped.loc[test_grouped['cluster_size'] >= cutoff]
# test_df_bins = test_grouped.groupby('genotype night light binned'.split(' ')).sum()[
#     'granules_in_category'].reset_index()

if __name__ == '__main__':

    clusters = pd.read_excel(project_path / 'clusters_ss4.xlsx')
    clusters = clusters.fillna(method='ffill')

    clusters[['night', 'hour', 'minute']] = clusters.timepoint.str.extract(r'N(\d+)D(\d\d)(\d\d)')
    clusters['light'] = clusters['hour'].astype(int) + clusters['minute'].astype(int) / 60
    clusters = clusters[['genotype', 'night', 'light', 'replicate', 'chloroplast', 'cluster_size']]
    clusters = clusters.loc[clusters['light'].isin(time_points)]

    # Derive the granule number
    granule_number = clusters.groupby(grouper)[['cluster_size']].sum()
    granule_number.columns = ['granule_number']
    granule_number = granule_number.reset_index().sort_values(by=['night', 'light'], ascending=True)
    granule_number.to_excel(project_path / 'granule_number.xlsx')

    mean_granule_number = granule_number.groupby('genotype night light replicate'.split(' ')).mean()[['granule_number']]
    n_chloroplasts_weights = granule_number.groupby('genotype night light replicate'.split(' ')).count()['chloroplast']
    mean_granule_number['weight'] = n_chloroplasts_weights
    mean_granule_number = mean_granule_number.groupby('genotype night light'.split(' ')).apply(
        lambda x: np.average(x['granule_number'],
                             weights=x['weight']))
    mean_granule_number.name = 'w_avg_granule_number'
    mean_granule_number.to_excel(project_path / 'mean_granule_number.xlsx')

    # Derive the pocket number
    pocket_number = (clusters
                     .groupby(grouper)[['cluster_size']]
                     .apply(pocket_score))
    pocket_number.columns = ['pocket_number']
    pocket_number = pocket_number.reset_index().sort_values(by=['night', 'light'], ascending=True)
    pocket_number.to_excel(project_path / 'pocket_number.xlsx')
    pocket_number.groupby('genotype night light replicate'.split(' ')).mean().to_excel(
        project_path / 'mean_pocket_number.xlsx')

    mean_pocket_number = pocket_number.groupby('genotype night light replicate'.split(' ')).mean()[['pocket_number']]
    n_chloroplasts_weights = granule_number.groupby('genotype night light replicate'.split(' ')).count()['chloroplast']
    mean_pocket_number['weight'] = n_chloroplasts_weights
    mean_pocket_number = mean_pocket_number.groupby('genotype night light'.split(' ')).apply(
        lambda x: np.average(x['pocket_number'],
                             weights=x['weight']))
    mean_pocket_number.name = 'w_avg_pocket_number'

    pocket_granules_mean_weighted = pd.concat([mean_granule_number, mean_pocket_number], axis=1)
    pocket_granules_mean_weighted.to_excel(project_path / f'mean_pocket_granule_number_{genotype}.xlsx')

    n_chloroplasts_weights.to_excel(project_path / f'chloroplasts_examined_{genotype}.xlsx')

    # Refactor weighting of cluster sizes
    clusters['granules_in_category'] = clusters['cluster_size']
    grouped = clusters.groupby('genotype night light cluster_size'.split(' ')).sum()[['granules_in_category']]
    grouped = grouped.reset_index()
    grouped = grouped.loc[grouped['cluster_size'] >= cutoff]
    grouped['binned'] = pd.cut(grouped['cluster_size'], bin_partition, labels=bin_partition[1:], right=False,
                               include_lowest=True)

    df_bins = grouped.groupby('genotype night light binned'.split(' ')).sum()['granules_in_category'].reset_index()

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
    for i, t in enumerate(sorted(df_bins.light.unique())):
        sub = grouped.loc[grouped['light'] == t]

        c[i].bar(sub['cluster_size'],
                 sub['granules_in_category'] / sub['granules_in_category'].sum() / binsize,  # normalise
                 color=barcolor,
                 align='center',
                 width=1)  # negative to align on the right edge

        if i == 1:
            c[i].set_xlabel("# granules in cluster category")

    for i, ax in enumerate(axs.ravel()):
        ax.set_xlim(0, bin_max)
        ax.set_xticks(tick_partition)

        # axs.set_ylim(0, .6)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', width=.5)
        if i % len(time_points) == 0:
            ax.set_ylabel('Frequency')

    fig.tight_layout(h_pad=3)
    fig.savefig(project_path / f'cluster_size_{genotype}_N{night}h_by{binsize}_cutoff{cutoff}_max{bin_max}.pdf')
