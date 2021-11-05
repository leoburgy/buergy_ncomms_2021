import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from clusters import (GROUPER_CHLP,
                      GROUPER_CLUSTERS,
                      GROUPER_REPLICATES,
                      derive_pocket_number,
                      derive_chloroplasts_examined,
                      derive_cluster_mean,
                      pocket_score,
                      clusters_categorise,
                      summary,
                      derive_granule_number, GROUPER_CONDITIONS)

# Define parameters for histograms
cutoff = 1
barcolor = '#2250d9'
bin_max = 16

test_df = pd.DataFrame({'genotype': ['wt', 'wt', 'wt', 'wt', 'wt', 'wt'],
                        'night': [16, 16, 16, 16, 16, 16],
                        'light': [0, 0, 0, 0, 8, 8],
                        'replicate': [1, 1, 1, 2, 1, 1],
                        'stack': [1, 1, 1, 1, 2, 2],
                        'chloroplast': [1, 2, 2, 1, 1, 1],
                        'cluster_size': [0, 8, 8, 2, 3, 1]})


def test_derive_chloroplast_number():
    res = derive_chloroplasts_examined(data=test_df, grouper_chlp=GROUPER_CHLP,
                                       grouper_replicates=GROUPER_REPLICATES)
    assert list(res) == [2, 1, 1]


def test_derive_cluster_mean():
    cluster_mean = derive_cluster_mean(test_df, GROUPER_CONDITIONS)
    assert list(cluster_mean) == [np.mean([0, 8, 8, 2]), np.mean([3, 1])]


def test_summary():
    chloroplasts_examined = derive_chloroplasts_examined(test_df, GROUPER_CHLP, GROUPER_REPLICATES)
    granule_number = derive_granule_number(test_df, GROUPER_CHLP)
    pocket_number = derive_pocket_number(test_df, GROUPER_CHLP)
    res = summary(granule_number, pocket_number, chloroplasts_examined)

    assert list(chloroplasts_examined == [2, 1, 1])
    assert list(res['w_avg_granule_number'] == [6., 4.])
    assert list(res['w_avg_pocket_number'] == [1., 2.])


def test_granule_number():
    granule_number = test_df.groupby(GROUPER_CHLP).sum()['cluster_size']
    assert list(granule_number) == [0, 16, 2, 4]


def test_pocket_number():
    pocket_number = test_df.groupby(GROUPER_CHLP).apply(pocket_score)['cluster_size']
    assert list(pocket_number) == [0, 2, 1, 2]


def test_clusters_categorise_bin_size_1():
    bin_size = 1
    bin_partition = np.arange(0, bin_max + 1, bin_size)
    test_grouped = clusters_categorise(test_df, grouper=GROUPER_CLUSTERS, bin_partition=bin_partition)
    assert list(test_grouped['granules_in_category']) == [0, 0, 2, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
                                                          0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    test_sub = test_grouped.loc[test_grouped['light'] == 0]
    plt.bar(test_sub['binned'], test_sub['granules_in_category'] / test_sub['granules_in_category'].sum() / bin_size,
            color=barcolor, align='edge', width=.8 * bin_size)
    plt.xticks(bin_partition)
    plt.xlim(0, bin_max)
    plt.savefig(f'/Users/leoburgy/Desktop/test_{bin_size}.pdf')


def test_clusters_categorise_bin_size_4():
    bin_size = 4
    bin_partition = np.arange(0, bin_max + 1, bin_size)
    test_grouped = clusters_categorise(test_df, grouper=GROUPER_CLUSTERS, bin_partition=bin_partition)
    assert list(test_grouped['granules_in_category']) == [2, 0, 16, 0, 4, 0, 0, 0]

    test_sub = test_grouped.loc[test_grouped['light'] == 0]
    plt.bar(test_sub['binned'], test_sub['granules_in_category'] / test_sub['granules_in_category'].sum() / bin_size,
            color=barcolor, align='edge', width=.8 * bin_size)
    plt.xticks(bin_partition)
    plt.xlim(0, bin_max)
    plt.savefig(f'/Users/leoburgy/Desktop/test_{bin_size}.pdf')
