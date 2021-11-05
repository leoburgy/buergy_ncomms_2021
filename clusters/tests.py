import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from clusters import GROUPER_CHLP, GROUPER_CLUSTERS
from clusters import pocket_score, clusters_categorise

# Define parameters for histograms
bin_max = 16
binsize = 1
cutoff = 1
barcolor = '#2250d9'
bin_partition = np.arange(0, bin_max + 1, binsize)
tick_partition = np.arange(0, bin_max + 1, bin_max // (bin_max / (4 * binsize)))


test_df = pd.DataFrame({'genotype': ['wt', 'wt', 'wt', 'wt', 'wt', 'wt'],
                        'night': [16, 16, 16, 16, 16, 16],
                        'light': [0, 0, 0, 0, 8, 8],
                        'replicate': [1, 1, 1, 2, 1, 1],
                        'stack': [1, 1, 1, 1, 2, 2],
                        'chloroplast': [1, 2, 2, 1, 1, 1],
                        'cluster_size': [0, 8, 8, 2, 3, 1]})

test_granule_number = test_df.groupby(GROUPER_CHLP).sum()['cluster_size']
assert list(test_granule_number) == [0, 16, 2, 4]

test_pocket_number = test_df.groupby(GROUPER_CHLP).apply(pocket_score)['cluster_size']
assert list(test_pocket_number) == [0, 2, 1, 2]

test_grouped = clusters_categorise(test_df, grouper=GROUPER_CLUSTERS, bin_partition=bin_partition)

assert list(test_grouped['binned']) == [0, 2, 8, 1, 3]
assert list(test_grouped['granules_in_category']) == [0, 2, 16, 1, 3]

test_sub = test_grouped.loc[test_grouped['light'] == 8]
plt.bar(test_sub['binned'], test_sub['granules_in_category'] / test_sub['granules_in_category'].sum() / binsize,
        color=barcolor, align='center', width=1)
plt.xticks(bin_partition)
plt.xlim(0, bin_max)
plt.savefig('/Users/leoburgy/Desktop/test.pdf')