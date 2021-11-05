import pandas as pd

GROUPER_CHLP = 'genotype night light replicate chloroplast'.split(' ')
GROUPER_CLUSTERS = 'genotype night light cluster_size'.split(' ')


def pocket_score(x):
    if (len(x) == 1) & (x['cluster_size'] == 0).all():
        return x.sum()
    else:
        return x.count()


def clusters_categorise(df, grouper, bin_partition):
    _df = df.copy()
    _df['granules_in_category'] = _df['cluster_size']
    grouped = _df.groupby(grouper).sum()[
        ['granules_in_category']].reset_index()
    grouped['binned'] = pd.cut(grouped['cluster_size'], bin_partition, labels=bin_partition[:-1], right=False)
    return grouped


def pretty_time(time):
    if time < 1:
        return f"+ {int(60 * time)} min"
    else:
        return f"+ {int(time)} h"
