import pandas as pd

GROUPER_CHLP = 'genotype night light replicate chloroplast'.split(' ')
GROUPER_CLUSTERS = 'genotype night light cluster_size'.split(' ')
GROUPER_REPLICATES = GROUPER_CHLP[:-1]
GROUPER_CONDITIONS = GROUPER_REPLICATES[:-1]


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


def import_cluster_excel(path, day_points):
    clusters = pd.read_excel(path)
    clusters = clusters.fillna(method='ffill')

    clusters[['night', 'hour', 'minute']] = clusters.timepoint.str.extract(r'N(\d+)D(\d\d)(\d\d)')
    clusters['light'] = clusters['hour'].astype(int) + clusters['minute'].astype(int) / 60
    clusters = clusters[['genotype', 'night', 'light', 'replicate', 'chloroplast', 'cluster_size']]
    clusters = clusters.loc[clusters['light'].isin(day_points)]
    return clusters


def derive_granule_number(data, grouper):
    granule_number = data.groupby(grouper)[['cluster_size']].sum()
    granule_number.columns = ['granule_number']
    granule_number = granule_number.reset_index().sort_values(by=['night', 'light'], ascending=True)
    return granule_number


def derive_pocket_number(data, grouper):
    pocket_number = (data
                     .groupby(grouper)[['cluster_size']]
                     .apply(pocket_score))
    pocket_number.columns = ['pocket_number']
    pocket_number = pocket_number.reset_index().sort_values(by=['night', 'light'], ascending=True)
    return pocket_number


def weighted_average(data, to_average_column='granule_number', weight_column='chloroplast'):
    weights = data.groupby(GROUPER_REPLICATES).count()[weight_column]
    data['weight'] = weights
    averaged = data.groupby(GROUPER_CONDITIONS).apply(
        lambda x: np.average(x[to_average_column],
                             weights=x['weight']))
    averaged.name = f'w_avg_{to_average_column}'

    return averaged
