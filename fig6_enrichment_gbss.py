from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica Neue',
        'weight': 'light',
        'size': 6}

plt.rc('font', **font)
plt.rc('pdf', fonttype=42)
plt.rc('lines', linewidth=8)

path_project = Path('/Users/leoburgy/Dropbox/buergy_ncomms/data/fig6/C')


def pretty_boxplot(data, credible_interval=(2.5, 97.5), ax=None, line_width=.5, **kwargs):
    """
    Plot the data using boxplot.
    :param data: a list of lists of data to plot. The keys are extracted for the labels.
    :param ax: the ax to accept the data
    :param credible_interval: The extent of the whiskers, useful for posterior boxplot.
    :return: the ax containing the boxplot
    """
    ax.boxplot(data,
               showfliers=False,
               widths=.5,
               showcaps=False,
               vert=True,
               whis=credible_interval,
               whiskerprops={'linewidth': line_width,
                             },
               medianprops={'color': 'black',
                            'linewidth': line_width},
               boxprops={'linewidth': line_width, 'color': 'black'},
               **kwargs
               )
    return ax


def read_rois(filename):
    if not Path(filename).exists():
        raise FileNotFoundError(f"{filename} does not exist.")

    rois = pd.read_csv(filename)
    col_subset = {'condition', 'roi', 'd13C'}

    if not col_subset.issubset(rois.columns):
        raise Exception(f"Not all columns ({set(rois.columns) - col_subset}) are present in {rois.columns}")

    rois = rois[col_subset]
    rois[['genotype', 'is_labelled', 'replicate']] = rois.condition.str.extract(r'(\w+)-T(\d)(\w)')
    rois = rois[['genotype', 'replicate', 'is_labelled', 'roi', 'd13C']]
    rois.is_labelled = rois.is_labelled.astype(bool)

    rois = rois.loc[rois.d13C > 0]

    rois = rois.loc[rois.is_labelled]
    rois = rois.loc[rois.roi != 'whole']

    rois = rois.sort_values(['genotype', 'roi'], ascending=False)
    rois['gen_roi'] = rois[['roi', 'genotype']].agg(' '.join, axis=1)

    rois = rois.reset_index(drop=True)

    return rois


def main():
    # ACTUAL DATA
    rois = read_rois(path_project / 'rois_wt_gbss.csv')
    rois = rois.replace('margin', 'surface', regex=True)
    rois = rois.set_index(['roi', 'genotype'])
    rois[['gen_roi', 'replicate', 'd13C']].to_excel(path_project / 'roi_wt_gbss_d13c.xlsx', index=False)
    rois_unique = rois.index.get_level_values('roi').unique()
    genotypes_unique = rois.index.get_level_values('genotype').unique()

    # PREDICTED DATA
    post_sample = pd.read_csv(path_project / 'gamma_reg_posterior.csv')
    preds = post_sample.filter(axis=1, like='pred').T
    preds.index = pd.factorize(rois.gen_roi)[1].str.split(expand=True)
    preds.index.names = ['roi', 'genotype']
    preds = preds.sort_index()

    fig, axs = plt.subplots(ncols=len(rois_unique), nrows=1, figsize=(3.5, 1.8))

    for i, roi in enumerate(rois_unique):
        ax = axs[i]
        for j, gen in enumerate(genotypes_unique):
            y_range = [0, 8] if roi == 'surface' else [0, 1]

            if gen == 'GBSS':
                gen_label = r'$gbss$'
            else:
                gen_label = 'WT'

            pred_sub = preds.loc[(roi, gen)]
            data_sub = rois.loc[(roi, gen)]

            pretty_boxplot(data_sub['d13C'] / 1000, ax=ax, positions=[j], labels=[gen_label])

            ax.plot(data_sub.index.get_level_values('genotype'),
                    data_sub.loc[:, 'd13C'] / 1000,
                    color='blue', alpha=.1, marker='_', ls='none')

            ax.set_title(roi.capitalize())
            ax.set_ylim(y_range)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-1, 2)

    fig.text(0.01, 0.5, 'Enrichment in $^{13}C$ (×1000)', va='center', rotation=90)

    fig.tight_layout(rect=(.05, .0, 1, 1))
    fig.savefig(path_project / 'gbss_enrichment_by_roi_posterior.pdf')


if __name__ == '__main__':
    main()
