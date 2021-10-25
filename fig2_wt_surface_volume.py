from pathlib import Path

import matplotlib
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

project_path = Path('/Users/leoburgy/epfl/buergy_ncomms/data/fig2/E')
genotype = 'WT'
night = 16
barcolor = '#2250d9'


def main():
    starch = pd.read_excel(project_path / 'surface_volumes.xlsx')

    fig, axs = plt.subplots(nrows=3, figsize=(1.8, 3.2))

    for i, gen in enumerate(starch.genotype.unique()):
        sub = starch.loc[(starch['genotype'] == gen)]

        axs[0].plot(sub['day'].astype(str), (sub['volume_tot_um3']),
                    color=barcolor, alpha=.5, marker='.', lw=0., mec='None')
        axs[0].set_ylabel('Volume $[µm^3]$')
        axs[0].set_ylim(0, 30)

        axs[1].plot(sub['day'].astype(str), (sub['surface_tot_um2']),
                    color=barcolor, alpha=.5, marker='.', lw=0., mec='None')
        axs[1].set_ylabel('Surface area $[µm^{2}]$')
        axs[1].set_ylim(0, 150)

        axs[2].plot(sub['day'].astype(str), (sub['surface_volume_ratio']),
                    color=barcolor, alpha=.5, marker='.', lw=0., mec='None')
        axs[2].set_ylabel('SA:V $[µm^{-1}]$')
        axs[2].set_ylim(0, 50)

    for ax in axs:
        ax.tick_params(axis='both', width=.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticklabels(['+ 15 min', '+ 30 min', '+ 8 h'])

    fig.text(.5, .05, 'Day', horizontalalignment='center')
    fig.tight_layout(rect=(.05, .05, 1, 1))

    fig.savefig(project_path / 'surface_volume.pdf')

    chloroplasts_examined = starch.groupby(['genotype', 'night', 'day']).count()

    chloroplasts_examined.to_excel(project_path / 'chloroplasts_segmented.xlsx')


if __name__ == '__main__':
    main()
