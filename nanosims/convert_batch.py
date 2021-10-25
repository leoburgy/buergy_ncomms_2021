import numpy as np
from math import log10, ceil
import pandas as pd
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import cm, colors

from utils import save_image, calculate_delta, extract, read_nrrd

font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica',
        'weight': 'bold',
        'size': 8}

plt.rc('font', **font)
plt.rc('pdf', fonttype=42)
plt.rc('lines', linewidth=1)

color_max = 3000
smooth = True
k_size = 3

if __name__ == '__main__':

    map = pd.read_csv('/Users/leoburgy/eth/phd/experiments/initiation_mutants/20200901_nanosims_ptst2and3/map.txt', sep='\t', index_col='path_to_im')

    data_dir = Path('/Users/leoburgy/eth/phd/experiments/initiation_mutants/20200901_nanosims_ptst2and3')
    nrrd_dir = data_dir / 'nanosims'
    # ims_list = [nrrd_dir / Path(file).name.replace('_delta.png', '.nrrd') for file in map.path_to_im]
    # ims_list = sorted(file for file in nrrd_dir.iterdir() if file.suffix == '.nrrd')
    #
    ims_list = [nrrd_dir / 'Burgy-Sep-2020_24.nrrd']

    out_dir = data_dir / 'pngs'
    out_dir.mkdir(exist_ok=True)

    print(f'Saving files at {out_dir}')

    for i, file in enumerate(ims_list):
        # print(color_max_list[i], file.name)
        color_max = map.loc[file.stem, 'd13c_max']
        # color_max = 8000
        header, data = read_nrrd(file)
        print(file.name, header['Mims_raster'])
        # continue

        cyanide = extract(data, header, mass_name='14N12C', cumulative=True)

        carbide13 = extract(data, header, mass_name='13C12C', cumulative=True)
        carbide12 = extract(data, header, mass_name='12C12C', cumulative=True)
        if smooth:
            carbide13 = cv2.GaussianBlur(carbide13, ksize=(k_size, k_size), sigmaX=0)
            carbide12 = cv2.GaussianBlur(carbide12, ksize=(k_size, k_size), sigmaX=0)

        delta = calculate_delta(numerator=carbide13, denominator=carbide12)
        delta = np.nan_to_num(delta)
        n = np.quantile(np.sort(delta.flatten()), .99)
        # color_max = 10**int(log10(n))*ceil(n/10**int(log10(n)))

        save_image(data=np.rot90(delta, -1), cm='cividis',
                   cbar_max=color_max,
                   dst=str(out_dir / f'{file.stem}_delta.png'))

        save_image(data=np.rot90(cyanide, -1), cm='gray',
                   cbar_max=None,
                   dst=str(out_dir / f'{file.stem}_cyanide.png'))

        cmap = cm.get_cmap('cividis')
        norm = colors.Normalize(vmin=0, vmax=color_max)

        fig, ax = plt.subplots(figsize=(1, 3))
        fig.subplots_adjust(left=0, bottom=0.05, right=.5, top=.95)
        ax.tick_params(labelsize=16)
        ticks = [0, color_max // 2, color_max]

        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=ax, orientation='vertical')
        cbar.set_ticks(ticks)
        # cbar.set_ticklabels([0] + [str(int(int(i) / 1000)) + 'k' for i in ticks if i >= 1000])
        # cbar.set_ticklabels([0] + [str(i) for i in ticks])
        fig.tight_layout()
        fig.savefig(str(out_dir / f'{file.stem}_cmap.pdf'))
        plt.close()
        print(file.stem)
        print(delta.max())
