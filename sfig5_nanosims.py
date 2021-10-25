import pandas as pd
from pathlib import Path
import logging

from matplotlib import cm, colors
from nanosims.contours import *
from matplotlib import pyplot as plt
from nanosims.utils import read_nrrd, extract, calculate_delta, save_image, save_colormap

logging.basicConfig(level=logging.INFO)

font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica',
        'weight': 'normal',
        'size': 6}

plt.rc('font', **font)
plt.rc('pdf', fonttype=42)
plt.rc('lines', linewidth=1)

color_max = 3000
smooth = True
k_size = 3


def main():
    map = pd.read_csv('nanosims_max_enrich.txt',
                      sep='\t', index_col='path_to_im')

    path_project = Path('/Users/leoburgy/Dropbox/buergy_ncomms/data/sfig5')
    nrrd_list = [path for path in path_project.rglob('*/*.nrrd') if not path.name.startswith('.')]

    for i, file in enumerate(nrrd_list):
        color_max = map.loc[file.stem, 'd13c_max']
        header, data = read_nrrd(file)
        logging.info(f"{file.name} ({header['Mims_raster']} nm)")

        cyanide = extract(data, header, mass_name='14N12C', cumulative=True)

        carbide13 = extract(data, header, mass_name='13C12C', cumulative=True)
        carbide12 = extract(data, header, mass_name='12C12C', cumulative=True)
        if smooth:
            carbide13 = cv2.GaussianBlur(carbide13, ksize=(k_size, k_size), sigmaX=0)
            carbide12 = cv2.GaussianBlur(carbide12, ksize=(k_size, k_size), sigmaX=0)

        delta = calculate_delta(numerator=carbide13, denominator=carbide12)
        delta = np.nan_to_num(delta)

        save_image(data=np.fliplr(delta), cm='cividis',
                   cbar_max=color_max,
                   dst=str(file.parent / f'{file.stem}_delta.png'))

        save_image(data=np.fliplr(cyanide), cm='gray',
                   cbar_max=None,
                   dst=str(file.parent / f'{file.stem}_cyanide.png'))

        save_colormap(cbar_max=color_max,
                      dst=str(file.parent / f'{file.stem}_cmap.pdf'))

        # fig, ax = plt.subplots(figsize=(1, 3))
        # fig.subplots_adjust(left=0, bottom=0.05, right=.5, top=.95)
        # ax.tick_params(labelsize=8)
        # cmap = cm.get_cmap('cividis')
        # norm = colors.Normalize(vmin=0, vmax=color_max)
        #
        # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
        #              cax=ax, orientation='vertical')
        # fig.tight_layout()
        # fig.savefig(str(path_project / f'{file.stem}_cmap.pdf'))


if __name__ == '__main__':
    main()
