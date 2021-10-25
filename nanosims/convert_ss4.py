import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm, colors

font = {'family' : 'sans-serif',
        'sans-serif':'Helvetica',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)
plt.rc('pdf', fonttype=42)
plt.rc('lines', linewidth=1)

from utils import save_image, save_colormap, calculate_delta, extract, read_nrrd

names = ['Burgy-April-2019_' + im_id + '.nrrd' for im_id in ['026', '027']]
color_max_list = [16000, 4000]

data_dir = Path('/Users/leoburgy/phd/data/nanosims/')
nrrd_dir = data_dir / 'nrrd'
ims_list = [nrrd_dir / ims for ims in names]
out_dir = data_dir / 'pngs'
out_dir.mkdir(exist_ok=True)

print(f'Saving files at {out_dir}')

for i, file in enumerate(ims_list):
    print(color_max_list[i], file.name)
    color_max = color_max_list[i]
    header, data = read_nrrd(file)
    
    cyanide = extract(data, header, mass_name='14N12C', cumulative=True)
    
    carbide13 = extract(data, header, mass_name='13C12C', cumulative=True)
    carbide13 = cv2.GaussianBlur(carbide13, ksize=(3, 3), sigmaX=0)
    carbide12 = extract(data, header, mass_name='12C12C', cumulative=True)
    carbide12 = cv2.GaussianBlur(carbide12, ksize=(3, 3), sigmaX=0)
    
    delta = calculate_delta(numerator=carbide13, denominator=carbide12)
    delta = np.nan_to_num(delta)
    
    save_image(data=np.fliplr(delta), cm='cividis',
               cbar_max=color_max,
               dst=str(out_dir / f'{file.stem}_delta.png'))

    save_image(data=np.fliplr(cyanide), cm='gray',
               cbar_max=None,
               dst=str(out_dir / f'{file.stem}_cyanide.png'))

    fig, ax = plt.subplots(figsize=(1, 3))
    fig.subplots_adjust(left=0, bottom=0.05, right=.5, top=.95)
    ax.tick_params(labelsize=8)
    cmap = cm.get_cmap('cividis')
    norm = colors.Normalize(vmin=0, vmax=color_max)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='vertical')
    fig.tight_layout()
    fig.savefig(str(out_dir / f'{file.stem}_cmap.pdf'))
