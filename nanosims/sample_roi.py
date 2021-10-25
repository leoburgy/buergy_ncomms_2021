from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import save_image, save_colormap, calculate_delta, extract, read_nrrd

proj_path = Path('/Users/leoburgy/phd/experiments/granule_development/20141101_nanosims_gbss/')
data_path = proj_path
ims = data_path / 'ims'
masks = data_path / 'masks'
output_path = data_path / 'out'
output_path.mkdir(exist_ok=True)

roi_types = ('background',
             'granule',
             'margin',
             'core')

color_max = 30000

for filename in {'Kopp-Nov-2014_20', 'Kopp-Nov-2014_48', 'Kopp-Nov-2014_101'}:
    print(filename)
    ims_filename = ims / f'{filename}.nrrd'
    if ims_filename.exists():
        header, data = read_nrrd(str(ims_filename))
        mask_filename_granule = masks / f'{filename}_delta13C_class_Granule.png'
        mask_filename_core = masks / f'{filename}_delta13C_class_Core.png'
        
        mask_granule = cv2.imread(str(mask_filename_granule), 0) / 255.
        mask_core = cv2.imread(str(mask_filename_core), 0) / 255.
        
        mask_margin = mask_granule - mask_core
        mask_margin[mask_margin < 0] = 1  # necessary when regions are patched e.g. non analysed granules
        plt.imshow(mask_margin)
        
        mask_background = 1 - mask_granule
        mask_not_margin = 1 - mask_margin
        
        rois = {'granule': mask_granule,
                'margin': mask_margin,
                'core': mask_core,
                'background': mask_background}
    
    c13 = extract(data=data, header=header, mass_name='13C12C')
    c12 = extract(data=data, header=header, mass_name='12C2')
    ratio = c13 / c12
    ratio = cv2.GaussianBlur(ratio, ksize=(3, 3), sigmaX=0)
    delta = calculate_delta(c13, c12, cleaned=False)
    delta = np.nan_to_num(delta)
    
    roi_core = delta * (mask_core)
    roi_not_margin = delta * (mask_not_margin)
    roi_margin = delta * (mask_margin)
    roi_background = delta * (mask_background)
    
    save_image(data=delta, cm='cividis',
               cbar_max=color_max,
               dst=str(output_path / f'{filename}_delta13C.png'))
    
    save_colormap(data=delta,
                  dst=str(output_path / f'{filename}_delta13C_cm.pdf'),
                  cm='cividis', cbar_max=color_max)
    
    save_image(data=roi_core, cm='cividis',
               cbar_max=1000,
               dst=str(output_path / f'{filename}_roi_not_margin.png'))
    
    save_colormap(data=roi_core,
                  dst=str(output_path / f'{filename}_roi_not_margin_cm.pdf'),
                  cm='cividis', cbar_max=1000)
