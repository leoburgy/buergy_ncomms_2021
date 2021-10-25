import sys
from pathlib import Path
from datetime import datetime

from matplotlib import cm, colors
from matplotlib import pyplot as plt
import nrrd


def read_nrrd(file):
    """
    Read a NRRD file.
    """
    
    filepath = Path(file)
    
    if filepath.suffix != '.nrrd':
        print("There is no such file.")
        sys.exit()
    
    data, header = nrrd.read(str(filepath))
    
    return header, data


def get_timestamp(date, time):
    timestamp_raw = date + ' ' + time
    formatted_time = datetime.strptime(timestamp_raw, '%d.%m.%y %H:%M')
    timestamp = datetime.strftime(formatted_time, '%Y-%m-%d %H:%M:%S')
    return timestamp


def _mass_index(mass_name, mass_names):
    """
    Correct mass name spelling.
    return: index of mass name
    """
    synonyms = {'14N12C': '12C14N',
                '13C12C': '12C13C',
                '12C12C': '12C2'
                }
    
    if mass_name not in mass_names:
        return mass_names.index(synonyms[mass_name])
    else:
        return mass_names.index(mass_name)


def extract(data, header, mass_name='14N12C', cumulative=True):
    """
    Extract the species from an ndarray.
    cumulative bool: whether to cumulate the planes.
    Return an 2-array.
    """
    mass_names = header['Mims_mass_symbols'].split(' ')
    
    try:
        idx = _mass_index(mass_name, mass_names)
        if cumulative:
            return data[:, :, :, idx].sum(axis=2)
        else:
            return data[:, :, :, idx]
    except LookupError:
        print(f'{mass_name} could not be found; choose from {" / ".join(mass_names)}')


def _impute(data):
    """
    Replace 0's with 1 in count data
    """
    data[data == 0] = 1
    return data


def calculate_delta(numerator, denominator, natural_ratio=.021, cleaned=False):
    """
    Compute the enrichment.
    """
    
    ratio = numerator / denominator
    delta = 1000 * ((ratio / natural_ratio) - 1)
    
    if cleaned:
        delta[delta < 0 * natural_ratio] = 0
    
    return delta


def save_image(dst, data, cm='cividis', cbar_max=8000):
    """
    Set a figure and write the data to the destination file.
    """
    sizes = data.shape
    
    height = float(sizes[0])
    width = float(sizes[1])
    
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(data, cmap=cm, vmin=0, vmax=cbar_max)
    plt.savefig(dst, dpi=height)
    plt.close()


def save_colormap(dst, cbar_max=8000):
    """
    Set a figure and write the data to the destination file.
    """
    # plt.imshow(data, cmap=cm, vmin=0, vmax=cbar_max)
    #
    # side, colormap = plt.subplots(1, 1, figsize=(1, 3))
    # side.subplots_adjust(left=0, bottom=0.05, right=.5, top=.95)
    # colormap.tick_params(labelsize=8)
    #
    # side.savefig(dst)
    #
    # plt.close()
    fig, ax = plt.subplots(figsize=(1, 3))
    fig.subplots_adjust(left=0, bottom=0.05, right=.5, top=.95)
    ax.tick_params(labelsize=8)
    cmap = cm.get_cmap('cividis')
    norm = colors.Normalize(vmin=0, vmax=cbar_max)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='vertical')
    fig.tight_layout()
    fig.savefig(dst)
