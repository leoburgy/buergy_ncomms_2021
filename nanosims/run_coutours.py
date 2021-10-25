from pathlib import Path
from math import log10
import numpy as np
from matplotlib import pyplot as plt

from contours import *
from utils import read_nrrd, extract, calculate_delta, save_image

if __name__ == "__main__":
    font = {'family': 'sans-serif',
            'sans-serif': 'Helvetica',
            'weight': 'bold',
            'size': 8}

    plt.rc('font', **font)
    plt.rc('pdf', fonttype=42)
    plt.rc('lines', linewidth=1)

    project_dir = Path('/Users/leoburgy/eth/phd/experiments/granule_development/20141101_nanosims_gbss')
    # 22 35 60
    # im_name = 'Jenny-Nov-2014_060'
    im_name = 'Kopp-Nov-2014_134'

    path_nrrd = project_dir / f'data/nanosims/{im_name}.nrrd'
    path_mask = project_dir / f'data/masks/{im_name}_delta_class_Granule.png'
    path_out = project_dir / 'out'

    path_out.mkdir(exist_ok=True)

    if not path_nrrd.exists():
        print('nrrd not found')
        sys.exit(-1)

    if not path_mask.exists():
        print('mask not found')
        sys.exit(-1)

    header, data = read_nrrd(path_nrrd)

    c13 = extract(data, header, mass_name='13C12C', cumulative=True)
    c12 = extract(data, header, mass_name='12C12C', cumulative=True)
    delta = calculate_delta(c13, c12, cleaned=True)
    delta = np.fliplr(delta)
    dims = delta.shape

    h = dims[0]
    w = dims[1]
    if len(dims) > 2:
        n_layers = dims[2]

    save_image(f'{path_out}/{im_name}_delta.png', delta)
    mask = cv2.imread(str(path_mask), 0)
    print(mask.shape)
    contours = retrieve_contours(mask)

    p = 3
    profiles = []
    for i, contour in enumerate(contours):
        profiles.append(extract_profile(delta, contour, p=p))

    log = np.zeros((h, w, 3))

    for i, contour in enumerate(contours):
        centre = find_center(contour)

        rooted_contour = root_profile(contour)
        start = (tuple(rooted_contour[0][0]))

        for j, edges in enumerate(rooted_contour):
            for point in edges:
                y, x = point
                log[x - p:x + p, y - p:y + p] = (0, 0, 255)

        cv2.circle(log, start, 5, (0, 255, 255), -1)

        cv2.putText(log, str(i + 1), centre, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (127, 127, 127), 2, cv2.LINE_AA)

    cv2.imwrite(str(project_dir / f'out/{Path(path_nrrd).stem}_log.png'), log)
    print(project_dir / f'out/{Path(path_nrrd).stem}_log.png')

    profiles_n = len(profiles)
    fig, axs = plt.subplots(nrows=6, figsize=(3.5, 3.5))
    n_profiles = len(profiles)

    for i, profile in enumerate(profiles):
        rev_i = n_profiles - i - 1
        ax = axs.ravel()[rev_i]
        len_rad = normalise_profile(profile)

        ax.plot(len_rad, profile, linewidth=.4, color='red')

        ax.text(2 * np.pi, 1.1 * profile.mean(), str(i + 1))
        ax.axhline(profile.mean(), xmin=0, xmax=2 * np.pi, color='black', lw=.1)
        profile_max = profile.max()
        y_tick_max = int(profile_max / 10 ** int(log10(profile_max)) + 1) * 10 ** int(log10(profile_max))
        print(profile_max, y_tick_max)
        ax.set_xlim((0, 2 * np.pi))
        ax.set_ylim(0, y_tick_max)
        ax.set_xticks([])
        ax.set_yticks([0, y_tick_max])
        ax.tick_params(bottom=False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    axs[profiles_n - 1].tick_params(bottom=True)
    axs[profiles_n - 1].set_xticks([np.round(i * 2 * np.pi, 2) for i in [0, .25, .5, .75, 1]])
    axs[profiles_n - 1].set_xticklabels(['0', 'π/2', 'π', '3/2π', '2π'])

    fig.tight_layout()
    fig.savefig(project_dir / f'out/{im_name}_profiles.pdf')
