import sys
import numpy as np
import cv2


def apply_mask(img, mask):
    """
    Keep nrrd values that are in the mask
    takes: an nrrd and a mask
    return: a masked ndarray
    """
    if len(img.shape) < 2:
        print('img is not an array')
        sys.exit()
    if len(mask.shape) < 2:
        print('mask is not an array')
        sys.exit()
    h_i, w_i, masses, layers = img.shape
    h_m, w_m = mask.shape
    assert (h_i, w_i) == (h_m, w_m)
    hyper_mask = np.zeros((h_m, w_m, masses, layers), dtype=np.float64)
    array_masked = img * hyper_mask
    return array_masked


def retrieve_contours(mask, c=-1):
    """
    Find contours in a binary images and save them in a list.
    takes: a mask
    return a list of contours (c is an index to select the cth contour)
    """

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    if c > -1:
        contours = contours[c]

    return contours


def find_center(contour):
    """
    Compute the barycentre of a contour.
    """
    moments = cv2.moments(contour)
    c_x: int = int(moments["m10"] / moments["m00"])
    c_y = int(moments["m01"] / moments["m00"])
    return c_x, c_y


def root_profile(contour, log=None, dev=False):
    """
    Untangle a profile into a list where the first element
    is the nearest point to the barycentre
    """
    center = find_center(contour)
    dists = [np.linalg.norm(p - center) for p in contour]
    start_index = dists.index(min(dists))
    start_point = contour[start_index][0]
    start_x, start_y = start_point
    contour = np.vstack((contour[start_index:], contour[:start_index]))
    if dev and log:
        cv2.circle(log, (start_x, start_y), 1, (0, 255, 0), -1)
        cv2.circle(log, center, 1, (255, 255, 255), -1)
        return log, contour
    return contour


def extract_profile(data, contour, p=2):
    """
    Sample enrichment along a contour given a isotopic map.
    
    data : np.array
    contour : np.array
    p : int, optional kernel width
    
    return: a list of enrichment on the unit circle.
    """
    
    dims = data.shape
    if len(dims) > 2:
        data = data.sum(axis=-1)
        
    points_n = len(contour)
    rooted_contour = root_profile(contour)
    
    profile = np.zeros(points_n, dtype=np.float64)
    
    for i, edges in enumerate(rooted_contour):
        for point in edges:
            y, x = point
            patch = data[x-p:x+p, y-p:y+p]

            profile[i] = np.mean(patch)

    return profile


def normalise_profile(profile):
    """
    Map a profile onto the unit circle.
    takes: a list of enrichment
    return: a list
    """
    points_n = len(profile)
    mapper = 2 * np.pi / points_n
    return mapper * np.linspace(0, points_n, points_n)


def read_mask(path_mask):
    """
    Read in mask file in a [0, 1] array.
    """

    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) / 255.
    return mask


def extrude_mask(mask1, mask2):
    """
    Compute the difference of two concentric masks.
    """

    mask1 = read_mask(mask1)
    mask2 = read_mask(mask2)

    mask_core = mask1 * mask2
    return mask_core
