# Copyright 2021 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import numpy as np
from skimage import io

import matplotlib.pyplot as plt
from mpl_point_clicker import clicker


def open_image(path,
               pw_nm,
               patch_size_um,
               patch_dims,
               patch_offset_um,
               idx=0,
               trunc=1,
               num_ramps=1,
               normalise=True):
    """
    Method to access image and let user pick patches for assessment

    Args:
    path (str) :: Path to raw image
    pw_nm (float) :: Pixel width in um
    patch_size_um (2-tuple) :: Dimensions of patches in um
    patch_dims (2-tuple) :: Number of patches in each direction for each ramp
    patch_offset_um (float) :: Separation of patches in um
    idx (int) :: Index of image to be assessed (if image is in a stack)
    trunc (int) :: Number of pixels to be truncated from image bottom (due to information bar)
    num_ramps (int) :: Number of ramps in image
    normalise (bool) :: Whether to normalise image

    Returns:
    ndarray*4
    """
    img = io.imread(path)
    if img.ndim == 2:
        img = img[np.newaxis, :, :-trunc]

    patch_size_px = np.array(patch_size_um)*1000 // np.array(pw_nm)
    patch_size_half = patch_size_px // 2

    # Collect user clicks
    fig, ax = plt.subplots(figsize=(15, 9))
    plt.title(f"{path}: {num_ramps} ramps")
    ax.imshow(img[idx, :-trunc], cmap="gist_gray")

    clicks = np.array(plt.ginput(num_ramps), dtype=int)
    clicks = clicks[:, ::-1]

    # Convert sample offset to pixels and create mesh
    patch_offset_px = np.array(patch_offset_um)*1000 // np.array(pw_nm)
    offsets_x_px = np.array(np.arange(-0.5*(patch_dims[0]-1),
                                      0.5*(patch_dims[0]+1)*patch_offset_px,
                                      dtype=int))
    offsets_y_px = np.array(np.arange(-0.5*(patch_dims[1]-1),
                                      0.5*(patch_dims[1]+1)*patch_offset_px,
                                      dtype=int))

    cntrs, uls, lrs = [], [], []
    for _, click in enumerate(clicks):
        for _, osx in enumerate(offsets_x_px):
            for _, osy in enumerate(offsets_y_px):
                cntrs.append(click + np.array([osx, osy]))
                uls.append(click + np.array([osx, osy]) - patch_size_half)
                lrs.append(click + np.array([osx, osy]) + patch_size_half)

    # Normalisation of image
    if normalise:
        midrange = [np.percentile(img, 10), np.percentile(img, 90)]
        img_cec = (img-midrange[0]) / (midrange[1]-midrange[0])
        img_cec[img_cec<0] = 0
        img_cec[img_cec>1] = 1

    # Close image
    plt.close(fig)

    return img_cec, np.array(cntrs, dtype=int), np.array(uls, dtype=int), np.array(lrs, dtype=int)
