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
        img = img[np.newaxis, ...]

    patch_size_px = np.array(patch_size_um)*1000 // np.array(pw_nm)

    # Collect user clicks
    fig, ax = plt.subplots(figsize=(15, 9))
    plt.title(f"{path}: {num_ramps} ramps")
    ax.imshow(img[idx, :-trunc], cmap="gist_gray")

    clicks = np.array(plt.ginput(num_ramps*2), dtype=int)
    clicks = clicks[:, ::-1]
    click_uls, click_lrs = clicks[::2], clicks[1::2]
    ramp_size = np.array([click_lrs[i]-click_uls[i] for i in range(len(click_uls))])

    # Calculate patch dims per ramp
    patch_offset_px = (np.array(patch_offset_um)*1000 // np.array(pw_nm)).astype(int)
    patch_dim = ((ramp_size-patch_size_px)//patch_offset_px).astype(int)

    # Convert sample offset to pixels and create mesh
    uls, lrs = [], []
    for click_idx, click in enumerate(click_uls):
        curr_patch_dim = patch_dim[click_idx]
        for i in range(curr_patch_dim[0]):
            ul_x = i*patch_offset_px
            for j in range(curr_patch_dim[1]):
                ul_y = j*patch_offset_px
                uls.append(click + np.array([ul_x, ul_y]))
                lrs.append(click + np.array([ul_x, ul_y]) + patch_size_px)

    # Close image
    plt.close(fig)

    # Normalisation of image
    if normalise:
        midrange = [np.percentile(img, 10), np.percentile(img, 90)]
        img_cec = (img-midrange[0]) / (midrange[1]-midrange[0])
        img_cec[img_cec<0] = 0
        img_cec[img_cec>1] = 1
        out_image = img_cec[idx, :-trunc]

    return out_image, np.array(uls, dtype=int), np.array(lrs, dtype=int)
