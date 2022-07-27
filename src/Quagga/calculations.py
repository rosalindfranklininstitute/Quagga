# Copyright 2022 Rosalind Franklin Institute
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


import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed, parallel_backend

import numpy as np
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from skimage.feature import match_template

from icecream import ic

np.seterr(all="ignore")

class Fake(object):
    """
    Class encapsulating a Fake object
    """
    def __init__(self, li_obj):
        self.obj = li_obj


def get_gaussian_sample(dims, sigma, theta):
    """
    Method to calculate a single Gaussian strip sample

    Args:
    dims (int) :: Width of sample (square shaped)
    sigma (float) :: Standard deviation of Gaussian strip
    theta (float) :: Rotational angle of Gaussian strip

    Returns:
    ndarray
    """
    xx = np.linspace(-dims/2, dims/2, dims)
    yy = np.linspace(-dims/2, dims/2, dims)
    vx, vy = np.meshgrid(xx, yy, indexing='ij')

    gaussian_base = np.exp(-0.5*(vy)**2 / sigma**2) / (sigma * np.sqrt(2*np.pi))
    gaussian = rotate(gaussian_base, theta, reshape=False)

    if theta != 0:
        crop = abs(int(np.ceil(0.5 * dims * (np.sin(np.deg2rad(theta))))))
        gaussian = gaussian[crop:-(crop+1), crop:-(crop+1)]

    return gaussian


def get_samples_dict(df_in, num_sigma):
    """
    Method to obtain Gaussian strip samples

    Args:
    df_in (DataFrame) :: Pandas dataframe containing all raw image information
    num_sigma (int) :: Number of gaussian samples

    Returns:
    dict
    """
    sigma_list = np.linspace(0.5, 5, num_sigma)
    curt_samples = {}

    for size in set(df_in[df_in["images"].notna()].block_size.to_numpy()):
        curt_samples[size] = [get_gaussian_sample(int(size), sigma, theta=0) for sigma in sigma_list]

    return curt_samples


def get_full_params_list(df_in, samples_in):
    """
    Method to obtain full flattened parameters list for processing

    Args:
    df_in (DataFrame) :: Pandas dataframe containing all raw image information
    samples_in (dict) :: Dictionary holding all gaussian samples

    Returns:
    list
    """
    params_list_df = df_in[df_in['images'].notna()][['images', 'block_size', 'uls', 'lrs']].values.tolist()
    params_list_cleaned = [[(item[0].obj, int(item[1]), item[2].obj, item[3].obj, sample) for \
                            sample in samples_in[item[1]]] for item in params_list_df]
    params_list_flattened = [item for sublist in params_list_cleaned for item in sublist]

    return params_list_flattened


def calc_corr(img_in, sample_in):
    """
    Method to calculate cross-correlation between image and sample

    Args:
    img_in (ndarray) :: Raw image
    sample_in (ndarray) :: Artificial Gaussian sample used for xcorr

    Returns:
    ndarray
    """
    filtered = match_template(img_in, sample_in, pad_input=True)

    return filtered


def get_proportion(matched_in):
    """
    Method to calculate the proportion of curtaining given a pre-calculated mask

    Args:
    matched_in (ndarray) :: Masking array

    Returns:
    float
    """
    proportion = np.sum(matched_in) / np.size(matched_in)

    return proportion


def get_extent(img_in, sample_shape):
    """
    Method to calculate extent score for a curtained region

    Args:
    img_in (ndarray) :: Region to be assessed
    sample_shape (2-tuple) :: Shape of the sample patch
    """
    extent_array = np.zeros(img_in.shape, dtype=float)
    medians = np.zeros(img_in.shape, dtype=float)

    for i in range(0, img_in.shape[0], sample_shape[0]):
        for j in range(0, img_in.shape[1], sample_shape[1]):
            curr_sample = img_in[i:i+sample_shape[0],
                                 j:j+sample_shape[1]]
            sample_fft = np.fft.fftshift(np.fft.fft2(curr_sample)) / curr_sample.size
            sum_fft = np.sum(np.log10(np.abs(sample_fft) + 1e-15), axis=1)

            extent_array[i:i+sample_shape[0], j:j+sample_shape[1]] = (np.nanmax(sum_fft)-np.nanmedian(sum_fft)) / ((2*np.sqrt(2*np.log(2)))*np.nanstd(sum_fft)+1e-15) - 1
            medians[i:i+sample_shape[0], j:j+sample_shape[1]] = np.nanmedian(sum_fft)

    return np.nan_to_num(extent_array)


def get_metrics(img_in, filtered_in, ft_patchsize, ul, lr, cc_thres, rough=False):
    img = img_in[ul[0]:lr[0], ul[1]:lr[1]]

    # First calculate the cross-correlation to determine ROI
    matched = (filtered_in > cc_thres)
    matched_cropped = matched[ul[0]:lr[0], ul[1]:lr[1]]

    # Then calculate the proportion of ROI in original image
    prop_metric = get_proportion(matched_cropped)

    if rough:
        return prop_metric

    # Lastly calculate the extent of curtaining using FT-based filtering
    extent_region = get_extent(img, ft_patchsize)*matched_cropped
    extent = np.zeros_like(img_in, dtype=float)
    extent[ul[0]:lr[0], ul[1]:lr[1]] = extent_region

    if extent_region[matched_cropped].size > 0:
        extent_metric = np.nanmean(extent_region[matched_cropped], axis=None)
    else:
        extent_metric = 0.0

    return prop_metric, extent_metric


def get_stats_1pt(ul, lr, img_in, sample_in, ft_patchsize, prop_range=(0.05, 0.95), rough_pts=50, fine_spacing=0.1):
    thres_list_rough = np.linspace(-0.5, 0.5, rough_pts)
    prop_list_rough = np.zeros(len(thres_list_rough))

    filtered = sample_in

    for idx, thres in enumerate(thres_list_rough):
        prop_metric = get_metrics(img_in=img_in,
                                  filtered_in=filtered,
                                  cc_thres=thres,
                                  ft_patchsize=ft_patchsize,
                                  ul=ul,
                                  lr=lr,
                                  rough=True)
        prop_list_rough[idx] = prop_metric

    f = interp1d(prop_list_rough, thres_list_rough)

    thres_list_interp = np.arange(f(prop_range[1]), f(prop_range[0])+fine_spacing, fine_spacing)
    prop_list_smooth = np.zeros(len(thres_list_interp))
    extent_list = np.zeros(len(thres_list_interp))
    for idx, thres in enumerate(thres_list_interp):
        prop_metric, extent_metric = get_metrics(img_in=img_in,
                                                 filtered_in=filtered,
                                                 cc_thres=thres,
                                                 ft_patchsize=ft_patchsize,
                                                 ul=ul,
                                                 lr=lr)
        prop_list_smooth[idx] = prop_metric
        extent_list[idx] = extent_metric

    av_extent = np.nansum(prop_list_smooth * extent_list) / np.nansum(prop_list_smooth)
    se_extent = np.nanstd(extent_list) * np.nansum(prop_list_smooth**2) / np.nansum(prop_list_smooth)**2

    return av_extent, se_extent


def get_image_stats(img_in, sample_in, ft_patchsize, ul_list, lr_list, patch_dim, prop_range=(0.2,0.8), rough_pts=30, fine_spacing=0.1):
    value_list = []
    for ul, lr in zip(ul_list, lr_list):
        av, _ = get_stats_1pt(img_in=img_in,
                              sample_in=sample_in,
                              ft_patchsize=ft_patchsize,
                              ul=ul,
                              lr=lr,
                              prop_range=prop_range,
                              rough_pts=rough_pts,
                              fine_spacing=fine_spacing)
        value_list.append(av)

    patch_size = np.prod(patch_dim)
    values = np.array(value_list).reshape((len(value_list)//patch_size, patch_size))
    mean_values = np.mean(values, axis=1)

    return mean_values


def calc_basis_func(params_in, rough_pts, fine_spacing, patch_dim):
    img_in, ft_patchsize, ul_list, lr_list, sample_in = params_in
    filtered = calc_corr(img_in=img_in,
                         sample_in=sample_in)
    av = get_image_stats(img_in=img_in,
                         sample_in=filtered,
                         ft_patchsize=(ft_patchsize, ft_patchsize),
                         ul_list=ul_list,
                         lr_list=lr_list,
                         patch_dim=patch_dim,
                         prop_range=(0.2,0.8),
                         rough_pts=rough_pts,
                         fine_spacing=fine_spacing)
    return av


def calc_parallel(rough_pts, fine_spacing, params_list_in, patch_dim):
    g = partial(calc_basis_func,
                rough_pts=rough_pts,
                fine_spacing=fine_spacing,
                patch_dim=patch_dim)

    with parallel_backend("loky", n_jobs=-1):
        heatmap = Parallel(verbose=5)(delayed(g)(params) for params in params_list_in)

    return heatmap


def aggregate(df_in, heatmap_in, num_samples):
    temp = np.empty((len(heatmap_in),), dtype=object)
    for idx, item in enumerate(heatmap_in):
        temp[idx] = item
    heatmap_reshaped = temp.reshape(len(df_in), num_samples)

    for idx, item in df_in.iterrows():
        image_scores_full = np.array(list(heatmap_reshaped[idx]))
        image_scores = image_scores_full.mean(axis=0)
        df_in.loc[idx, "S_ramps"] = Fake(image_scores)
        df_in.loc[idx, "S_sum"] = np.sum(image_scores)
        df_in.loc[idx, "S_sumsq"] = np.sum(image_scores**2)
        df_in.loc[idx, "S_mean"] = np.mean(image_scores)
        df_in.loc[idx, "S_sd"] = np.std(image_scores)

    # Aggregating results and regrouping with settings
    df_score = df_in[['Filename', 'Image name', 'Gas', 'FIB Current (nA)', 'Nramps', 'S_sum', 'S_sumsq', 'S_mean', 'S_sd']]
    df_score.loc[:, 'Settings'] = df_score['Gas'].str.cat(df_score['FIB Current (nA)'].astype(str), sep='_')

    df_score_agg = df_score[['Settings', 'Nramps', 'S_sum', 'S_sumsq']].groupby('Settings', as_index=False).sum()

    mean_agg = df_score_agg.S_sum / df_score_agg.Nramps
    se_agg = np.sqrt((df_score_agg.S_sumsq - 2*mean_agg*df_score_agg.S_sum + df_score_agg.Nramps * mean_agg**2)) / df_score_agg.Nramps

    sep = df_score_agg.Settings.str.split('_', expand=True)
    df_score_agg['Gas'] = sep[0]
    df_score_agg['FIB Current (nA)'] = sep[1]
    df_score_agg.drop(columns=['Settings', 'S_sum', 'S_sumsq'], inplace=True)

    df_score_agg['S_mean'] = mean_agg
    df_score_agg['S_error_95'] = se_agg * 1.96

    return df_score_agg
