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


from magicgui import magicgui as mg
import pandas as pd
import numpy as np
from pathlib import Path

from functools import partial
from joblib import Parallel, delayed, parallel_backend

import cProfile as profile
import pstats
from icecream import ic


from . import io
from . import calculations as Calc
from . import logger as Logger


@mg(
    call_button="Run!",
    layout="vertical",
    result_widget=False,

    input_csv={"widget_type": "FileEdit",
               "label": "Path to input images metadata (CSV)"},
    patch_dims={"widget_type": "LiteralEvalLineEdit",
                "label": "Number of patches per ramp in each direction (Nx, Ny)"},
    patch_size_um={"widget_type": "LiteralEvalLineEdit",
                   "label": "Dimensions of each patch in um"},
    patch_offset_um={"min": 0,
                     "label": "Patch offset in um"},
    sample_size_nm={"label": "Size of artificial Gaussian samples in nm",
                    "min": 300},
    num_samples={"label": "Number of artificial Gaussian samples used",
                 "min": 1},
    output_pkl_image={"label": "Path to pickled output scores by image"},
    output_pkl_setting={"label": "Path to pickled output scores by setting"},
)
def calculate(
        input_csv = Path("."),
        patch_dims = (2, 2),
        patch_size_um = (1, 1),
        patch_offset_um = 0.2,
        sample_size_nm = 1000.0,
        num_samples = 15,
        output_pkl_image = Path("."),
        output_pkl_setting = Path(".")
):
    """
    Method to run calculation
    """
    # Create logger object
    log = Logger.Logger()

    # Get user parameter inputs and read input CSV
    params = locals()
    log("Reading input CSV...")
    df = pd.read_csv(str(params['input_csv']))

    # Check if essential columns exist in dataframe
    essential_columns = ["Filename", "Image name", "Gas", "FIB Current (nA)", "PW (nm)", "Nramps"]
    try:
        assert (set(essential_columns).issubset(df.columns))
    except:
        log("Error - The columns {set(essential_columns)} must be present in input CSV file.")
        return

    # Patch-picking
    for path in df.Filename.to_list():
        block_size = int(params['sample_size_nm'] // df.loc[df.Filename==path]["PW (nm)"].values[0])
        sample_dims = [int(i*1000/df.loc[df.Filename==path]["PW (nm)"].values[0]) for i in params['patch_size_um']]

        image, cntrs, uls, lrs = io.open_image(path=path,
                                               pw_nm=df.loc[df.Filename==path]["PW (nm)"].values[0],
                                               patch_size_um=params['patch_size_um'],
                                               patch_dims=params['patch_dims'],
                                               patch_offset_um=params['patch_offset_um'],
                                               num_ramps=df.loc[df.Filename==path]["Nramps"].values[0],
                                               normalise=True
        )
        df.loc[df.Filename==path, "block_size"] = block_size
        df.loc[df.Filename==path, "images"] = Calc.Fake(image)
        df.loc[df.Filename==path, "centres"] = Calc.Fake(cntrs)
        df.loc[df.Filename==path, "uls"] = Calc.Fake(uls)
        df.loc[df.Filename==path, "lrs"] = Calc.Fake(lrs)

    # Get gaussian strip samples
    log("Preparing xcorr templates...")
    curt_samples = Calc.get_samples_dict(df_in=df,
                                         num_sigma=params['num_samples'])

    # Get aggregated parameter list
    log("Preparing job list...")
    params_list = Calc.get_full_params_list(df_in=df,
                                            samples_in=curt_samples)

    # Actual parallel calculation
    log(f"Starting calculation with {len(params_list)} jobs... \n")
    heatmap = Calc.calc_parallel(30, 0.1, params_list, patch_dim=params['patch_dims'])

    # Aggregate data and output to pickle
    print("")
    log("All calculation finished. Now aggregating results...")
    scores_df = Calc.aggregate(df_in=df,
                               heatmap_in=heatmap,
                               num_samples=params['num_samples'])

    print(df[["Gas", "FIB Current (nA)", "Image name", "S_mean", "S_sd"]])

    df.to_pickle(str(params['output_pkl_image']))
    scores_df.to_pickle(str(params['output_pkl_setting']))

    print("")
    log("All finished. The GUI can be safely closed now.")
    return


def calculate_wrapper():
    calculate.show(run=True)
