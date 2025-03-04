"""This creates data for tests"""

import os
import sys
import numpy as np
import scipy.stats as sps
import pandas as pd
from scipy import interpolate as sciinterp
from scipy import integrate as sciint
from pathlib import Path


import qp
from qp.parameterizations.packed_interp.packing_utils import PackingType, pack_array
from qp.parameterizations.sparse_interp import sparse_rep


###############################################################################
## Variables
###############################################################################

np.random.seed(1234)

NPDF = 11
NBIN = 61
NSAMPLES = 100
XMIN = 0.0
XMAX = 5.0
LOC = np.expand_dims(np.linspace(0.5, 2.5, NPDF), -1)
SCALE = np.expand_dims(np.linspace(0.2, 1.2, NPDF), -1)
LOC_SHIFTED = LOC + SCALE
TEST_XVALS = np.linspace(XMIN, XMAX, 201)
XBINS = np.linspace(XMIN, XMAX, NBIN)
XARRAY = np.ones((NPDF, NBIN)) * XBINS
YARRAY = np.expand_dims(np.linspace(0.5, 2.5, NPDF), -1) * (
    1.0 + 0.1 * np.random.uniform(size=(NPDF, NBIN))
)
HIST_DATA = YARRAY[:, 0:-1]
QUANTS = np.linspace(0.01, 0.99, NBIN)
QLOCS = sps.norm(loc=LOC, scale=SCALE).ppf(QUANTS)
SAMPLES = sps.norm(loc=LOC, scale=SCALE).rvs(size=(NPDF, NSAMPLES))

MEAN_MIXMOD = np.vstack(
    [
        np.linspace(0.5, 2.5, NPDF),
        np.linspace(0.5, 1.5, NPDF),
        np.linspace(1.5, 2.5, NPDF),
    ]
).T
STD_MIXMOD = np.vstack(
    [
        np.linspace(0.2, 1.2, NPDF),
        np.linspace(0.2, 0.5, NPDF),
        np.linspace(0.2, 0.5, NPDF),
    ]
).T
WEIGHT_MIXMOD = np.vstack(
    [0.7 * np.ones((NPDF)), 0.2 * np.ones((NPDF)), 0.1 * np.ones((NPDF))]
).T

HIST_TOL = 4.0 / NBIN
QP_TOPDIR = os.path.dirname(os.path.dirname(__file__))


###############################################################################
## Data construction helper functions
###############################################################################


def calc_ypacked():
    """Writes out test data and parameters for the packed interp parameterization"""

    ypacked_lin, ymax_lin = pack_array(PackingType.linear_from_rowmax, YARRAY.copy())
    ypacked_log, ymax_log = pack_array(
        PackingType.log_from_rowmax, YARRAY.copy(), log_floor=-3
    )

    return ypacked_lin, ymax_lin, ypacked_log, ymax_log


def get_sparse_data():
    """Writes out test data and parameters for the sparse parameterization"""

    TEST_DIR = Path(__file__).resolve().parent
    filein = os.path.join(TEST_DIR.parent, "test_data", "CFHTLens_sample.P.npy")

    # FORMAT FILE, EACH ROW IS THE PDF FOR EACH GALAXY, LAST ROW IS THE REDSHIFT POSITION
    P = np.load(filein)
    z = P[-1]
    P = P[:NPDF]
    P = P / sciint.trapezoid(P, z).reshape(-1, 1)
    minz = np.min(z)
    nz = 301
    _, j = np.where(P > 0)
    maxz = np.max(z[j + 1])
    newz = np.linspace(minz, maxz, nz)
    interp = sciinterp.interp1d(z, P, assume_sorted=True)
    newpdf = interp(newz)
    newpdf = newpdf / sciint.trapezoid(newpdf, newz).reshape(-1, 1)
    sparse_idx, meta, _ = sparse_rep.build_sparse_representation(
        newz, newpdf, verbose=False
    )

    return sparse_idx, meta
