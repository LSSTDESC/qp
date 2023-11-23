"""This small module collects inputs for unit tests"""

import os
import numpy as np
import scipy.stats as sps

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
