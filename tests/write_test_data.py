"""This small module collects inputs for unit tests"""

import os
import sys
import numpy as np
import scipy.stats as sps
import pandas as pd
from scipy import interpolate as sciinterp
from scipy import integrate as sciint


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
## Hist test data
###############################################################################
def write_hist_test_data(test_data_dir):
    hist_test_data = dict(
        hist=dict(
            gen_func=qp.hist,
            ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),
            convert_data=dict(bins=XBINS),
            atol_diff=1e-1,
            atol_diff2=1e-1,
            test_xvals=TEST_XVALS,
        ),
        hist_samples=dict(
            gen_func=qp.hist,
            ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),
            convert_data=dict(bins=XBINS, method="samples", size=NSAMPLES),
            atol_diff=1e-1,
            atol_diff2=1e-1,
            test_xvals=TEST_XVALS,
            do_samples=True,
        ),
    )

    # write data
    df = pd.DataFrame(hist_test_data)
    filepath = test_data_dir / f"{qp.hist.name}-test-data.csv"
    df.to_csv(filepath, index=False)


###############################################################################
## Interp test data
###############################################################################

interp_test_data = dict(
    interp=dict(
        gen_func=qp.interp,
        ctor_data=dict(xvals=XBINS, yvals=YARRAY),
        convert_data=dict(xvals=XBINS),
        test_xvals=TEST_XVALS,
    )
)

interp_irreg_test_data = dict(
    interp_irregular=dict(
        gen_func=qp.interp_irregular,
        ctor_data=dict(xvals=XARRAY, yvals=YARRAY),
        convert_data=dict(xvals=XBINS),
        test_xvals=TEST_XVALS,
    )
)

###############################################################################
## Mixmod test data
###############################################################################

mixmod_test_data = dict(
    mixmod=dict(
        gen_func=qp.mixmod,
        ctor_data=dict(weights=WEIGHT_MIXMOD, means=MEAN_MIXMOD, stds=STD_MIXMOD),
        convert_data={},
        test_xvals=TEST_XVALS,
        atol_diff2=1.0,
    )
)

###############################################################################
## Quant test data
###############################################################################

quant_test_data = dict(
    gen_func=qp.quant,
    ctor_data=dict(quants=QUANTS, locs=QLOCS),
    convert_data=dict(quants=QUANTS),
    test_xvals=TEST_XVALS,
)

###############################################################################
## Norm test data
###############################################################################

norm_test_data = dict(
    norm=dict(
        gen_func=qp.stats.norm,
        ctor_data=dict(loc=LOC, scale=SCALE),
        test_xvals=TEST_XVALS,
        do_samples=True,
        ancil=dict(zmode=LOC),
    ),
    norm_shifted=dict(
        gen_func=qp.stats.norm,
        ctor_data=dict(loc=LOC, scale=SCALE),
        test_xvals=TEST_XVALS,
    ),
    norm_multi_d=dict(
        gen_func=qp.stats.norm,
        ctor_data=dict(loc=np.array([LOC, LOC]), scale=np.array([SCALE, SCALE])),
        test_xvals=TEST_XVALS,
        do_samples=True,
    ),
)

###############################################################################
## Packed interp test data
###############################################################################

ypacked_lin, ymax_lin = pack_array(PackingType.linear_from_rowmax, YARRAY.copy())
ypacked_log, ymax_log = pack_array(
    PackingType.log_from_rowmax, YARRAY.copy(), log_floor=-3
)

packed_interp_test_data = dict(
    lin_packed_interp=dict(
        gen_func=qp.packed_interp,
        ctor_data=dict(
            packing_type=PackingType.linear_from_rowmax,
            xvals=XBINS,
            ypacked=ypacked_lin,
            ymax=ymax_lin,
        ),
        convert_data=dict(xvals=XBINS, packing_type=PackingType.linear_from_rowmax),
        test_xvals=TEST_XVALS,
    ),
    log_packed_interp=dict(
        gen_func=qp.packed_interp,
        ctor_data=dict(
            packing_type=PackingType.log_from_rowmax,
            xvals=XBINS,
            ypacked=ypacked_log,
            ymax=ymax_log,
            log_floor=-3.0,
        ),
        convert_data=dict(
            xvals=XBINS,
            packing_type=PackingType.log_from_rowmax,
            log_floor=-3.0,
        ),
        test_xvals=TEST_XVALS,
    ),
)

###############################################################################
## Sparse test data
###############################################################################

qproot = sys.modules["qp"].__path__[0]
filein = os.path.join(qproot, "../../tests/qp/test_data/CFHTLens_sample.P.npy")
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

sparse_test_data = dict(
    sparse=dict(
        gen_func=qp.sparse,
        ctor_data=dict(
            xvals=meta["xvals"],
            mu=meta["mu"],
            sig=meta["sig"],
            dims=meta["dims"],
            sparse_indices=sparse_idx,
        ),
        test_xvals=TEST_XVALS,
    ),
)

###############################################################################
## Spline test data
###############################################################################

SPLX, SPLY, SPLN = qp.spline.build_normed_splines(XARRAY, YARRAY)
spline_test_data = dict(
    spline=dict(
        gen_func=qp.spline,
        ctor_data=dict(splx=SPLX, sply=SPLY, spln=SPLN),
        test_xvals=TEST_XVALS[::10],
    ),
    spline_kde=dict(
        gen_func=qp.spline_from_samples,
        ctor_data=dict(samples=SAMPLES, xvals=np.linspace(0, 5, 51)),
        convert_data=dict(xvals=np.linspace(0, 5, 51), method="samples"),
        test_xvals=TEST_XVALS,
        atol_diff2=1.0,
        test_pdf=False,
    ),
    spline_xy=dict(
        gen_func=qp.spline_from_xy,
        ctor_data=dict(xvals=XARRAY, yvals=YARRAY),
        convert_data=dict(xvals=np.linspace(0, 5, 51), method="xy"),
        test_xvals=TEST_XVALS,
        test_pdf=False,
    ),
)
