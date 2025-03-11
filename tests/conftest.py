import pytest
from pathlib import Path
import numpy as np
import scipy.stats as sps
from tests.helpers import test_data_helper
import qp
from qp.parameterizations.packed_interp.packing_utils import PackingType


@pytest.fixture
def test_dir() -> Path:
    """Return path to test directory

    Returns
    -------
    Path
        Path to test directory
    """
    return Path(__file__).resolve().parent


@pytest.fixture
def test_data_dir(test_dir) -> Path:
    return test_dir / "test_data"


@pytest.fixture
def hist_test_data():
    """Writes out test data and parameters for the histogram parameterization"""

    hist_test_data = dict(
        hist=dict(
            gen_func=qp.hist,
            ctor_data=dict(
                bins=test_data_helper.XBINS, pdfs=test_data_helper.HIST_DATA
            ),
            convert_data=dict(bins=test_data_helper.XBINS),
            atol_diff=1e-1,
            atol_diff2=1e-1,
            test_xvals=test_data_helper.TEST_XVALS,
        ),
        hist_samples=dict(
            gen_func=qp.hist,
            ctor_data=dict(
                bins=test_data_helper.XBINS, pdfs=test_data_helper.HIST_DATA
            ),
            convert_data=dict(
                bins=test_data_helper.XBINS,
                method="samples",
                size=test_data_helper.NSAMPLES,
            ),
            atol_diff=1e-1,
            atol_diff2=1e-1,
            test_xvals=test_data_helper.TEST_XVALS,
            do_samples=True,
        ),
    )
    return hist_test_data


@pytest.fixture
def interp_test_data():
    """Writes out test data and parameters for the interp parameterizations"""

    interp_test_data = dict(
        interp=dict(
            gen_func=qp.interp,
            ctor_data=dict(xvals=test_data_helper.XBINS, yvals=test_data_helper.YARRAY),
            convert_data=dict(xvals=test_data_helper.XBINS),
            test_xvals=test_data_helper.TEST_XVALS,
        )
    )
    return interp_test_data


@pytest.fixture
def interp_irreg_test_data():

    interp_irreg_test_data = dict(
        interp_irregular=dict(
            gen_func=qp.interp_irregular,
            ctor_data=dict(
                xvals=test_data_helper.XARRAY, yvals=test_data_helper.YARRAY
            ),
            convert_data=dict(xvals=test_data_helper.XBINS),
            test_xvals=test_data_helper.TEST_XVALS,
        )
    )

    return interp_irreg_test_data


@pytest.fixture
def mixmod_test_data():
    """Writes out test data and parameters for the mixmod parameterization"""

    mixmod_test_data = dict(
        mixmod=dict(
            gen_func=qp.mixmod,
            ctor_data=dict(
                weights=test_data_helper.WEIGHT_MIXMOD,
                means=test_data_helper.MEAN_MIXMOD,
                stds=test_data_helper.STD_MIXMOD,
            ),
            convert_data={},
            test_xvals=test_data_helper.TEST_XVALS,
            atol_diff2=1.0,
        )
    )
    return mixmod_test_data


@pytest.fixture
def write_test_quant_data():
    """Writes out test data and parameters for the quant parameterization"""

    quant_test_data = dict(
        gen_func=qp.quant,
        ctor_data=dict(quants=test_data_helper.QUANTS, locs=test_data_helper.QLOCS),
        convert_data=dict(quants=test_data_helper.QUANTS),
        test_xvals=test_data_helper.TEST_XVALS,
    )

    return quant_test_data


@pytest.fixture
def norm_test_data():
    """Writes out test data and parameters for the norm parameterization"""

    norm_test_data = dict(
        norm=dict(
            gen_func=qp.stats.norm,
            ctor_data=dict(loc=test_data_helper.LOC, scale=test_data_helper.SCALE),
            test_xvals=test_data_helper.TEST_XVALS,
            do_samples=True,
            ancil=dict(zmode=test_data_helper.LOC),
        ),
        norm_shifted=dict(
            gen_func=qp.stats.norm,
            ctor_data=dict(loc=test_data_helper.LOC, scale=test_data_helper.SCALE),
            test_xvals=test_data_helper.TEST_XVALS,
        ),
        norm_multi_d=dict(
            gen_func=qp.stats.norm,
            ctor_data=dict(
                loc=np.array([test_data_helper.LOC, test_data_helper.LOC]),
                scale=np.array([test_data_helper.SCALE, test_data_helper.SCALE]),
            ),
            test_xvals=test_data_helper.TEST_XVALS,
            do_samples=True,
        ),
    )

    return norm_test_data


@pytest.fixture
def packed_interp_test_data():
    """Writes out test data and parameters for the packed interp parameterization"""

    ypacked_lin, ymax_lin, ypacked_log, ymax_log = test_data_helper.calc_ypacked()

    packed_interp_test_data = dict(
        lin_packed_interp=dict(
            gen_func=qp.packed_interp,
            ctor_data=dict(
                packing_type=PackingType.linear_from_rowmax,
                xvals=test_data_helper.XBINS,
                ypacked=ypacked_lin,
                ymax=ymax_lin,
            ),
            convert_data=dict(
                xvals=test_data_helper.XBINS,
                packing_type=PackingType.linear_from_rowmax,
            ),
            test_xvals=test_data_helper.TEST_XVALS,
        ),
        log_packed_interp=dict(
            gen_func=qp.packed_interp,
            ctor_data=dict(
                packing_type=PackingType.log_from_rowmax,
                xvals=test_data_helper.XBINS,
                ypacked=ypacked_log,
                ymax=ymax_log,
                log_floor=-3.0,
            ),
            convert_data=dict(
                xvals=test_data_helper.XBINS,
                packing_type=PackingType.log_from_rowmax,
                log_floor=-3.0,
            ),
            test_xvals=test_data_helper.TEST_XVALS,
        ),
    )

    return packed_interp_test_data


@pytest.fixture
def sparse_test_data():
    """Writes out test data and parameters for the sparse parameterization"""

    sparse_idx, meta = test_data_helper.get_sparse_data()

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
            test_xvals=test_data_helper.TEST_XVALS,
        ),
    )

    return sparse_test_data


@pytest.fixture
def spline_test_data():
    """Writes out test data and parameters for the spline parameterization"""

    SPLX, SPLY, SPLN = qp.spline.build_normed_splines(
        test_data_helper.XARRAY, test_data_helper.YARRAY
    )
    spline_test_data = dict(
        spline=dict(
            gen_func=qp.spline,
            ctor_data=dict(splx=SPLX, sply=SPLY, spln=SPLN),
            test_xvals=test_data_helper.TEST_XVALS[::10],
        ),
        spline_kde=dict(
            gen_func=qp.spline_from_samples,
            ctor_data=dict(
                samples=test_data_helper.SAMPLES, xvals=np.linspace(0, 5, 51)
            ),
            convert_data=dict(xvals=np.linspace(0, 5, 51), method="samples"),
            test_xvals=test_data_helper.TEST_XVALS,
            atol_diff2=1.0,
            test_pdf=False,
        ),
        spline_xy=dict(
            gen_func=qp.spline_from_xy,
            ctor_data=dict(
                xvals=test_data_helper.XARRAY, yvals=test_data_helper.YARRAY
            ),
            convert_data=dict(xvals=np.linspace(0, 5, 51), method="xy"),
            test_xvals=test_data_helper.TEST_XVALS,
            test_pdf=False,
        ),
    )

    return spline_test_data


## Fixtures to create ensembles
@pytest.fixture
def hist_ensemble(hist_test_data):
    ens_h = qp.hist.create_ensemble(**hist_test_data["hist"]["ctor_data"])
    return ens_h
