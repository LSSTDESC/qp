import numpy as np
import pytest

import qp
from tests.helpers.test_funcs import assert_all_close


@pytest.fixture
def quant_ensemble(quant_test_data) -> qp.Ensemble:
    ens = qp.quant.create_ensemble(**quant_test_data["quant"]["ctor_data"])
    return ens


def test_xsamples(quant_ensemble):

    xsamps = quant_ensemble.x_samples()

    xmin = np.min(quant_ensemble.objdata["locs"])
    xmax = np.max(quant_ensemble.objdata["locs"])
    min_dx = np.median(np.diff(quant_ensemble.objdata["locs"]))

    npts = (xmax - xmin) // min_dx
    npts = np.min([int(npts), 10000])
    x_compare = np.linspace(xmin, xmax, npts)

    assert_all_close(xsamps, x_compare)


@pytest.mark.parametrize(
    "quants, locs, match_string",
    [
        (
            np.array([0, 0.5, 1]),
            np.array([0.2, 0.56, np.inf]),
            "There are non-finite values in the locs for the distributions:",
        )
    ],
)
def test_warnings(quants, locs, match_string):
    """Test that warnings when building an interpolated Ensemble work."""

    with pytest.warns(RuntimeWarning, match=match_string):
        ens = qp.quant.create_ensemble(quants=quants, locs=locs)


@pytest.mark.parametrize(
    "quants, locs, match_string",
    [
        (
            np.array([-0.5, 0, 0.5]),
            np.array([0.2, 0.5, 0.6]),
            "Invalid quants: One or more of the given quants is outside the allowed range",
        ),
        (
            np.array([0, 0.5, 0.2]),
            np.array([0.2, 1.0, 1.5]),
            "There are decreasing values, quants must be given in order from 0 to 1",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([0.5, 0.2, 1.2]),
            "The given data does not produce a one-to-one CDF",
        ),
    ],
)
def test_validate_input(quants, locs, match_string):
    """Tests that the appropriate errors are raised when invalid input is given"""

    with pytest.raises(ValueError, match=match_string) as err:
        ens_h = qp.quant.create_ensemble(quants=quants, locs=locs, warn=False)


@pytest.mark.parametrize(
    "pdf_constructor_name",
    [
        ("piecewise"),
        ("piecewise".encode()),
    ],
)
def test_quant_constructor(pdf_constructor_name):
    """Make sure that quant parameterization doesn't work if the constructor name
    is not in the given list."""

    quants = np.linspace(0, 1, 5)
    locs = np.linspace(-1, 1, 5)

    with pytest.raises(ValueError, match="Unknown interpolator provided:") as exec_info:
        qp.quant.create_ensemble(quants, locs, pdf_constructor_name)
