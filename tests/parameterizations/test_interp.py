import pytest
import numpy as np
import warnings

import qp
from tests.helpers.test_data_helper import NPDF
from tests.helpers.test_funcs import assert_all_close


@pytest.fixture
def interp_ensemble(interp_test_data):
    ens_i = qp.interp.create_ensemble(**interp_test_data["interp"]["ctor_data"])
    return ens_i


@pytest.fixture
def irreg_interp_ensemble(interp_irreg_test_data):
    ens_i = qp.interp_irregular.create_ensemble(
        **interp_irreg_test_data["interp_irregular"]["ctor_data"]
    )
    return ens_i


#
# Interp Ensemble tests
#


def test_norm(interp_test_data):
    """Test that the norm method works as expected."""

    norm = False
    ancil = {"ancil": np.linspace(0, 11, NPDF)}
    ens_i = qp.interp.create_ensemble(
        **interp_test_data["interp"]["ctor_data"], norm=norm, ancil=ancil
    )

    assert ens_i.npdf == NPDF

    # make sure that values were not normalized
    xmin = ens_i.frozen.dist._xmin
    xmax = ens_i.frozen.dist._xmax
    integral = ens_i.integrate([xmin, xmax])
    assert not np.allclose(integral, np.ones(NPDF), rtol=1e-2)

    # make sure that the normalization has worked as expected
    ens_i.norm()
    integral_norm = ens_i.integrate([xmin, xmax])
    assert_all_close(integral_norm, np.ones(NPDF), rtol=1e-2)

    # make sure that the ancil table has not changed
    assert ens_i.ancil == ancil


def test_xsamples(interp_ensemble):
    """Make sure that x_samples generates values as expected."""

    x_samps = interp_ensemble.x_samples()
    assert len(interp_ensemble.metadata["xvals"]) == len(x_samps)

    assert_all_close(x_samps, interp_ensemble.metadata["xvals"])


@pytest.mark.parametrize(
    "xvals, yvals, match_string",
    [
        (
            np.array([0, 0.5, np.inf]),
            np.array([1.0, 2.0, 3.0]),
            "The given xvals contain non-finite values",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([1, np.nan, 1]),
            "There are non-finite values in the yvals for the following distributions:",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([-0.5, 1, 1.5]),
            "There are negative values in the yvals for the following distributions:",
        ),
    ],
)
def test_warnings(xvals, yvals, match_string):
    """Test that warnings when building an interpolated Ensemble work."""

    with pytest.warns(RuntimeWarning, match=match_string):
        ens_i = qp.interp.create_ensemble(xvals=xvals, yvals=yvals)


@pytest.mark.parametrize(
    "xvals, yvals, match_string",
    [
        (
            np.array([1.0, 1.5, 2.0]),
            np.array([1, 2, 3, 4]),
            "Shape of xbins in xvals",
        ),
        (
            np.array([0.5, 1.5, 1.0]),
            np.array([1, 3, 2]),
            "Invalid xvals: The given xvals are not sorted:",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([-0.5, -1, -0.5]),
            "cannot be properly normalized",
        ),
    ],
)
def test_invalid_input(xvals, yvals, match_string):
    """Tests that the appropriate errors are raised when invalid input is given"""

    with pytest.raises(ValueError, match=match_string) as err:
        ens_h = qp.interp.create_ensemble(xvals=xvals, yvals=yvals, warn=False)


#
# Irregular Interp Ensemble tests
#


def test_irreg_norm(interp_irreg_test_data):
    """Test that the norm method works as expected."""

    norm = False
    ancil = {"ancil": np.linspace(0, 11, NPDF)}
    ens_i = qp.interp_irregular.create_ensemble(
        **interp_irreg_test_data["interp_irregular"]["ctor_data"],
        norm=norm,
        ancil=ancil,
    )

    assert ens_i.npdf == NPDF

    # make sure that values were not normalized
    xmin = ens_i.frozen.dist._xmin
    xmax = ens_i.frozen.dist._xmax
    integral = ens_i.integrate([xmin, xmax])
    assert not np.allclose(integral, np.ones(NPDF), rtol=1e-2)

    # make sure that the normalization has worked as expected
    ens_i.norm()
    integral_norm = ens_i.integrate([xmin, xmax])
    assert_all_close(integral_norm, np.ones(NPDF), rtol=1e-2)

    # make sure that the ancil table has not changed
    assert ens_i.ancil == ancil


def test_irreg_xsamples(irreg_interp_ensemble):
    """Make sure that x_samples generates values as expected."""

    x_samps = irreg_interp_ensemble.x_samples()
    dx = np.min(np.diff(irreg_interp_ensemble.frozen.dist._yvals))
    compar_val = np.arange(
        np.min(irreg_interp_ensemble.frozen.dist._xvals),
        np.max(irreg_interp_ensemble.frozen.dist._yvals),
        dx,
    )

    assert_all_close(x_samps, compar_val)


@pytest.mark.parametrize(
    "xvals, yvals, match_string",
    [
        (
            np.array([0, 0.5, np.inf]),
            np.array([1.0, 2.0, 3.0]),
            "The given xvals contain non-finite values for the following distributions",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([1, np.nan, 1]),
            "There are non-finite values in the yvals for the following distributions:",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([-0.5, 1, 1.5]),
            "There are negative values in the yvals for the following distributions:",
        ),
    ],
)
def test_irreg_warnings(xvals, yvals, match_string):
    """Test that warnings when building an interpolated Ensemble work."""

    with pytest.warns(RuntimeWarning, match=match_string):
        ens_i = qp.interp_irregular.create_ensemble(xvals=xvals, yvals=yvals)


@pytest.mark.parametrize(
    "xvals, yvals, match_string",
    [
        (
            np.array([1.0, 1.5, 2.0]),
            np.array([1, 2, 3, 4]),
            "Shape of xvals",
        ),
        (
            np.array([0.5, 1.5, 1.0]),
            np.array([1, 3, 2]),
            "Invalid xvals: The given xvals are not sorted:",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([-0.5, -1, -0.5]),
            "The integral is < 0 for distributions at indices",
        ),
    ],
)
def test_irreg_invalid_input(xvals, yvals, match_string):
    """Tests that the appropriate errors are raised when invalid input is given"""

    with pytest.raises(ValueError, match=match_string) as err:
        ens_i = qp.interp_irregular.create_ensemble(
            xvals=xvals, yvals=yvals, warn=False
        )


def test_compute_ycumul_error():
    """Make sure that computing the CDF when it's less than 0 raises the appropriate error, even when norm=False."""

    xvals = np.array([0, 0.5, 1])
    yvals = np.array([-0.5, -1, -0.5])
    ens_i = qp.interp_irregular.create_ensemble(
        xvals=xvals, yvals=yvals, norm=False, warn=False
    )
    with pytest.raises(
        ValueError, match="The integral is < 0 for distributions at indices"
    ):
        value = ens_i.cdf(1.0)
