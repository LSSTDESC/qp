import pytest
import warnings
import numpy as np
import qp

from tests.helpers.test_data_helper import NPDF
from tests.helpers.test_funcs import assert_all_close


def test_norm(hist_test_data):
    """Test that the histogram norm method works as expected."""

    norm = False
    ancil = {"ancil": np.linspace(0, 11, NPDF)}
    ens_h = qp.hist.create_ensemble(
        **hist_test_data["hist"]["ctor_data"], norm=norm, ancil=ancil
    )

    assert ens_h.npdf == NPDF

    # make sure that values were not normalized
    xmin = ens_h.frozen.dist._xmin
    xmax = ens_h.frozen.dist._xmax
    integral = ens_h.integrate([xmin, xmax])
    assert not np.allclose(integral, np.ones(NPDF))

    # make sure that the normalization has worked as expected
    ens_h.norm()
    integral_norm = ens_h.integrate([xmin, xmax])
    assert_all_close(integral_norm, np.ones(NPDF))

    # make sure that the ancil table has not changed
    assert ens_h.ancil == ancil


def test_xsamples(hist_ensemble):
    """Make sure that x_samples generates values as expected."""

    x_samps = hist_ensemble.x_samples()
    assert len(hist_ensemble.metadata["bins"]) - 1 == len(x_samps)

    bin_centers = (
        hist_ensemble.metadata["bins"][:-1] + hist_ensemble.metadata["bins"][1:]
    ) / 2

    assert_all_close(x_samps, bin_centers)


@pytest.mark.parametrize(
    "bins, pdfs",
    [
        (np.array([0, 0.5, 1, np.inf]), np.array([1.0, 2.0, 3.0])),
        (np.array([0, 0.5, 1, 1.5]), np.array([1, np.nan, 1])),
        (np.array([0, 0.5, 1, 1.5]), np.array([-0.5, 1, 1.5])),
    ],
)
def test_warnings(bins, pdfs):
    """Test that warnings when building a histogram Ensemble work."""

    with pytest.warns(RuntimeWarning):
        ens_h = qp.hist.create_ensemble(bins=bins, pdfs=pdfs)


@pytest.mark.parametrize(
    "bins,pdfs, match_string",
    [
        (
            np.array([1.0, 0.5, 1.5, 2]),
            np.array([1, 2, 3]),
            "Invalid bins: The given bins are not sorted:",
        ),
        (np.array([0, 0.5, 1, 1.5]), np.array([1, 2, 3, 4]), "Number of bins"),
        (
            np.array([0, 0.5, 1, 1.5]),
            np.array([-0.5, -1, -0.5]),
            "cannot be properly normalized",
        ),
    ],
)
def test_invalid_input(bins, pdfs, match_string):
    """Tests that the appropriate errors are raised when invalid input is given"""

    with pytest.raises(ValueError, match=match_string) as err:
        ens_h = qp.hist.create_ensemble(bins=bins, pdfs=pdfs, warn=False)


def test_hist_bins2d():
    """Test that passing 2D bin array works"""

    bins = np.array([[0, 0.5, 1, 1.5]])
    pdfs = np.array([1, 2, 3])

    ens = qp.hist.create_ensemble(bins=bins, pdfs=pdfs)
