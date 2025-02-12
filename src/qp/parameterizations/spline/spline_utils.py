import numpy as np
from scipy import stats as sps

from ...utils.conversion import extract_xy_vals

# Conversion utility functions


def spline_extract_xy_vals(in_dist, **kwargs):
    """Wrapper for extract_xy_vals. Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data"""

    xvals = kwargs.pop("xvals", None)
    if xvals is None:  # pragma: no cover
        raise ValueError("To convert using extract_xy_vals you must specify xvals")
    return extract_xy_vals(in_dist, xvals)


def extract_samples(in_dist, **kwargs):
    """Convert using a set of values sampled from the PDF

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    size : `int`
        Number of samples to generate

    Returns
    -------
    data : `dict`
        The extracted data
    """
    samples = in_dist.rvs(size=kwargs.pop("size", 1000))
    xvals = kwargs.pop("xvals")
    return dict(samples=samples, xvals=xvals, yvals=None)


# Creation utility functions


def build_kdes(samples, **kwargs):
    """
    Build a set of Gaussian Kernel Density Estimates

    Parameters
    ----------
    samples : array-like
        X-values used for the spline

    Keywords
    --------
    Passed to the `scipy.stats.gaussian_kde` constructor

    Returns
    -------
    kdes : list of `scipy.stats.gaussian_kde` objects
    """
    return [sps.gaussian_kde(row, **kwargs) for row in samples]


def evaluate_kdes(xvals, kdes):
    """
    Build a evaluate a set of kdes

    Parameters
    ----------
    xvals : array_like
        X-values used for the spline
    kdes : list of `sps.gaussian_kde`
        The kernel density estimates

    Returns
    -------
    yvals : array_like
        The kdes evaluated at the xvals
    """
    return np.vstack([kde(xvals) for kde in kdes])
