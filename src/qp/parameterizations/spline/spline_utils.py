import numpy as np
from scipy import stats as sps
from scipy.integrate import quad
from scipy.interpolate import splev, splrep
from scipy.special import errstate  # pylint: disable=no-name-in-module

from ...utils.conversion import extract_xy_vals


def normalize_spline(xvals, yvals, limits, **kwargs):
    """
    Normalize a set of 1D interpolators

    Parameters
    ----------
    xvals : array-like
        X-values used for the spline
    yvals : array-like
        Y-values used for the spline
    limits : tuple (2)
        Lower and Upper limits of integration

    Keywords
    --------
    Passed to the `scipy.quad` integration function

    Returns
    -------
    ynorm: array-like
        Normalized y-vals
    """

    def row_integral(irow):
        def spl(xv):
            return splev(xv, splrep(xvals[irow], yvals[irow]))

        return quad(spl, limits[0], limits[1], **kwargs)[0]

    vv = np.vectorize(row_integral)
    with errstate(all="ignore"):
        integrals = vv(np.arange(xvals.shape[0]))
    return (yvals.T / integrals).T


def build_splines(xvals, yvals):
    """
    Build a set of 1D spline representations

    Parameters
    ----------
    xvals : array-like
        X-values used for the spline
    yvals : array-like
        Y-values used for the spline

    Returns
    -------
    splx : array-like
        Spline knot xvalues
    sply : array-like
        Spline knot yvalues
    spln : array-like
        Spline knot order parameters
    """
    l_x = []
    l_y = []
    l_n = []
    for xrow, yrow in zip(xvals, yvals):
        rep = splrep(xrow, yrow)
        l_x.append(rep[0])
        l_y.append(rep[1])
        l_n.append(rep[2])
    return np.vstack(l_x), np.vstack(l_y), np.vstack(l_n)


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
