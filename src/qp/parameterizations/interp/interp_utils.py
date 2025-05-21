from __future__ import annotations

from typing import Any
import numpy as np
from scipy import integrate as sciint
from scipy import interpolate as sciinterp

from ...utils.conversion import extract_xy_vals
from ..sparse_interp.sparse_rep import (
    build_sparse_representation,
    decode_sparse_indices,
)


# Conversion functions


def irreg_interp_extract_xy_vals(in_dist: "Ensemble", **kwargs):  # -> dict[str, Any]:
    """Wrapper for extract_xy_vals. Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions
    xvals : np.ndarray[float]
        Locations at which the pdf is evaluated

    Returns
    -------
    data : dict
        The extracted data"""

    xvals = kwargs.pop("xvals", None)
    if xvals is None:  # pragma: no cover
        raise ValueError("To convert using extract_xy_vals you must specify xvals")
    return extract_xy_vals(in_dist, xvals)


def extract_vals_at_x(in_dist: "Ensemble", **kwargs) -> dict[str, np.ndarray[float]]:
    """Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    xvals : np.ndarray[float]
        Locations at which the pdf is evaluated

    Returns
    -------
    data : dict[str, np.ndarray[float]]
        The extracted data
    """
    xvals = kwargs.pop("xvals", None)
    if xvals is None:  # pragma: no cover
        raise ValueError("To convert to extract_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    return dict(xvals=xvals, yvals=yvals)


def extract_xy_sparse(
    in_dist: "Ensemble", **kwargs
) -> dict[str, Any]:  # pragma: no cover
    """Extract xy-interpolated representation from an sparse representation

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    xvals : ArrayLike
        Used to override the y-values
    xvals : ArrayLike
        Used to override the x-values
    nvals : int
        Used to override the number of bins

    Returns
    -------
    metadata : dict[str, Any]
        Dictionary with data for interpolated representation

    Notes
    -----
    This function will rebin to a grid more suited to the in_dist support by
    removing x-values corresponding to y=0
    """

    yvals = in_dist.objdata["yvals"]
    default = in_dist.metadata["xvals"][0]
    xvals = kwargs.pop("xvals", default)
    nvals = kwargs.pop("nvals", 300)
    # rebin to a grid more suited to the in_dist support
    xmin = np.min(xvals)
    _, j = np.where(yvals > 0)
    xmax = np.max(xvals[j])
    newx = np.linspace(xmin, xmax, nvals)
    interp = sciinterp.interp1d(xvals, yvals, assume_sorted=True)
    newpdf = interp(newx)
    sparse_indices, sparse_meta, A = build_sparse_representation(newx, newpdf)
    # decode the sparse indices into basis indices and weights
    basis_indices, weights = decode_sparse_indices(sparse_indices)
    # retrieve the weighted array of basis functions for each object
    pdf_y = A[:, basis_indices] * weights
    # normalize and sum the weighted pdfs
    x = sparse_meta["z"]
    y = pdf_y.sum(axis=-1)
    norms = sciint.trapezoid(y.T, x)
    y /= norms
    # super(sparse_gen, self).__init__(x, y.T, *args, **kwargs)
    xvals = x
    yvals = y.T
    return dict(xvals=xvals, yvals=yvals, **kwargs)
