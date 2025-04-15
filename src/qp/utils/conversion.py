"""This module implements functions to convert distributions between various representations
These functions should then be registered with the `qp.ConversionDict` using `qp_add_mapping`.
That will allow the automated conversion mechanisms to work.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as sciint
from scipy import interpolate as sciinterp

from ..parameterizations.sparse_interp.sparse_rep import (
    build_sparse_representation,
    indices2shapes,
)


def extract_xy_vals(in_dist: "Ensemble", xvals: np.ndarray) -> dict[str, np.ndarray]:
    """Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions
    xvals : np.ndarray
        Locations at which the pdf is evaluated

    Returns
    -------
    data : dict[str, np.ndarray]
        The extracted data
    """

    yvals = in_dist.pdf(xvals)
    expand_x = np.ones(yvals.shape) * np.squeeze(xvals)
    return dict(xvals=expand_x, yvals=yvals)


#
# Unused conversion functions -- not tested, use at own risk
#


def extract_fit(in_dist, **kwargs):  # pragma: no cover
    """Convert to a functional distribution by fitting it to a set of x and y values

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : dict
        The extracted data
    """
    raise NotImplementedError("extract_fit")
    # xvals = kwargs.pop('xvals', None)
    # if xvals is None:
    #   raise ValueError("To convert using extract_fit you must specify xvals")
    ##vals = in_dist.pdf(xvals)


def extract_voigt_mixmod(in_dist, **kwargs):  # pragma: no cover
    """Convert to a voigt mixture model starting with a gaussian mixture model,
    trivially by setting gammas to 0

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Returns
    -------
    data : dict
        The extracted data
    """
    objdata = in_dist.objdata
    means = objdata["means"]
    stds = objdata["stds"]
    weights = objdata["weights"]
    gammas = np.zeros_like(means)
    return dict(means=means, stds=stds, weights=weights, gammas=gammas, **kwargs)


def extract_voigt_xy(in_dist, **kwargs):  # pragma: no cover
    """Build a voigt function basis and run a match-pursuit algorithm to fit gridded data

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Returns
    -------
    data : dict
        The extracted data as sparse indices, basis, and metadata to rebuild the basis
    """

    sparse_results = extract_voigt_xy_sparse(in_dist, **kwargs)
    indices = sparse_results["indices"]
    meta = sparse_results["metadata"]

    w, m, s, g = indices2shapes(indices, meta)
    return dict(means=m, stds=s, weights=w, gammas=g)


def extract_voigt_xy_sparse(in_dist, **kwargs):  # pragma: no cover
    """Build a voigt function basis and run a match-pursuit algorithm to fit gridded data

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Returns
    -------
    data : dict
        The extracted data as shaped parameters means, stds, weights, gammas
    """

    yvals = in_dist.objdata["yvals"]

    default = in_dist.metadata["xvals"][0]
    z = kwargs.pop("xvals", default)
    nz = kwargs.pop("nz", 300)

    minz = np.min(z)
    _, j = np.where(yvals > 0)
    maxz = np.max(z[j])
    newz = np.linspace(minz, maxz, nz)
    interp = sciinterp.interp1d(z, yvals, assume_sorted=True)
    newpdf = interp(newz)
    newpdf = newpdf / sciint.trapezoid(newpdf, newz).reshape(-1, 1)
    ALL, bigD, _ = build_sparse_representation(newz, newpdf)
    return dict(indices=ALL, metadata=bigD)
