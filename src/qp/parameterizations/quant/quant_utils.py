from __future__ import annotations

import sys
import numpy as np
from numpy.typing import ArrayLike

from ...utils.array import (
    get_eval_case,
    CASE_2D,
    CASE_FLAT,
    CASE_FACTOR,
    CASE_PRODUCT,
    get_bin_indices,
)

# Conversion functions


def extract_quantiles(in_dist: "Ensemble", **kwargs) -> dict[str, np.ndarray[float]]:
    """Convert using a set of quantiles and the locations at which they are reached

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    quants : np.ndarray
        Quantile values to use

    Returns
    -------
    data : dict[str, Any]
        The extracted data
    """
    quants = kwargs.pop("quants", None)
    if quants is None:  # pragma: no cover
        raise ValueError("To convert using extract_quantiles you must specify quants")
    locs = in_dist.ppf(quants)
    return dict(quants=quants, locs=locs)


# Creation functions


def pad_quantiles(
    quants: ArrayLike, locs: ArrayLike
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Pad the quantiles and locations used to build a quantile representation.
    Ensuring 0 and 1 are part of quantiles.
    Extrapolates loc at 0 by taking a linear extrapolation from the first two points
    and following to where it intersects 0
    Extrapolates loc at 1 by taking a linear extrapolation from the last two points
    and following to where it intersects 1

    This will add additional data points to the quants and locs


    Parameters
    ----------
    quants : ArrayLike
        The quantiles used to build the CDF
    locs : ArrayLike
        The locations at which those quantiles are reached

    Returns
    -------
    quants : np.ndarray[float]
        The quantiles used to build the CDF
    locs : np.ndarray[float]
        The locations at which those quantiles are reached
    """
    n_out = n_vals = quants.size
    if quants[0] > sys.float_info.epsilon:
        offset_lo = 1
        pad_lo = True
        n_out += 1
    else:
        offset_lo = 0
        pad_lo = False
    if quants[-1] < 1.0:
        pad_hi = True
        n_out += 1
    else:
        pad_hi = False
    if n_out == n_vals:
        return quants, locs
    quants_out = np.zeros((n_out), quants.dtype)
    locs_out = np.zeros((locs.shape[0], n_out), quants.dtype)
    quants_out[offset_lo : n_vals + offset_lo] = quants
    locs_out[:, offset_lo : n_vals + offset_lo] = locs
    if pad_lo:
        locs_out[:, 0] = locs[:, 0] - quants[0] * (locs[:, 1] - locs[:, 0]) / (
            quants[1] - quants[0]
        )

    if pad_hi:
        quants_out[-1] = 1.0
        locs_out[:, -1] = locs[:, -1] - (1.0 - quants[-1]) * (
            locs[:, -2] - locs[:, -1]
        ) / (quants[-1] - quants[-2])

    return quants_out, locs_out


def evaluate_hist_multi_x_multi_y(
    x: ArrayLike, row: ArrayLike, bins: ArrayLike, vals: ArrayLike, derivs=None
) -> np.ndarray[float]:
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : ArrayLike
        X values to interpolate at
    row : ArrayLike
        Which rows to interpolate at
    bins : ArrayLike, shape (npdf, N+1)
        'x' bin edges
    vals : ArrayLike, shape (npdf, N)
        'y' bin contents

    Returns
    -------
    out : np.ndarray[float]
        The histogram values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return evaluate_hist_multi_x_multi_y_product(xx, rr, bins, vals, derivs)
    if case_idx == CASE_2D:
        return evaluate_hist_multi_x_multi_y_2d(xx, rr, bins, vals, derivs)
    return evaluate_hist_multi_x_multi_y_flat(xx, rr, bins, vals, derivs)


def evaluate_hist_multi_x_multi_y_flat(
    x: ArrayLike, row: ArrayLike, bins: ArrayLike, vals: ArrayLike, derivs=None
) -> np.ndarray[float]:  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, length n
        Which rows to interpolate at
    bins : ArrayLike, shape (npdf, N+1)
        'x' bin edges
    vals : ArrayLike, shape (npdf, N)
        'y' bin contents

    Returns
    -------
    out : np.ndarray[float], length n
        The histogram values
    """

    def evaluate_row(xv, rv):
        bins_row = bins[rv]
        idx, mask = get_bin_indices(bins_row, xv)
        delta = xv - bins_row[idx]
        if derivs is None:
            return np.where(mask, vals[rv, idx], 0)
        return np.where(mask, vals[rv, idx] + delta * derivs[rv, idx], 0)

    vv = np.vectorize(evaluate_row)
    return vv(x, row)


def evaluate_hist_multi_x_multi_y_product(
    x: ArrayLike, row: ArrayLike, bins: ArrayLike, vals: ArrayLike, derivs=None
) -> np.ndarray[float]:  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : ArrayLike, length npts
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    bins : ArrayLike, shape (npdf, N+1)
        'x' bin edges
    vals : ArrayLike, shape (npdf, N)
        'y' bin contents

    Returns
    -------
    out : np.ndarray[float], shape (npdf, npts)
        The histogram values
    """

    def evaluate_row(rv):
        bins_flat = bins[rv].flatten()
        idx, mask = get_bin_indices(bins_flat, x)
        delta = x - bins_flat[idx]
        if derivs is None:
            return np.where(mask, np.squeeze(vals[rv])[idx], 0).flatten()
        return np.where(
            mask, np.squeeze(vals[rv])[idx] + delta * np.squeeze(derivs[rv])[idx], 0
        )

    vv = np.vectorize(evaluate_row, signature="(1)->(%i)" % (x.size))
    return vv(row)


def evaluate_hist_multi_x_multi_y_2d(
    x: ArrayLike, row: ArrayLike, bins: ArrayLike, vals: ArrayLike, derivs=None
) -> np.ndarray[float]:  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : ArrayLike, shape (npdf, npts)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    bins : ArrayLike, shape (npdf, N+1)
        'x' bin edges
    vals : ArrayLike, shape (npdf, N)
        'y' bin contents

    Returns
    -------
    out : np.ndarray[float], shape (npdf, npts)
        The histogram values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        flat_bins = bins[rv].flatten()
        idx, mask = get_bin_indices(flat_bins, xv)
        delta = xv - flat_bins[idx]
        if derivs is None:
            return np.where(mask, np.squeeze(vals[rv])[idx], 0).flatten()
        return np.where(
            mask, np.squeeze(vals[rv])[idx] + delta * np.squeeze(derivs[rv])[idx], 0
        ).flatten()

    vv = np.vectorize(evaluate_row, signature="(1),(%i)->(%i)" % (nx, nx))
    return vv(row, x)
