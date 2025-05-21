"""Utility functions for array handling in the the qp package"""

from __future__ import annotations

import sys

import numpy as np
from typing import Mapping, Union
from numpy.typing import ArrayLike

# epsilon = sys.float_info.epsilon
# infty = sys.float_info.max * epsilon
# lims = (epsilon, 1.0)

CASE_PRODUCT = 0
CASE_FACTOR = 1
CASE_2D = 2
CASE_FLAT = 3


_ = """
def normalize_quantiles(in_data, threshold=epsilon, vb=False):
    Evaluates PDF from quantiles including endpoints from linear extrapolation

    Parameters
    ----------
    in_data: tuple, numpy.ndarray, float
        tuple of CDF values iy corresponding to quantiles and the points x at
        which those CDF values are achieved
    threshold: float, optional
        optional minimum threshold for PDF
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, ndarray, float
        tuple of values x at which CDF is achieved, including extrema, and
        normalized PDF values y at x

    (iy, x) = in_data
    (xs, ys) = evaluate_quantiles((iy, x), vb=vb)
    # xs = xs[1:-1]
    # ys = ys[1:-1]
    x_min = xs[0] - 2 * iy[0] / ys[0]
    x_max = xs[-1] + 2 * (1. - iy[-1]) / ys[-1]
    xs = sandwich(xs, (x_min, x_max))
    ys = sandwich(ys, (threshold, threshold))
    out_data = (xs, ys)
    return out_data

"""


def edge_to_center(edges: ArrayLike) -> np.ndarray:
    """Return the centers of a set of bins given the edges"""
    return 0.5 * (edges[1:] + edges[:-1])


def bin_widths(edges: ArrayLike) -> np.ndarray:
    """Return the widths of a set of bins given the edges"""
    return edges[1:] - edges[:-1]


def get_bin_indices(bins: ArrayLike, x: ArrayLike) -> np.ndarray[int]:
    """Return the bin indexes for a set of values

    If the bins are equal width this will use arithmetic,
    If the bins are not equal width this will use a binary search
    """
    widths = bin_widths(bins)
    n_bins = np.size(bins) - 1
    if np.allclose(widths, widths[0]):
        idx = np.atleast_1d(np.floor((x - bins[0]) / widths[0]).astype(int))
    else:
        idx = np.atleast_1d(np.searchsorted(bins, x, side="left") - 1)
    mask = (idx >= 0) * (idx < bins.size - 1)
    np.putmask(idx, 1 - mask, 0)
    xshape = np.shape(x)
    return idx.reshape(xshape).clip(0, n_bins - 1), mask.reshape(xshape)


def get_eval_case(x: ArrayLike, row: ArrayLike) -> tuple[int, np.ndarray, np.ndarray]:
    """Figure out which of the various input formats scipy.stats has passed us

    Parameters
    ----------
    x : ArrayLike
        Pdf x-vals
    row : ArrayLike
        Pdf row indices

    Returns
    -------
    case : int
        The case code
    xx : np.ndarray
        The x-values properly shaped
    rr : np.ndarrray
        The y-values, properly shaped

    Notes
    -----
    The cases are:

    CASE_FLAT : x, row have shapes (n), (n) and do not factor
    CASE_FACTOR : x, row have shapes (n), (n) but can be factored to shapes (1, nx) and (npdf, 1)
                  (i.e., they were flattened by scipy)
    CASE_PRODUCT : x, row have shapes (1, nx) and (npdf, 1)
    CASE_2D : x, row have shapes (npdf, nx) and (npdf, nx)

    """
    nd_x = np.ndim(x)
    nd_row = np.ndim(row)
    # if nd_x > 2 or nd_row > 2:  #pragma: no cover
    #    raise ValueError("Too many dimensions: x(%s), row(%s)" % (np.shape(x), np.shape(row)))
    if nd_x >= 2 and nd_row != 1:
        return CASE_2D, x, row
    if nd_x >= 2 and nd_row == 1:  # pragma: no cover
        raise ValueError(
            "Dimension mismatch: x(%s), row(%s)" % (np.shape(x), np.shape(row))
        )
    if nd_row >= 2:
        return CASE_PRODUCT, x, row
    if np.size(x) == 1 or np.size(row) == 1:
        return CASE_FLAT, x, row
    xx = np.unique(x)
    rr = np.unique(row)
    if np.size(xx) == np.size(x):
        xx = x
    if np.size(rr) == np.size(row):
        rr = row
    if np.size(xx) * np.size(rr) != np.size(x):
        return CASE_FLAT, x, row
    return CASE_FACTOR, xx, np.expand_dims(rr, -1)


def profile(
    x_data: ArrayLike, y_data: ArrayLike, x_bins: ArrayLike, std: bool = True
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Make a 'profile' plot

    Parameters
    ----------
    x_data : ArrayLike, length n
        The x-values
    y_data : ArrayLike, length n
        The y-values
    x_bins : ArrayLike, length nbins+1
        The values of the bin edges
    std : bool, optional
        If true, return the standard deviations, if false return the errors on the
        means, default True.

    Returns
    -------
    vals : np.ndarray[float], length nbins
        The means
    errs : np.ndarray[float], length nbins
        The standard deviations or errors on the means
    """
    idx, mask = get_bin_indices(x_bins, x_data)
    count = np.zeros(x_bins.size - 1)
    vals = np.zeros(x_bins.size - 1)
    errs = np.zeros(x_bins.size - 1)
    for i in range(x_bins.size - 1):
        mask_col = mask * (idx == i)
        count[i] = mask_col.sum()
        if mask_col.sum() == 0:  # pragma: no cover
            vals[i] = np.nan
            errs[i] = np.nan
            continue
        masked_vals = y_data[mask_col]
        vals[i] = masked_vals.mean()
        errs[i] = masked_vals.std()
    if not std:
        errs /= np.sqrt(count)
    return vals, errs


def reshape_to_pdf_size(vals: np.ndarray, split_dim: int) -> np.ndarray:
    """Reshape an array to match the number of PDFs in a distribution

    Parameters
    ----------
    vals : np.ndarray
        The input array
    split_dim : int
        The dimension at which to split between pdf indices and per_pdf indices

    Returns
    -------
    out : np.ndarray
        The reshaped array
    """
    in_shape = np.shape(vals)
    npdf = np.prod(in_shape[:split_dim]).astype(int)
    per_pdf = in_shape[split_dim:]
    out_shape = np.hstack([npdf, per_pdf])
    return vals.reshape(out_shape)


def reshape_to_pdf_shape(
    vals: np.ndarray, pdf_shape: int, per_pdf: int | ArrayLike
) -> np.ndarray:
    """Reshape an array to match the shape of PDFs in a distribution

    Parameters
    ----------
    vals : np.ndarray
        The input array
    pdf_shape : int
        The shape for the pdfs
    per_pdf : int | ArrayLike
        The shape per pdf

    Returns
    -------
    out : np.ndarray
        The reshaped array
    """
    outshape = np.hstack([pdf_shape, per_pdf])
    return vals.reshape(outshape)


def encode_strings(data: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Encodes any dictionary values that are Unicode strings (or just strings
    if not numpy arrays). Other data types are not affected.

    Parameters
    ----------
    data : Mapping[str, np.ndarray]
        Dictionary of data to encode.

    Returns
    -------
    Mapping[str, np.ndarray]
        Dictionary of data with strings encoded.
    """

    converted_data = {}
    for key, val in data.items():
        new_val = val
        if isinstance(val, np.ndarray):
            # encode unicode strings as bytes to work with hdf5
            if val.dtype.kind == "U":
                new_val = np.strings.encode(val, "utf-8")
        else:
            # is not a numpy array
            if isinstance(val[0], str):
                new_val = np.strings.encode(val, "utf-8")

        converted_data[key] = new_val

    return converted_data


def decode_strings(data: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Decodes dictionary values that have been encoded (dtype = bytes). Other
    data types are not affected.

    Parameters
    ----------
    data : Mapping[str, np.ndarray]
        The dictionary of data to be decoded.

    Returns
    -------
    Mapping[str, np.ndarray]
        The dictionary of data with any strings decoded.
    """
    converted_data = {}
    for key, val in data.items():
        new_val = val

        if isinstance(val, np.ndarray):
            # decode any string objects as necessary
            if val.dtype.kind == "S":
                new_val = np.strings.decode(val, "utf-8")
        else:
            # decode string objects that are not numpy arrays
            if isinstance(val[0], bytes):
                new_val = np.strings.decode(val, "utf-8")

        converted_data[key] = new_val

    return converted_data


def reduce_dimensions(arr: np.ndarray, x: ArrayLike) -> Union[float, np.ndarray]:
    """If the given array has dimensionality greater than x, reduces its dimensionality
    to match x, if this will not result in a loss of data.

    Parameters
    ----------
    arr : np.ndarray
        Array to reduce dimensionality
    x : ArrayLike
        Object to match dimensionality to

    Returns
    -------
    Union[float, np.ndarray]
        The array with dimension reduced (if possible)
    """
    if np.ndim(x) < 1 and np.ndim(arr) >= 2:
        if np.shape(arr) == (1, 1):
            return arr.item()
    if np.ndim(x) == 1 and np.ndim(arr) > 1:
        if arr.shape[0] == 1:
            return np.squeeze(arr)
    return arr
