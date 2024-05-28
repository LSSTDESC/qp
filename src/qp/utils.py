"""Utility functions for the qp package"""

import sys

import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d

epsilon = sys.float_info.epsilon
infty = sys.float_info.max * epsilon
lims = (epsilon, 1.0)

CASE_PRODUCT = 0
CASE_FACTOR = 1
CASE_2D = 2
CASE_FLAT = 3


def safelog(arr, threshold=epsilon):
    """
    Takes the natural logarithm of an array of potentially non-positive numbers

    Parameters
    ----------
    arr: numpy.ndarray, float
        values to be logged
    threshold: float
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: numpy.ndarray
        logarithms, with approximation in place of zeros and negative numbers
    """
    return np.log(np.array(arr).clip(threshold, np.inf))


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


def edge_to_center(edges):
    """Return the centers of a set of bins given the edges"""
    return 0.5 * (edges[1:] + edges[:-1])


def bin_widths(edges):
    """Return the widths of a set of bins given the edges"""
    return edges[1:] - edges[:-1]


def get_bin_indices(bins, x):
    """Return the bin indexes for a set of values

    If the bins are equal width this will use arithmatic,
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


def normalize_interp1d(xvals, yvals):
    """
    Normalize a set of 1D interpolators

    Parameters
    ----------
    xvals : array-like
        X-values used for the interpolation
    yvals : array-like
        Y-values used for the interpolation

    Returns
    -------
    ynorm: array-like
        Normalized y-vals
    """
    # def row_integral(irow):
    #    return quad(interp1d(xvals[irow], yvals[irow], **kwargs), limits[0], limits[1])[0]

    # vv = np.vectorize(row_integral)
    # integrals = vv(np.arange(xvals.shape[0]))
    integrals = np.sum(
        xvals[:, 1:] * yvals[:, 1:] - xvals[:, :-1] * yvals[:, 1:], axis=1
    )
    return (yvals.T / integrals).T


def build_kdes(samples, **kwargs):
    """
    Build a set of Gaussian Kernal Density Estimates

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
        The kdes evaluated at the xvamls
    """
    return np.vstack([kde(xvals) for kde in kdes])


def get_eval_case(x, row):
    """Figure out which of the various input formats scipy.stats has passed us

    Parameters
    ----------
    x : array_like
        Pdf x-vals
    row : array_like
        Pdf row indices

    Returns
    -------
    case : `int`
        The case code
    xx : array_like
        The x-values properly shaped
    rr : array_like
        The y-values, properly shaped

    Notes
    -----
    The cases are:

    CASE_FLAT : x, row have shapes (n), (n) and do not factor
    CASE_FACTOR : x, row have shapes (n), (n) but can be factored to shapes (1, nx) and (npdf, 1)
                  (i.e., they were flattend by scipy)
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


def evaluate_hist_x_multi_y_flat(x, row, bins, vals, derivs=None):  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like (n)
        The histogram values
    """
    assert np.ndim(x) < 2 and np.ndim(row) < 2
    idx, mask = get_bin_indices(bins, x)
    if derivs is None:
        deltas = np.zeros(idx.shape)
    else:
        deltas = x - bins[idx]

    def evaluate_row(idxv, maskv, rv, delta):
        if derivs is None:
            return np.where(maskv, vals[rv, idxv], 0)
        return np.where(maskv, vals[rv, idxv] + delta * derivs[rv, idxv], 0)

    vv = np.vectorize(evaluate_row)
    return vv(idx, mask, row, deltas)


def evaluate_hist_x_multi_y_product(
    x, row, bins, vals, derivs=None
):  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (npts)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    bins : array_like (N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like (npdf, npts)
        The histogram values
    """
    # assert np.ndim(x) < 2 and np.ndim(row) == 2
    idx, mask0 = get_bin_indices(bins, x)
    mask = np.ones(row.shape) * mask0
    if derivs is None:
        return np.where(mask, vals[:, idx][np.squeeze(row)], 0)
    deltas = x - bins[idx]
    return np.where(
        mask,
        vals[:, idx][np.squeeze(row)] + deltas * derivs[:, idx][np.squeeze(row)],
        0,
    )


def evaluate_hist_x_multi_y_2d(x, row, bins, vals, derivs=None):  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (npdf, npts)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    bins : array_like (N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like (npdf, npts)
        The histogram values
    """
    assert np.ndim(x) >= 2 and np.ndim(row) >= 2
    idx, mask = get_bin_indices(bins, x)
    if derivs is None:
        deltas = np.zeros(idx.shape)
    else:
        deltas = x - bins[idx]

    def evaluate_row(idxv, maskv, rv, delta):
        if derivs is None:
            return np.where(maskv, vals[rv, idxv], 0)
        return np.where(maskv, vals[rv, idxv] + delta * derivs[rv, idxv], 0)

    vv = np.vectorize(evaluate_row)
    return vv(idx, mask, row, deltas)


def evaluate_hist_x_multi_y(x, row, bins, vals, derivs=None):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like
        X values to interpolate at
    row : array_like
        Which rows to interpolate at
    bins : array_like (N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like
        The histogram values

    Notes
    -----
    Depending on the shape of 'x' and 'row' this will
    use one of the three specific implementations.
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return evaluate_hist_x_multi_y_product(xx, rr, bins, vals, derivs)
    if case_idx == CASE_2D:
        return evaluate_hist_x_multi_y_2d(xx, rr, bins, vals, derivs)
    return evaluate_hist_x_multi_y_flat(xx, rr, bins, vals, derivs)


def evaluate_hist_multi_x_multi_y_flat(
    x, row, bins, vals, derivs=None
):  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (npdf, N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like (n)
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
    x, row, bins, vals, derivs=None
):  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (npts)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    bins : array_like (npdf, N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like (npdf, npts)
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
    x, row, bins, vals, derivs=None
):  # pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (npdf, npts)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    bins : array_like (npdf, N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like (npdf, npts)
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


def evaluate_hist_multi_x_multi_y(x, row, bins, vals, derivs=None):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like
        X values to interpolate at
    row : array_like
        Which rows to interpolate at
    bins : array_like (npdf, N+1)
        'x' bin edges
    vals : array_like (npdf, N)
        'y' bin contents

    Returns
    -------
    out : array_like
        The histogram values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return evaluate_hist_multi_x_multi_y_product(xx, rr, bins, vals, derivs)
    if case_idx == CASE_2D:
        return evaluate_hist_multi_x_multi_y_2d(xx, rr, bins, vals, derivs)
    return evaluate_hist_multi_x_multi_y_flat(xx, rr, bins, vals, derivs)


def interpolate_x_multi_y_flat(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """

    def single_row(xv, rv):
        return interp1d(xvals, yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


def interpolate_x_multi_y_product(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """
    rr = np.squeeze(row)
    return interp1d(xvals, yvals[rr], **kwargs)(x)


def interpolate_x_multi_y_2d(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (npdf, n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals, yvals[rv], **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(1),(%i)->(%i)" % (nx, nx))
    return vv(row, x)


def interpolate_x_multi_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (npdf, n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like
        The interpoalted values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_x_multi_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_x_multi_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_x_multi_y_flat(xx, rr, xvals, yvals, **kwargs)


def interpolate_multi_x_multi_y_flat(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """

    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


def interpolate_multi_x_multi_y_product(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """
    rr = np.squeeze(row)
    nx = np.shape(x)[-1]

    def single_row(rv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(x)

    vv = np.vectorize(single_row, signature="()->(%i)" % (nx))
    return vv(rr)


def interpolate_multi_x_multi_y_2d(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (npdf, n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(),(%i)->(%i)" % (nx, nx))
    return vv(np.squeeze(row), x)


def interpolate_multi_x_multi_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (npdf, n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf, npts)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like
        The interpoalted values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_multi_x_multi_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_multi_x_multi_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_multi_x_multi_y_flat(xx, rr, xvals, yvals, **kwargs)


def interpolate_multi_x_y_flat(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """

    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


def interpolate_multi_x_y_product(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """
    rr = np.squeeze(row)
    nx = np.shape(x)[-1]

    def single_row(rv):
        return interp1d(xvals[rv], yvals, **kwargs)(x)

    vv = np.vectorize(single_row, signature="()->(%i)" % (nx))
    return vv(rr)


def interpolate_multi_x_y_2d(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (npdf, n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpoalted values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(),(%i)->(%i)" % (nx, nx))
    return vv(np.squeeze(row), x)


def interpolate_multi_x_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (npdf, n)
        X values to interpolate at
    row : array_like (npdf, 1)
        Which rows to interpolate at
    xvals : array_like (npdf, npts)
        X-values used for the interpolation
    yvals : array_like (npdf)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like
        The interpoalted values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_multi_x_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_multi_x_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_multi_x_y_flat(xx, rr, xvals, yvals, **kwargs)


def profile(x_data, y_data, x_bins, std=True):
    """Make a 'profile' plot

    Parameters
    ---------
    x_data : array_like (n)
        The x-values
    y_data : array_like (n)
        The y-values
    x_bins : array_like (nbins+1)
        The values of the bin edges
    std : bool
        If true, return the standard deviations, if false return the errors on the means

    Returns
    -------
    vals : array_like (nbins)
        The means
    errs : array_like (nbins)
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


def reshape_to_pdf_size(vals, split_dim):
    """Reshape an array to match the number of PDFs in a distribution

    Parameters
    ----------
    vals : array
        The input array
    split_dim : int
        The dimension at which to split between pdf indices and per_pdf indices

    Returns
    -------
    out : array
        The reshaped array
    """
    in_shape = np.shape(vals)
    npdf = np.prod(in_shape[:split_dim]).astype(int)
    per_pdf = in_shape[split_dim:]
    out_shape = np.hstack([npdf, per_pdf])
    return vals.reshape(out_shape)


def reshape_to_pdf_shape(vals, pdf_shape, per_pdf):
    """Reshape an array to match the shape of PDFs in a distribution

    Parameters
    ----------
    vals : array
        The input array
    pdf_shape : int
        The shape for the pdfs
    per_pdf : int or array_like
        The shape per pdf

    Returns
    -------
    out : array
        The reshaped array
    """
    outshape = np.hstack([pdf_shape, per_pdf])
    return vals.reshape(outshape)
