"""Utility functions for the qp package"""

import numpy as np

from scipy import stats as sps
from scipy.interpolate import interp1d
import sys

epsilon = sys.float_info.epsilon
infty = sys.float_info.max * epsilon
lims = (epsilon, 1.)


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
    return 0.5*(edges[1:] + edges[:-1])


def bin_widths(edges):
    """Return the widths of a set of bins given the edges"""
    return edges[1:] - edges[:-1]


def get_bin_indices(bins, x):
    """Return the bin indexes for a set of values

    If the bins are equal width this will use arithmatic,
    If the bins are not equal width this will use a binary search
    """
    widths = bin_widths(bins)
    if np.allclose(widths, widths[0]):
        idx = np.atleast_1d(np.floor((x-bins[0])/widths[0]).astype(int))
    else:
        idx = np.atleast_1d(np.searchsorted(bins, x, side='left')-1)
    mask = (idx >= 0) * (idx < bins.size-1)
    np.putmask(idx, 1-mask, 0)
    xshape = np.shape(x)
    return idx.reshape(xshape), mask.reshape(xshape)


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
    #def row_integral(irow):
    #    return quad(interp1d(xvals[irow], yvals[irow], **kwargs), limits[0], limits[1])[0]

    #vv = np.vectorize(row_integral)
    #integrals = vv(np.arange(xvals.shape[0]))
    integrals = np.sum(xvals[:,1:]*yvals[:,1:] - xvals[:,:-1]*yvals[:,1:], axis=1)
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
    return [ sps.gaussian_kde(row, **kwargs) for row in samples ]



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


def evaluate_hist_x_multi_y(x, row, bins, vals):  #pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    idx, mask = get_bin_indices(bins, x)
    mask = np.ones((row.size, 1)) * mask
    return np.where(mask.flatten(), vals[:,idx][row].flatten(), 0)


def evaluate_unfactored_hist_x_multi_y(x, row, bins, vals):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    idx, mask = get_bin_indices(bins, x)
    def evaluate_row(idxv, maskv, rv):
        return np.where(maskv, vals[rv, idxv], 0)
    vv = np.vectorize(evaluate_row)
    return vv(idx, mask, row)


def evaluate_hist_multi_x_multi_y(x, row, bins, vals):  #pragma: no cover
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N, M)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    n_vals = bins.shape[-1] - 1
    def evaluate_row(rv):
        idx, mask = get_bin_indices(bins[rv].flatten(), x)
        return np.where(mask, np.squeeze(vals[rv])[idx.clip(0, n_vals-1)], 0).flatten()
    vv = np.vectorize(evaluate_row, signature="(1)->(%i)" % (x.size))
    return vv(np.expand_dims(row, -1)).flatten()


def evaluate_unfactored_hist_multi_x_multi_y(x, row, bins, vals):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N, M)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    n_vals = bins.shape[-1] - 1
    def evaluate_row(xv, rv):
        idx, mask = get_bin_indices(bins[rv], xv)
        return np.where(mask, vals[rv,idx.clip(0, n_vals-1)], 0)
    vv = np.vectorize(evaluate_row)
    return vv(x, row)


def interpolate_unfactored_x_multi_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (N)
        X-values used for the interpolation
    yvals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    # This is kinda stupid, computes a lot of extra values, but it is vectorized
    vals = interp1d(xvals, np.squeeze(yvals[row]), **kwargs)(x)
    if vals.ndim < 2:
        return vals
    return vals.diagonal().T


def interpolate_unfactored_multi_x_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (N, M)
        X-values used for the interpolation
    yvals : array_like (N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)
    vv = np.vectorize(single_row)
    return vv(x, row)



def interpolate_unfactored_multi_x_multi_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (N, M)
        X-values used for the interpolation
    yvals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)



def interpolate_multi_x_multi_y(x, xvals, yvals, **kwargs):  #pragma: no cover
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_line (n)
        X values to interpolate at:
    xvals : array_like (M, N)
        X-values used for the interpolation
    yvals : array_like (M, N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    xy_vals = np.hstack([xvals, yvals])
    nx = xvals.shape[-1]
    def single_row(xy_vals_):
        return interp1d(xy_vals_[0:nx], xy_vals_[nx:], **kwargs)(x)
    vv = np.vectorize(single_row, signature="(%i)->(%i)" % (xvals.shape[-1]+yvals.shape[-1], x.size))
    return vv(xy_vals)


def interpolate_x_multi_y(x, xvals, yvals, **kwargs):  #pragma: no cover
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_line (n)
        X values to interpolate at:
    xvals : array_like (N)
        X-values used for the interpolation
    yvals : array_like (M, N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    return interp1d(xvals, yvals, **kwargs)(x)



def interpolate_multi_x_y(x, xvals, yvals, **kwargs):  #pragma: no cover
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_line (n)
        X values to interpolate at:
    xvals : array_like (M, N)
        X-values used for the interpolation
    yvals : array_like (N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    #def single_row(xrow):
    #    idx = np.searchsorted(xrow, x, side='left').clip(1, xrow.size-1)
    #    x0 = xrow[idx-1]
    #    x1 = xrow[idx]
    #    f = (x1 - x)/(x1 - x0)
    #    y0 = yvals[idx-1]
    #    y1 = yvals[idx]
    #    return f*y1 + (1 - f)*y0
    def single_row(xrow):
        return interp1d(xrow, yvals, **kwargs)(x)
    vv = np.vectorize(single_row, signature="(%i)->(%i)" % (yvals.size, x.size))
    return vv(xvals)


def profile(x_data, y_data, x_bins, std=True):
    """Make a 'profile' plot

    Paramters
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
    count = np.zeros(x_bins.size-1)
    vals = np.zeros(x_bins.size-1)
    errs = np.zeros(x_bins.size-1)
    for i in range(x_bins.size-1):
        mask_col = mask * (idx == i)
        count[i] = mask_col.sum()
        if mask_col.sum() == 0:  #pragma: no cover
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
    in_shape = vals.shape
    npdf = np.product(in_shape[:split_dim])
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
