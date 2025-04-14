from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .array import get_eval_case, CASE_PRODUCT, CASE_FACTOR, CASE_2D, CASE_FLAT


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like
        The interpolated values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_multi_x_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_multi_x_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_multi_x_y_flat(xx, rr, xvals, yvals, **kwargs)


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(),(%i)->(%i)" % (nx, nx))
    return vv(np.squeeze(row), x)


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
    """

    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)

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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
    """
    rr = np.squeeze(row)
    return interp1d(xvals, yvals[rr], **kwargs)(x)


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like
        The interpolated values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_x_multi_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_x_multi_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_x_multi_y_flat(xx, rr, xvals, yvals, **kwargs)


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals, yvals[rv], **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(1),(%i)->(%i)" % (nx, nx))
    return vv(row, x)


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
    """

    def single_row(xv, rv):
        return interp1d(xvals, yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like (npdf, n)
        The interpolated values
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
        Y-values used for the interpolation

    Returns
    -------
    vals : array_like
        The interpolated values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_multi_x_multi_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_multi_x_multi_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_multi_x_multi_y_flat(xx, rr, xvals, yvals, **kwargs)
