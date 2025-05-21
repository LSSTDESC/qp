from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .array import get_eval_case, CASE_PRODUCT, CASE_FACTOR, CASE_2D, CASE_FLAT


def interpolate_multi_x_y(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, shape (npdf, n)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray
        The interpolated values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_multi_x_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_multi_x_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_multi_x_y_flat(xx, rr, xvals, yvals, **kwargs)


def interpolate_multi_x_y_product(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, length npdf
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """
    rr = np.squeeze(row)
    nx = np.shape(x)[-1]

    def single_row(rv):
        return interp1d(xvals[rv], yvals, **kwargs)(x)

    vv = np.vectorize(single_row, signature="()->(%i)" % (nx))
    return vv(rr)


def interpolate_multi_x_y_2d(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, shape (npdf, n)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, length npdf
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(),(%i)->(%i)" % (nx, nx))
    return vv(np.squeeze(row), x)


def interpolate_multi_x_y_flat(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, length n
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, length npdf
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """

    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


def interpolate_x_multi_y_product(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, length npts
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """
    rr = np.squeeze(row)
    return interp1d(xvals, yvals[rr], **kwargs)(x)


def interpolate_x_multi_y(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, shape (npdf, n)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, length npts
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray
        The interpolated values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_x_multi_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_x_multi_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_x_multi_y_flat(xx, rr, xvals, yvals, **kwargs)


def interpolate_x_multi_y_2d(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, shape (npdf, n)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, length npts
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals, yvals[rv], **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(1),(%i)->(%i)" % (nx, nx))
    return vv(row, x)


def interpolate_x_multi_y_flat(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, length n
        Which rows to interpolate at
    xvals : ArrayLike, length npts
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """

    def single_row(xv, rv):
        return interp1d(xvals, yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


def interpolate_multi_x_multi_y_flat(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, length n
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """

    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)


def interpolate_multi_x_multi_y_product(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, length n
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """
    rr = np.squeeze(row)
    nx = np.shape(x)[-1]

    def single_row(rv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(x)

    vv = np.vectorize(single_row, signature="()->(%i)" % (nx))
    return vv(rr)


def interpolate_multi_x_multi_y_2d(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, shape (npdf, n)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray, shape (npdf, n)
        The interpolated values
    """
    nx = np.shape(x)[-1]

    def evaluate_row(rv, xv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(xv)

    vv = np.vectorize(evaluate_row, signature="(),(%i)->(%i)" % (nx, nx))
    return vv(np.squeeze(row), x)


def interpolate_multi_x_multi_y(
    x: ArrayLike, row: ArrayLike, xvals: ArrayLike, yvals: ArrayLike, **kwargs
) -> np.ndarray:
    """
    Interpolate a set of values

    Parameters
    ----------
    x : ArrayLike, shape (npdf, n)
        X values to interpolate at
    row : ArrayLike, shape (npdf, 1)
        Which rows to interpolate at
    xvals : ArrayLike, shape (npdf, npts)
        X-values used for the interpolation
    yvals : ArrayLike, shape (npdf, npts)
        Y-values used for the interpolation

    Returns
    -------
    vals : np.ndarray
        The interpolated values
    """
    case_idx, xx, rr = get_eval_case(x, row)
    if case_idx in [CASE_PRODUCT, CASE_FACTOR]:
        return interpolate_multi_x_multi_y_product(xx, rr, xvals, yvals, **kwargs)
    if case_idx == CASE_2D:
        return interpolate_multi_x_multi_y_2d(xx, rr, xvals, yvals, **kwargs)
    return interpolate_multi_x_multi_y_flat(xx, rr, xvals, yvals, **kwargs)
