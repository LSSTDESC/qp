import numpy as np
from scipy import stats as sps
import sys

global epsilon
epsilon = sys.float_info.epsilon
global infty
infty = sys.float_info.max * epsilon
global lims
lims = (epsilon, 1.)

def sandwich(in_arr, ends):
    """
    Adds given values to the ends of a 1D array

    Parameters
    ----------
    in_arr: numpy.ndarray, float
        original array
    ends: numpy.ndarray or tuple or list, float or numpy.ndarray, float
        values to be added to the beginning and end

    Returns
    -------
    out_arr: numpy.ndarray, float
        array with front and back concatenations
    """
    if isinstance(ends[0], np.ndarray):
        prepend = len(ends[0])
    else:
        prepend = 1
    if isinstance(ends[-1], np.ndarray):
        append = -1 * len(ends[-1])
    else:
        append = -1
    out_arr = np.zeros(prepend + len(in_arr) - append)
    out_arr[:prepend] = ends[0]
    out_arr[prepend:append] = in_arr
    out_arr[append:] = ends[-1]
    return out_arr

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
    shape = np.shape(arr)
    flat = arr.flatten()
    logged = np.log(np.array([max(a, threshold) for a in flat])).reshape(shape)
    return logged

def normalize_integral(in_data, vb=False):
    """
    Normalizes integrals of PDF evaluations on a grid

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of points x at which function is evaluated and the PDF y at those
        points
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        tuple of ordered input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    # if vb:
    a = x.argsort()
    #     try:
    #         assert np.array_equal(x[a], x.sort())
    #     except AssertionError:
    x.sort()
    y = y[a]
    dx = x[1:] - x[:-1]
    my = (y[1:] + y[:-1]) / 2.
    norm = np.dot(my, dx)
    y = y / norm
    if vb:
        try:
            my = (y[1:] + y[:-1]) / 2.
            assert np.isclose(np.dot(my, dx), 1.)
        except AssertionError:
            print(('`qp.utils.normalize_integral`: broken integral = '+str((my, dx))))
            assert False
    out_data = (x, y)
    return out_data

def evaluate_samples(in_data, bw_method=None, vb=False):
    """
    Produces PDF values given samples

    Parameters
    ----------
    in_data: numpy.ndarray, float
        samples x from the PDF
    bw_method: string or scalar or callable function, optional
        `scipy.stats.gaussian_kde` bandwidth methods: 'scott', 'silverman'
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, float
        sorted samples x and corresponding PDF values y
    """
    x = in_data
    x.sort()
    kde = sps.gaussian_kde(x, bw_method)
    if vb:
        print(('`qp.utils.evaluate_samples` made a KDE with bandwidth = '+str(kde.factor)))
    y = kde(x)
    out_data = (x, y)
    return out_data

def evaluate_histogram(in_data, threshold=epsilon, vb=False):
    """
    Produces PDF values given samples

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of (n+1) bin endpoints x and (n) CDF y between endpoints
    threshold: float, optional

    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, float
        sorted samples x and corresponding PDF values y
    """
    (x, y) = in_data
    dx = threshold
    xs = np.zeros(2 * len(y))
    ys = xs
    xs[::2] = x[:-1] + dx
    xs[1::2] = x[1:] - dx
    ys = np.repeat(y, 2)
    xs = sandwich(xs, (x[0] - dx, x[-1] + dx))
    ys = sandwich(ys, (threshold, threshold))
    if vb:
        try:
            assert np.all(ys >= threshold)
        except AssertionError:
            print(('broken self-evaluations in `qp.utils.evaluate_histogram`: '+str((xs, ys))))
            assert False
    out_data = (xs, ys)
    return out_data

def normalize_histogram(in_data, threshold=epsilon, vb=False):
    """
    Normalizes histogram parametrizations

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of (n+1) bin endpoints x and (n) CDF y between endpoints
    threshold: float, optional
        optional minimum threshold
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    dx = x[1:] - x[:-1]
    y[y < threshold] = threshold
    y /= np.dot(y, dx)
    if vb:
        try:
            assert np.isclose(np.dot(y, dx), 1.)
        except AssertionError:
            print(('`qp.utils.normalize_histogram`: broken integral = '+str(np.dot(y, dx))))
            assert False
    out_data = (x, y)
    return out_data

def normalize_gridded(in_data, thresholds=(epsilon, infty)):
    """
    Removes extreme values from gridded parametrizations

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of points x at which function is evaluated and the PDF y at those
        points
    thresholds: tuple, float, optional
        optional min/max thresholds for normalization

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    y[y < thresholds[0]] = thresholds[0]
    y[y > thresholds[-1]] = thresholds[-1]
    out_data = (x, y)
    return out_data

def evaluate_quantiles(in_data, threshold=epsilon, vb=False):
    """
    Estimates PDF values given quantile information

    Parameters
    ----------
    in_data: tuple, numpy.ndarray, float
        tuple of CDF values iy and values x at which those CDFs are achieved
    threshold: float, optional
        optional minimum threshold for CDF difference
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        values xs and corresponding PDF values ys
    """
    (iy, x) = in_data
    dx = x[1:] - x[:-1]
    if vb:
        try:
            assert np.all(dx > threshold)
        except AssertionError:
            print(('broken quantile locations in `qp.utils.evaluate_quantiles`: '+str(x)))
            assert False
    diy = iy[1:] - iy[:-1]
    if vb:
        try:
            assert np.all(diy > threshold)
        except AssertionError:
            print(('broken CDF values in `qp.utils.evaluate_quantiles`: '+str(iy)))
            assert False
    y = diy / dx
    (xs, ys) = evaluate_histogram((x, y), threshold=threshold, vb=vb)
    if vb:
        print(('input shape: '+str((len(x), len(y)))+', output shape: '+str((len(xs), len(ys)))))
    #     try:
    #         assert (np.all(xs > threshold) and np.all(ys > threshold))
    #     except AssertionError:
    #         print('broken quantile self-evaluations in `qp.utils.evaluate_quantiles`: '+str((xs, ys)))
    #         assert False
    # ys = ys[1:-1]
    # xs = xs[1:-1]
    # ys = sandwich(ys, (threshold, threshold))
    # x_min = xs[0] - 2 * iy[0] / y[0]
    # x_max = xs[-1] + 2 * iy[-1] / y[-1]
    # xs = sandwich(xs, (x_min, x_max))
    out_data = (xs[1:-1], ys[1:-1])
    return out_data

def normalize_quantiles(in_data, threshold=epsilon, vb=False):
    """
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
    """
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

def make_kludge_interpolator(in_data, threshold=epsilon):
    """
    Linear interpolation by hand for debugging

    Parameters
    ----------
    in_data: tuple, numpy.ndarray, float
        values x and PDF values y at which interpolator is fit
    threshold: float, optional
        minimum value to use outside interpolation range

    Returns
    -------
    kludge_interpolator: function
        evaluates linear interpolant based on input points
    """
    (x, y) = in_data
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    def kludge_interpolator(xf):
        yf = np.ones(np.shape(xf)) * threshold
        for i in range(len(x)):
            inside = ((xf >= x[i]) & (xf <= x[i+1])).nonzero()[0]
            yf[inside] = y[i] + (y[i+1] - y[i]) * (xf[inside] - x[i]) / dx[i]
        return yf
    return kludge_interpolator
