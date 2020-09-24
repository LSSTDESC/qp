"""Utility functions for the qp package"""

import numpy as np
from scipy import stats as sps
import sys

from sklearn import mixture


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
    return np.log(arr.clip(threshold, np.inf))



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



def histogramize_dist(dist, bins, normalize=False):
    """
    Extracts a histogram from a distribution

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous`
        distribution
    bins : `np.array'
        Histogram bin edges
    normalize : `bool`
        If true, normalize the histogram

    Returns
    -------
    (bins, heights) : the histogram
    """
    cdf = dist.cdf(bins)
    heights = cdf[1:] - cdf[:-1]
    if normalize:
        return normalize_histogram((bins, heights))
    return (bins, heights)


def integrate_dist(dist, limits):
    """Integrate a distribution between two limits

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous`
        distribution
    limits : (float, flaot) or `None`
        The limits of integration, if `None` use the limits of support of the distribution

    Returns
    -------
    integral : float
    """
    if limits is None:
        limits = dist.get_support()
    cdf = dist.cdf(limits)
    return cdf[1] - cdf[0]


def approximate_dist(dist, points):
    """Approximate a distribution by taking values at a set of points

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous`
        distribution
    points : `np.array`
        points at which to evaluate the PDF of the distribution

    Returns
    -------
    values : `np.array`
       The PDF as evaluated at the points requested
    """
    points.sort()
    interpolated = dist.pdf(points)
    interpolated = normalize_gridded((points, interpolated))
    return interpolated


def mix_mod_fit_dist_grid(grid, n_components):
    """Fit a set of gridded values to a Gaussian mixture

    Parameters
    ----------
    grip : `np.array`
        Values to fit
    n_components : `int`
        Number of components to fit

    Returns
    -------
    out_dict : `qp.sum_dist`
        The fitted output distribution
    """
    # FIXME
    raise NotImplementedError('mix_mod_fit_dist_grid')


def mix_mod_fit_dist_samples(samples, n_components):
    """Fit a set of samples to a Gaussian mixture

    Parameters
    ----------
    samples : `np.array`
        Samples to fit
    n_components : `int`
        Number of components to fit

    Returns
    -------
    weights, dists : The wieghts and distribtuions.
    """


    estimator = mixture.GaussianMixture(n_components=n_components)
    estimator.fit(samples)

    weights = estimator.weights_
    means = estimator.means_[:, 0]
    stdevs = np.sqrt(estimator.covariances_[:, 0, 0])

    dists = [sps.norm(loc=mean, scale=stdev) for  mean, stdev in zip(means, stdevs)]
    return weights, dists
