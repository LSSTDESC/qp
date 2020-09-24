"""This module implements functions to convert distributions between various representations

These functions should then be registered with the `qp.ConversionDict` using `qp_add_mapping`.
That will allow the automated conversion mechanisms to work.

"""

import numpy as np

from .conversion import qp_add_mapping
from .utils import mix_mod_fit_dist_samples, histogramize_dist
from .scipy_extend import interp_dist, spline_dist, intspline_dist, kde_dist, quantile_dist, sum_dist
from scipy.stats import rv_histogram


def convert_using_xy_vals(in_dist, class_to, **kwargs):
    """Convert using a set of x and y values.

    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the x and y values
    """

    xvals = kwargs.pop('xvals', None)
    if xvals is None:
        raise ValueError("To convert to class %s using convert_using_xy_vals you must specify xvals" % class_to)
    yvals = in_dist.pdf(xvals)
    return class_to(xvals, yvals, **kwargs)


def convert_using_samples(in_dist, class_to, **kwargs):
    """Convert using a set of values samples from the PDF

    Keywords
    --------
    size : `int`
        Number of samples to generate

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the x and y values
    """
    samples = in_dist.rvs(kwargs.pop('size', 10000))
    return class_to(samples, **kwargs)


def convert_using_hist_values(in_dist, class_to, **kwargs):
    """Convert using a set the CDF to make a histogram

    Keywords
    --------
    bins : `np.array`
        Histogram bin edges
    size : `int`
        Number of samples to generate

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the histogrammed samples
    """

    bins = kwargs.pop('bins', None)
    if bins is None:
        raise ValueError("To convert to class %s using convert_using_hist_samples you must specify xvals" % class_to)
    hist = histogramize_dist(in_dist, bins=bins)
    return class_to((hist[1], hist[0]), **kwargs)


def convert_using_hist_samples(in_dist, class_to, **kwargs):
    """Convert using a set of values samples that are then histogramed

    Keywords
    --------
    bins : `np.array`
        Histogram bin edges
    size : `int`
        Number of samples to generate

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the histogrammed samples
    """

    bins = kwargs.pop('bins', None)
    if bins is None:
        raise ValueError("To convert to class %s using convert_using_hist_samples you must specify xvals" % class_to)
    hist = np.histogram(in_dist.rvs(kwargs.pop('size', 10000)), bins=bins)
    return class_to(hist, **kwargs)


def convert_using_quantiles(in_dist, class_to, **kwargs):
    """Convert using a set of quantiles and the locations at which they are reached

    Keywords
    --------
    quantiles : `np.array`
        Quantile values to use

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the qunatile values and locations
    """

    quantiles = kwargs.pop('quantiles', None)
    if quantiles is None:
        raise ValueError("To convert to class %s using convert_using_quantiles you must specify quantiles" % class_to)
    xvals = in_dist.ppf(quantiles)
    return class_to(quantiles=quantiles, par_values=xvals, **kwargs)


def convert_using_fit(in_dist, class_to, **kwargs):
    """Convert to a functional distribution by fitting it to a set of x and y values

    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated by fitting to the samples.
    """
    xvals = kwargs.pop('xvals', None)
    if xvals is None:
        raise ValueError("To convert to class %s using convert_using_fit you must specify xvals" % class_to)
    yvals = in_dist.pdf(xvals)
    return class_to.fit(xvals, yvals)


def convert_using_mixmod_fit_samples(in_dist, class_to, **kwargs):
    """Convert to a mixture model using a set of values sample from the pdf

    Keywords
    --------
    ncomps : `int`
        Number of components in mixture model to use
    nsamples : `int`
        Number of samples to generate

    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated by fitting to the samples.
    """

    if class_to != sum_dist:
        raise TypeError("convert_using_mixmod_fit_samples can only convert to %s, not %s" % (sum_dist, class_to))
    n_comps = kwargs.pop('ncomps', 5)
    n_sample = kwargs.pop('nsamples', 1000)
    samples = in_dist.rvs(size=n_sample).reshape(-1, 1)
    weights, dists = mix_mod_fit_dist_samples(samples, n_comps)
    return sum_dist(weights, dists, **kwargs)


qp_add_mapping(convert_using_xy_vals, interp_dist, None)
qp_add_mapping(convert_using_xy_vals, spline_dist, None)
qp_add_mapping(convert_using_xy_vals, intspline_dist, None)
qp_add_mapping(convert_using_quantiles, quantile_dist, None)
qp_add_mapping(convert_using_samples, kde_dist, None)
qp_add_mapping(convert_using_hist_values, rv_histogram, None)
qp_add_mapping(convert_using_hist_samples, rv_histogram, kde_dist)
qp_add_mapping(convert_using_mixmod_fit_samples, sum_dist, None)
qp_add_mapping(convert_using_fit, None, None)
