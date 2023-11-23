"""This module contains functions copied from the 1.10.0dev branch of Scipy.
The original code can be found here:
https://github.com/scipy/scipy/blob/maintenance/1.10.x/scipy/stats/_fit.py#L722
The original Scipy 1.10.0dev code is wrapped with a function that prevents
passing more than 1 distribution at a time.
So, temporarily, we'll make use of the underlying, vectorized functions.
Fortunately, the vectorized Scipy code works without modification. 
Once Scipy 1.10 is made available, we can swap out the copied functions for those in Scipy 1.10.
"""

import numpy as np


def _anderson_darling(dist, data):
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    i = np.arange(1, n + 1)
    Si = (2 * i - 1) / n * (dist.logcdf(x) + dist.logsf(x[..., ::-1]))
    S = np.sum(Si, axis=-1)
    return -n - S


def _kolmogorov_smirnov(dist, data):
    x = np.sort(data, axis=-1)
    cdfvals = dist.cdf(x)
    Dplus = _compute_dplus(cdfvals)  # always works along last axis
    Dminus = _compute_dminus(cdfvals)
    return np.maximum(Dplus, Dminus)


def _compute_dplus(cdfvals):  # adapted from _stats_py before gh-17062
    n = cdfvals.shape[-1]
    return (np.arange(1.0, n + 1) / n - cdfvals).max(axis=-1)


def _compute_dminus(cdfvals, axis=-1):
    n = cdfvals.shape[-1]
    return (cdfvals - np.arange(0.0, n) / n).max(axis=axis)


def _cramer_von_mises(dist, data):
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    cdfvals = dist.cdf(x)
    u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    w = 1 / (12 * n) + np.sum((u - cdfvals) ** 2, axis=-1)
    return w


# The following methods can be replaced by:
# scipy.stats._fit._anderson_darling,
# scipy.stats._fit._cramer_von_mises, and
# scipy.stats._fit._kolmogorov_smirnov when Scipy 1.10 is available.
goodness_of_fit_metrics = {
    "ad": _anderson_darling,
    "cvm": _cramer_von_mises,
    "ks": _kolmogorov_smirnov,
}
