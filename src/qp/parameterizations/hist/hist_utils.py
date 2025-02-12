import numpy as np

from ...utils.array import (
    get_bin_indices,
    get_eval_case,
    CASE_PRODUCT,
    CASE_FACTOR,
    CASE_2D,
)


#
# PDF evaluation functions
#


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


#
# Conversion functions
#


def extract_hist_values(in_dist, **kwargs):
    """Convert using a set of values sampled from the PDF

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    bins : `np.array`
        Histogram bin edges

    Returns
    -------
    data : `dict`
        The extracted data
    """
    bins = kwargs.pop("bins", None)
    if bins is None:  # pragma: no cover
        raise ValueError("To convert using extract_hist_samples you must specify bins")
    bins, pdfs = in_dist.histogramize(bins)
    return dict(bins=bins, pdfs=pdfs)


def extract_hist_samples(in_dist, **kwargs):
    """Convert using a set of values samples that are then histogramed

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    bins : `np.array`
        Histogram bin edges
    size : `int`
        Number of samples to generate

    Returns
    -------
    data : `dict`
        The extracted data
    """
    bins = kwargs.pop("bins", None)
    size = kwargs.pop("size", 1000)
    if bins is None:  # pragma: no cover
        raise ValueError("To convert using extract_hist_samples you must specify bins")
    samples = in_dist.rvs(size=size)

    def hist_helper(sample):
        return np.histogram(sample, bins=bins)[0]

    vv = np.vectorize(
        hist_helper, signature="(%i)->(%i)" % (samples.shape[0], bins.size - 1)
    )
    pdfs = vv(samples)
    return dict(bins=bins, pdfs=pdfs)
