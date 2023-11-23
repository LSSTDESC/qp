"""This module implements functions to convert distributions between various representations
These functions should then be registered with the `qp.ConversionDict` using `qp_add_mapping`.
That will allow the automated conversion mechanisms to work.
"""
import numpy as np
from scipy import integrate as sciint
from scipy import interpolate as sciinterp

from .lazy_modules import mixture
from .sparse_rep import (
    build_sparse_representation,
    decode_sparse_indices,
    indices2shapes,
)


def extract_vals_at_x(in_dist, **kwargs):
    """Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data
    """
    xvals = kwargs.pop("xvals", None)
    if xvals is None:  # pragma: no cover
        raise ValueError("To convert to extract_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    return dict(xvals=xvals, yvals=yvals)


def extract_xy_vals(in_dist, **kwargs):
    """Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data
    """
    xvals = kwargs.pop("xvals", None)
    if xvals is None:  # pragma: no cover
        raise ValueError("To convert using extract_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    expand_x = np.ones(yvals.shape) * np.squeeze(xvals)
    return dict(xvals=expand_x, yvals=yvals)


def extract_samples(in_dist, **kwargs):
    """Convert using a set of values sampled from the PDF

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    size : `int`
        Number of samples to generate

    Returns
    -------
    data : `dict`
        The extracted data
    """
    samples = in_dist.rvs(size=kwargs.pop("size", 1000))
    xvals = kwargs.pop("xvals")
    return dict(samples=samples, xvals=xvals, yvals=None)


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


def extract_quantiles(in_dist, **kwargs):
    """Convert using a set of quantiles and the locations at which they are reached

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    quantiles : `np.array`
        Quantile values to use

    Returns
    -------
    data : `dict`
        The extracted data
    """
    quants = kwargs.pop("quants", None)
    if quants is None:  # pragma: no cover
        raise ValueError("To convert using extract_quantiles you must specify quants")
    locs = in_dist.ppf(quants)
    return dict(quants=quants, locs=locs)


def extract_fit(in_dist, **kwargs):  # pragma: no cover
    """Convert to a functional distribution by fitting it to a set of x and y values

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data
    """
    raise NotImplementedError("extract_fit")
    # xvals = kwargs.pop('xvals', None)
    # if xvals is None:
    #   raise ValueError("To convert using extract_fit you must specify xvals")
    ##vals = in_dist.pdf(xvals)


def extract_mixmod_fit_samples(in_dist, **kwargs):
    """Convert to a mixture model using a set of values sample from the pdf

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    ncomps : `int`
        Number of components in mixture model to use
    nsamples : `int`
        Number of samples to generate
    random_state : `int`
        Used to reproducibly generate random variate from in_dist

    Returns
    -------
    data : `dict`
        The extracted data
    """
    n_comps = kwargs.pop("ncomps", 3)
    n_sample = kwargs.pop("nsamples", 1000)
    random_state = kwargs.pop("random_state", None)
    samples = in_dist.rvs(size=n_sample, random_state=random_state)

    def mixmod_helper(samps):
        estimator = mixture.GaussianMixture(n_components=n_comps)
        estimator.fit(samps.reshape(-1, 1))
        weights = estimator.weights_
        means = estimator.means_[:, 0]
        stdevs = np.sqrt(estimator.covariances_[:, 0, 0])
        ov = np.vstack([weights, means, stdevs])
        return ov

    vv = np.vectorize(mixmod_helper, signature="(%i)->(3,%i)" % (n_sample, n_comps))
    fit_vals = vv(samples)
    return dict(
        weights=fit_vals[:, 0, :], means=fit_vals[:, 1, :], stds=fit_vals[:, 2, :]
    )


def extract_voigt_mixmod(in_dist, **kwargs):  # pragma: no cover
    """Convert to a voigt mixture model starting with a gaussian mixture model,
    trivially by setting gammas to 0

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Returns
    -------
    data : `dict`
        The extracted data
    """
    objdata = in_dist.objdata()
    means = objdata["means"]
    stds = objdata["stds"]
    weights = objdata["weights"]
    gammas = np.zeros_like(means)
    return dict(means=means, stds=stds, weights=weights, gammas=gammas, **kwargs)


def extract_voigt_xy(in_dist, **kwargs):  # pragma: no cover
    """Build a voigt function basis and run a match-pursuit algorithm to fit gridded data

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Returns
    -------
    data : `dict`
        The extracted data as sparse indices, basis, and metadata to rebuild the basis
    """

    sparse_results = extract_voigt_xy_sparse(in_dist, **kwargs)
    indices = sparse_results["indices"]
    meta = sparse_results["metadata"]

    w, m, s, g = indices2shapes(indices, meta)
    return dict(means=m, stds=s, weights=w, gammas=g)


def extract_voigt_xy_sparse(in_dist, **kwargs):  # pragma: no cover
    """Build a voigt function basis and run a match-pursuit algorithm to fit gridded data

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Returns
    -------
    data : `dict`
        The extracted data as shaped parameters means, stds, weights, gammas
    """

    yvals = in_dist.objdata()["yvals"]

    default = in_dist.metadata()["xvals"][0]
    z = kwargs.pop("xvals", default)
    nz = kwargs.pop("nz", 300)

    minz = np.min(z)
    _, j = np.where(yvals > 0)
    maxz = np.max(z[j])
    newz = np.linspace(minz, maxz, nz)
    interp = sciinterp.interp1d(z, yvals, assume_sorted=True)
    newpdf = interp(newz)
    newpdf = newpdf / sciint.trapz(newpdf, newz).reshape(-1, 1)
    ALL, bigD, _ = build_sparse_representation(newz, newpdf)
    return dict(indices=ALL, metadata=bigD)


def extract_sparse_from_xy(in_dist, **kwargs):  # pragma: no cover
    """Extract sparse representation from an xy interpolated representation

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    xvals : array-like
        Used to override the y-values
    xvals : array-like
        Used to override the x-values
    nvals : int
        Used to override the number of bins

    Returns
    -------
    metadata : `dict`
        Dictionary with data for sparse representation

    Notes
    -----
    This function will rebin to a grid more suited to the in_dist support by
    removing x-values corrsponding to y=0
    """
    default = in_dist.objdata()["yvals"]
    yvals = kwargs.pop("yvals", default)
    default = in_dist.metadata()["xvals"][0]
    xvals = kwargs.pop("xvals", default)
    nvals = kwargs.pop("nvals", 300)
    # rebin to a grid more suited to the in_dist support
    xmin = np.min(xvals)
    _, j = np.where(yvals > 0)
    xmax = np.max(xvals[j])
    newx = np.linspace(xmin, xmax, nvals)
    interp = sciinterp.interp1d(xvals, yvals, assume_sorted=True)
    newpdf = interp(newx)
    sparse_indices, metadata, _ = build_sparse_representation(newx, newpdf)
    metadata["xvals"] = newx
    metadata["sparse_indices"] = sparse_indices
    metadata.pop("Ntot")
    return metadata


def extract_xy_sparse(in_dist, **kwargs):  # pragma: no cover
    """Extract xy-interpolated representation from an sparese representation

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Other Parameters
    ----------------
    xvals : array-like
        Used to override the y-values
    xvals : array-like
        Used to override the x-values
    nvals : int
        Used to override the number of bins

    Returns
    -------
    metadata : `dict`
        Dictionary with data for interpolated representation

    Notes
    -----
    This function will rebin to a grid more suited to the in_dist support by
    removing x-values corrsponding to y=0
    """

    yvals = in_dist.objdata()["yvals"]
    default = in_dist.metadata()["xvals"][0]
    xvals = kwargs.pop("xvals", default)
    nvals = kwargs.pop("nvals", 300)
    # rebin to a grid more suited to the in_dist support
    xmin = np.min(xvals)
    _, j = np.where(yvals > 0)
    xmax = np.max(xvals[j])
    newx = np.linspace(xmin, xmax, nvals)
    interp = sciinterp.interp1d(xvals, yvals, assume_sorted=True)
    newpdf = interp(newx)
    sparse_indices, sparse_meta, A = build_sparse_representation(newx, newpdf)
    # decode the sparse indices into basis indices and weights
    basis_indices, weights = decode_sparse_indices(sparse_indices)
    # retrieve the weighted array of basis functions for each object
    pdf_y = A[:, basis_indices] * weights
    # normalize and sum the weighted pdfs
    x = sparse_meta["z"]
    y = pdf_y.sum(axis=-1)
    norms = sciint.trapz(y.T, x)
    y /= norms
    # super(sparse_gen, self).__init__(x, y.T, *args, **kwargs)
    xvals = x
    yvals = y.T
    return dict(xvals=xvals, yvals=yvals, **kwargs)
