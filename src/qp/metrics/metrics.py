"""This module implements some performance metrics for distribution parameterization"""
import logging
from collections import namedtuple
from functools import partial

import numpy as np
from deprecated import deprecated

from qp.metrics import array_metrics
from qp.metrics.brier import Brier
from qp.metrics.goodness_of_fit import goodness_of_fit_metrics
from qp.utils import epsilon

Grid = namedtuple(
    "Grid", ["grid_values", "cardinality", "resolution", "hist_bin_edges", "limits"]
)


def _calculate_grid_parameters(limits, dx: float = 0.01) -> Grid:
    """
    Create a grid of points and return parameters describing it.

    Args:
        limits (Iterable) often a 2-tuple or numpy array with shape (2,)
            the max and min values of the 1d grid
        dx (float, optional):
            the desired delta between points. Used to define the cardinality. Defaults to 0.01.

    Returns:
        Grid: a namedtuple containing a 1d grid's values and attributes.
            grid_values: np.array with size = cardinality
            cardinality: int, number of elements in grid_value
            resolution: float, equal to grid_values[i] - grid_values[i-1]
            hist_bin_edges: np.array with size = cardinality+1.
                Equally spaced histogram bin edges starting at limit-resolution/2.
                Assumes that grid_value[i] should be centered in the bin defined by
                (hist_bin_edge[i], hist_bin_edge[i+1]).
            limits: 2-tuple, the limits passed in and used in this function
    """
    cardinality = int((limits[-1] - limits[0]) / dx)
    grid_values = np.linspace(limits[0], limits[1], cardinality)
    resolution = (limits[-1] - limits[0]) / (cardinality - 1)
    hist_bin_edges = np.histogram_bin_edges(
        (limits[0] - resolution / 2, limits[1] + resolution / 2), cardinality
    )

    return Grid(grid_values, cardinality, resolution, hist_bin_edges, limits)


def calculate_moment(p, N, limits, dx=0.01):
    """
    Calculates a moment of a qp.Ensemble object

    Parameters
    ----------
    p: qp.Ensemble object
        the collection of PDFs whose moment will be calculated
    N: int
        order of the moment to be calculated
    limits: tuple of floats
        endpoints of integration interval over which to calculate moments
    dx: float
        resolution of integration grid

    Returns
    -------
    M: float
        value of the moment
    """
    # Make a grid from the limits and resolution
    grid = _calculate_grid_parameters(limits, dx)

    # Evaluate the functions on the grid
    pe = p.gridded(grid.grid_values)[1]

    # calculate the moment
    grid_to_N = grid.grid_values**N
    M = array_metrics.quick_moment(pe, grid_to_N, grid.resolution)

    return M


def calculate_kld(p, q, limits, dx=0.01):
    """
    Calculates the Kullback-Leibler Divergence between two qp.Ensemble objects.

    Parameters
    ----------
    p: Ensemble object
        probability distribution closer to the truth
    q: Ensemble object
        probability distribution that approximates p
    limits: tuple of floats
        endpoints of integration interval in which to calculate KLD
    dx: float
        resolution of integration grid

    Returns
    -------
    Dpq: float
        the value of the Kullback-Leibler Divergence from `q` to `p`

    Notes
    -----
    TO DO: have this take number of points not dx!
    """
    if p.shape != q.shape:
        raise ValueError(
            "Cannot calculate KLD between two ensembles with different shapes"
        )

    # Make a grid from the limits and resolution
    grid = _calculate_grid_parameters(limits, dx)

    # Evaluate the functions on the grid and normalize
    pe = p.gridded(grid.grid_values)
    pn = pe[1]
    qe = q.gridded(grid.grid_values)
    qn = qe[1]

    # Calculate the KLD from q to p
    Dpq = array_metrics.quick_kld(
        pn, qn, grid.resolution
    )  # np.dot(pn * logquotient, np.ones(len(grid)) * dx)

    if np.any(Dpq < 0.0):  # pragma: no cover
        print("broken KLD: " + str((Dpq, pn, qn, grid.resolution)))
        Dpq = epsilon * np.ones(Dpq.shape)
    return Dpq


def calculate_rmse(p, q, limits, dx=0.01):
    """
    Calculates the Root Mean Square Error between two qp.Ensemble objects.

    Parameters
    ----------
    p: qp.Ensemble object
        probability distribution function whose distance between its truth and the
        approximation of `q` will be calculated.
    q: qp.Ensemble object
        probability distribution function whose distance between its approximation and the
        truth of `p` will be calculated.
    limits: tuple of floats
        endpoints of integration interval in which to calculate RMS
    dx: float
        resolution of integration grid

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`

    Notes
    -----
    TO DO: change dx to N
    """
    if p.shape != q.shape:
        raise ValueError(
            "Cannot calculate RMSE between two ensembles with different shapes"
        )

    # Make a grid from the limits and resolution
    grid = _calculate_grid_parameters(limits, dx)

    # Evaluate the functions on the grid
    pe = p.gridded(grid.grid_values)[1]
    qe = q.gridded(grid.grid_values)[1]

    # Calculate the RMS between p and q
    rms = array_metrics.quick_rmse(
        pe, qe, grid.cardinality
    )  # np.sqrt(dx * np.sum((pe - qe) ** 2))

    return rms


def calculate_rbpe(p, limits=(np.inf, np.inf)):
    """
    Calculates the risk based point estimates of a qp.Ensemble object.
    Algorithm as defined in 4.2 of 'Photometric redshifts for Hyper Suprime-Cam
    Subaru Strategic Program Data Release 1' (Tanaka et al. 2018).

    Parameters
    ----------
    p: qp.Ensemble object
        Ensemble of PDFs to be evalutated
    limits: tuple
        The limits at which to evaluate possible z_best estimates.
        If custom limits are not provided then all potential z value will be
        considered using the scipy.optimize.minimize_scalar function.

    Returns
    -------
    rbpes: array of floats
        The risk based point estimates of the provided ensemble.
    """
    rbpes = []

    def evaluate_pdf_at_z(z, dist):
        return dist.pdf(z)[0][0]

    for n in range(0, p.npdf):
        if p[n].npdf != 1:
            raise ValueError(
                "quick_rbpe only handles Ensembles with a single PDF "
                "for ensembles with more than one PDF, use the qp.metrics.risk_based_point_estimate function."
            )

        this_dist_pdf_at_z = partial(evaluate_pdf_at_z, dist=p[n])
        integration_bounds = (p[n].ppf(0.01)[0][0], p[n].ppf(0.99)[0][0])

        rbpes.append(
            array_metrics.quick_rbpe(this_dist_pdf_at_z, integration_bounds, limits)
        )

    return np.array(rbpes)


def _prepare_for_brier(p, truth, limits, dx=0.01):
    # Ensure that the number of distributions objects in the Ensemble
    # is equal to the length of the truth array
    if p.npdf != len(truth):
        raise ValueError(
            "Number of distributions in the Ensemble (%d) is not equal to the number of truth values (%d)"
            % (p.npdf, len(truth))
        )

    # Values of truth that are outside the defined limits will not appear truth_array.
    # Consider expanding the limits or using numpy.clip to restrict truth values to the limits.
    if np.any(np.less(truth, limits[0])) or np.any(np.greater(truth, limits[1])):
        raise ValueError(f"Input truth values exceed the defined limits ({min(truth)}, {max(truth)}) ({limits[0]} {limits[1]})")

    # Make a grid object that defines grid values and histogram bin edges using limits and dx
    grid = _calculate_grid_parameters(limits, dx)

    # Evaluate the pdf of the distributions on the grid.
    # The value returned from p.gridded is a 2-tuple. The 0th index is the array of grid points,
    # the 1st index is the array of PDF values. Thus we call p.gridded(...)[1]
    pdf_values = p.gridded(grid.grid_values)[1]

    # Create the NxM truth_array.
    # Note np.histogram returns a 2-tuple. The 0th index is the histogram array,
    # thus we call np.squeeze to remove extra dimensions.
    truth_array = np.squeeze([np.histogram(t, grid.hist_bin_edges)[0] for t in truth])

    # instantiate the Brier metric object
    return Brier(pdf_values, truth_array)

def calculate_brier(p, truth, limits, dx=0.01):
    """This function will do the following:

    1) Generate a Mx1 sized grid based on `limits` and `dx`.
    2) Produce an NxM array by evaluating the pdf for each of the N distribution objects
       in the Ensemble p on the grid.
    3) Produce an NxM truth_array using the input truth and the generated grid. All values will be 0 or 1.
    4) Create a Brier metric evaluation object
    5) Return the result of the Brier metric calculation.

    Parameters
    ----------
    p: qp.Ensemble object
        of N distributions probability distribution functions that will be gridded and compared against truth.
    truth: Nx1 sequence
        the list of true values, 1 per distribution in p.
    limits: 2-tuple of floats
        endpoints grid to evaluate the PDFs for the distributions in p
    dx: float
        resolution of the grid Defaults to 0.01.

    Returns
    -------
    Brier_metric: float
    """

    brier_metric_evaluation = _prepare_for_brier(p, truth, limits, dx)

    # return the results of evaluating the Brier metric
    return brier_metric_evaluation.evaluate()

def calculate_brier_for_accumulation(p, truth, limits, dx=0.01):
    brier_metric_evaluation = _prepare_for_brier(p, truth, limits, dx)

    brier_sum = brier_metric_evaluation.accumulate()

    return (brier_sum, p.npdf)

@deprecated(
    reason="""
    This implementation is deprecated for performance reasons and does not accommodate N vs 1 comparisons.
    Please use calculate_goodness_of_fit instead. This method is for testing purposes only.
    """,
    category=DeprecationWarning,
)
def calculate_anderson_darling(
    p, scipy_distribution="norm", num_samples=100, _random_state=None
):  # pylint: disable=unused-argument
    """This function is deprecated and will be completely removed in a later version.
    Please use `calculate_goodness_of_fit` instead.

    Returns
    -------
    logger.warning
    """
    logging.warning(
        "This function is deprecated, please use `calculate_goodness_of_fit` with `fit_metric='ad'`"
    )  # pragma: no cover


@deprecated(
    reason="""
    This implementation is deprecated for performance reasons and does not accommodate N vs 1 comparisons.
    Please use calculate_goodness_of_fit instead. This method is for testing purposes only.
    """,
    category=DeprecationWarning,
)
def calculate_cramer_von_mises(
    p, q, num_samples=100, _random_state=None, **kwargs
):  # pylint: disable=unused-argument
    """This function is deprecated and will be completely removed in a later version.
    Please use `calculate_goodness_of_fit` instead.

    Returns
    -------
    logger.warning
    """
    logging.warning(
        "This function is deprecated, please use `calculate_goodness_of_fit` with `fit_metric='cvm'`"
    )  # pragma: no cover


@deprecated(
    reason="""
    This implementation is deprecated for performance reasons and does not accommodate N vs 1 comparisons.
    Please use calculate_goodness_of_fit instead. This method is for testing purposes only.
    """,
    category=DeprecationWarning,
)
def calculate_kolmogorov_smirnov(
    p, q, num_samples=100, _random_state=None
):  # pylint: disable=unused-argument
    """This function is deprecated and will be completely removed in a later version.
    Please use `calculate_goodness_of_fit` instead.

    Returns
    -------
    logger.warning
    """
    logging.warning(
        "This function is deprecated, please use `calculate_goodness_of_fit` with `fit_metric='ks'`"
    )  # pragma: no cover


def calculate_outlier_rate(p, lower_limit=0.0001, upper_limit=0.9999):
    """Fraction of outliers in each distribution

    Parameters
    ----------
    p : qp.Ensemble
        A collection of N distributions. This implementation expects that Ensembles are not nested.
    lower_limit : float, optional
        Lower bound CDF for outliers, by default 0.0001
    upper_limit : float, optional
        Upper bound CDF for outliers, by default 0.9999

    Returns
    -------
    [float]
        1xN array where each element is the percent of outliers for a distribution in the Ensemble.
    """

    # Validate that all the distributions in the Ensemble are single distributions - i.e. no nested Ensembles
    try:
        _check_ensemble_is_not_nested(p)
    except ValueError:  #pragma: no cover - unittest coverage for _check_ensemble_is_not_nested is complete
        logging.warning("Each element in the ensemble `p` must be a single distribution.")

    outlier_rates = np.array([(dist.cdf(lower_limit) + (1. - dist.cdf(upper_limit)))[0][0] for dist in p])
    return outlier_rates


def calculate_goodness_of_fit(
    estimate, reference, fit_metric="ks", num_samples=100, _random_state=None
):
    """This method calculates goodness of fit between the distributions in the
    `estimate` and `reference` Ensembles using the specified fit_metric.

    Parameters
    ----------
    estimate : Ensemble containing N distributions
        Random variate samples will be drawn from this Ensemble
    reference : Ensemble containing N or 1 distributions
        The CDF of the distributions in this Ensemble are used in the goodness of fit
        calculation.
    fit_metric : string, optional
        The goodness of fit metric to use. One of ['ad', 'cvm', 'ks']. For clarity,
        'ad' = Anderson-Darling, 'cvm' = Cramer-von Mises, and 'ks' = Kolmogorov-Smirnov, by default 'ks'
    num_samples : int, optional
        Number of random variates to draw from each distribution in `estimate`, by default 100
    _random_state : _type_, optional
        Used for testing to create reproducible sets of random variates, by default None

    Returns
    -------
    output: [float]
        A array of floats where each element is the result of the statistic calculation.

    Raises
    ------
    KeyError
        If the requested `fit_metric` is not contained in `goodness_of_fit_metrics` dictionary,
        raise a KeyError.

    Notes
    -----
    The calculation of the goodness of fit metrics is not symmetric.
    i.e. `calculate_goodness_of_fit(p, q, ...) != calculate_goodness_of_fit(q, p, ...)`

    In the future, we should be able to do this directly from the PDFs without needing to
    take random variates from the `estimate` Ensemble.

    The vectorized implementations of fit metrics are copied over (unmodified) from
    the developer branch of Scipy 1.10.0dev. When Scipy 1.10 is released, we can replace
    the copied implementation with the ones in Scipy.
    """

    try:
        _check_ensembles_contain_correct_number_of_distributions(estimate, reference)
    except (
        ValueError
    ):  # pragma: no cover - unittest coverage for _check_ensembles_contain_correct_number_of_distributions is complete  pylint: disable=line-too-long
        logging.warning(
            "The ensemble `reference` should have 1 or N distributions.  "
            "With N = number of distributions in the ensemble `estimate`."
        )

    try:
        _check_ensemble_is_not_nested(estimate)
    except (
        ValueError
    ):  # pragma: no cover - unittest coverage for _check_ensemble_is_not_nested is complete
        logging.warning(
            "Each element in the ensemble `estimate` must be a single distribution."
        )

    try:
        _check_ensemble_is_not_nested(reference)
    except (
        ValueError
    ):  # pragma: no cover - unittest coverage for _check_ensemble_is_not_nested is complete
        logging.warning(
            "Each element in the ensemble `reference` must be a single distribution."
        )

    if fit_metric not in goodness_of_fit_metrics:
        metrics = list(goodness_of_fit_metrics.keys())
        raise KeyError(f"`fit_metric` should be one of {metrics}.")

    return goodness_of_fit_metrics[fit_metric](
        reference,
        np.squeeze(estimate.rvs(size=num_samples, random_state=_random_state)),
    )


def _check_ensembles_are_same_size(p, q):
    """This utility function ensures checks that two Ensembles contain equal numbers of distributions"

    Args:
        p qp.Ensemble: An Ensemble containing 0 or more distributions
        q qp.Ensemble: A second Ensemble containing 0 or more distributions

    Raises:
        ValueError: If the result of evaluating qp.Ensemble.npdf on each Ensemble is not the same.
    """
    if p.npdf != q.npdf:
        raise ValueError("Input ensembles should have the same number of distributions")


def _check_ensemble_is_not_nested(p):
    """This utility function ensures that each element in the Ensemble is a single distribution.

    Args:
        p qp.Ensemble: An Ensemble that could contain nested Ensembles with multiple distributions in each

    Raises:
        ValueError: If there are any elements of the input Ensemble that contain more than 1 PDF.
    """
    for dist in p:
        if dist.npdf != 1:
            raise ValueError(
                "Each element in the input Ensemble should be a single distribution."
            )


def _check_ensembles_contain_correct_number_of_distributions(estimate, reference):
    """This utility function ensures that the number of distributions in the ensembles matches
    expectations. estimate can contain 1 to N distributions.
    reference should contain either 1 or the same number of distributions
    as estimate.

    Example logic:
    estimate=N, reference=N (1 <= N) -> Pass
    estimate=N, reference=1 (1 <= N) -> Pass
    estimate=1, reference=N (N != 1) -> Raise ValueError
    estimate=N, reference=M (N != M) -> Raise ValueError

    Parameters
    ----------
    estimate : Ensemble
        Used to calculate goodness of fit metrics, random
        variates will be produced from the distributions in this ensemble.
    reference : Ensemble
        The CDFs of the distributions in this ensemble will be used to calculate
        goodness of fit metrics.

    Raises
    ------
    ValueError
        If the number of distributions in the reference ensemble does not meet the requirements,
        raise a ValueError.
    """

    if estimate.npdf == reference.npdf:
        pass
    elif reference.npdf == 1:
        pass
    else:
        raise ValueError(
            "`reference` should contain either 1 distribution "
            "or the same number of distributions as `estimate`."
        )
