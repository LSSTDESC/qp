"""This module implements some performance metrics for distribution parameterization"""
import logging
from collections import namedtuple
from functools import partial

import numpy as np
from scipy import stats

import qp.metrics.array_metrics as array_metrics
from qp.metrics.brier import Brier
from qp.utils import epsilon

Grid = namedtuple('Grid', ['grid_values', 'cardinality', 'resolution', 'hist_bin_edges', 'limits'])

def _calculate_grid_parameters(limits, dx:float=0.01) -> Grid:
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
    hist_bin_edges = np.histogram_bin_edges((limits[0]-resolution/2, limits[1]+resolution/2), cardinality)

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
    grid_to_N = grid.grid_values ** N
    M = array_metrics.quick_moment(pe, grid_to_N, grid.resolution)

    return M


def calculate_kld(p, q, limits, dx=0.01):
    """
    Calculates the Kullback-Leibler Divergence between two qp.Ensemble objects.

    Parameters
    ----------
    p: Ensemble object
        probability distribution whose distance _from_ `q` will be calculated.
    q: Ensemble object
        probability distribution whose distance _to_ `p` will be calculated.
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
        raise ValueError('Cannot calculate KLD between two ensembles with different shapes')

    # Make a grid from the limits and resolution
    grid = _calculate_grid_parameters(limits, dx)

    # Evaluate the functions on the grid and normalize
    pe = p.gridded(grid.grid_values)
    pn = pe[1]
    qe = q.gridded(grid.grid_values)
    qn = qe[1]

    # Calculate the KLD from q to p
    Dpq = array_metrics.quick_kld(pn, qn, grid.resolution)# np.dot(pn * logquotient, np.ones(len(grid)) * dx)

    if np.any(Dpq < 0.): #pragma: no cover
        print('broken KLD: '+str((Dpq, pn, qn, grid.resolution)))
        Dpq = epsilon*np.ones(Dpq.shape)
    return Dpq


def calculate_rmse(p, q, limits, dx=0.01):
    """
    Calculates the Root Mean Square Error between two qp.Ensemble objects.

    Parameters
    ----------
    p: qp.Ensemble object
        probability distribution function whose distance between its truth and the approximation of `q` will be calculated.
    q: qp.Ensemble object
        probability distribution function whose distance between its approximation and the truth of `p` will be calculated.
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
        raise ValueError('Cannot calculate RMSE between two ensembles with different shapes')

    # Make a grid from the limits and resolution
    grid = _calculate_grid_parameters(limits, dx)

    # Evaluate the functions on the grid
    pe = p.gridded(grid.grid_values)[1]
    qe = q.gridded(grid.grid_values)[1]

    # Calculate the RMS between p and q
    rms = array_metrics.quick_rmse(pe, qe, grid.cardinality)# np.sqrt(dx * np.sum((pe - qe) ** 2))

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
    limits, tuple of floats
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
            raise ValueError('quick_rbpe only handles Ensembles with a single PDF, for ensembles with more than one PDF, use the qp.metrics.risk_based_point_estimate function.')

        this_dist_pdf_at_z = partial(evaluate_pdf_at_z, dist=p[n])
        integration_bounds = (p[n].ppf(0.01)[0][0], p[n].ppf(0.99)[0][0])

        rbpes.append(array_metrics.quick_rbpe(this_dist_pdf_at_z, integration_bounds, limits))

    return np.array(rbpes)

def calculate_brier(p, truth, limits, dx=0.01):
    """This function will do the following:
        1) Generate a Mx1 sized grid based on `limits` and `dx`.
        2) Produce an NxM array by evaluating the pdf for each of the N distribution objects in the Ensemble p on the grid. 
        3) Produce an NxM truth_array using the input truth and the generated grid. All values will be 0 or 1.
        4) Create a Brier metric evaluation object
        5) Return the result of the Brier metric calculation.

    Args:
        p: qp.Ensemble object of N distributions
            probability distribution functions that will be gridded and compared against truth.
        truth: Nx1 sequence
            the list of true values, 1 per distribution in p.
        limits: 2-tuple of floats
            endpoints grid to evaluate the PDFs for the distributions in p
        dx: float
            resolution of the grid Defaults to 0.01.
    """

    # Ensure that the number of distributions objects in the Ensemble is equal to the length of the truth array
    if p.npdf != len(truth):
        raise ValueError("Number of distributions in the Ensemble (%d) is not equal to the number of truth values (%d)" % (p.npdf, len(truth)))

    # Values of truth that are outside the defined limits will not appear truth_array.
    # Consider expanding the limits or using numpy.clip to restrict truth values to the limits.
    if np.any(np.less(truth, limits[0])) or np.any(np.greater(truth, limits[1])):
        raise ValueError("Input truth values exceed the defined limits")

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
    brier_metric_evaluation = Brier(pdf_values, truth_array)

    # return the results of evaluating the Brier metric
    return brier_metric_evaluation.evaluate()

def calculate_outlier_rate(p, lower_limit=0.0001, upper_limit=0.9999):
    """Fraction of outliers in each distribution

    Args:
        p qp.Ensemble of N distributions: This implementation expects that Ensembles are not nested.
        lower_limit (float, optional): _description_. Defaults to 0.0001.
        upper_limit (float, optional): _description_. Defaults to 0.9999.

    Returns
    -------
    outlier_rates: 1xN array of floats
        The percent of outliers for each distribution in the Ensemble.
    """

    # Validate that all the distributions in the Ensemble are single distributions - i.e. no nested Ensembles
    for dist in p:
        if dist.npdf != 1:
            raise ValueError("The implementation of outlier rate expects each element in the Ensemble to be a single distribution")

    outlier_rates = [(dist.cdf(lower_limit) + (1. - dist.cdf(upper_limit)))[0][0] for dist in p]
    return outlier_rates

def calculate_anderson_ksamp(cdf_at_truth_values, cdf_min=0., cdf_max=1.):
    """ Use scipy.stats.anderson_ksamp to compute the Anderson-Darling statistic
    for the cdf(truth) values by comparing with a uniform distribution between 0 and 1.
    Up to the current version (1.6.2), scipy.stats.anderson does not support
    uniform distributions as reference for 1-sample test, therefore we create a uniform
    "distribution" and pass it as the second value in the list of parameters to the scipy 
    implementation.

    Parameters
    ----------
    cdf_at_truth_values: [float], A 1xN list, where each element is the result of evaluating cdf_n(truth_n)
        for the nth distribution in a set of N distributions.
    cdf_min: float, optional
        cdf(x) values below this are discarded
    cdf_max: float, optional
        cdf(x) values greater than this are discarded

    Returns
    -------
    Anderson_ksampResult: A namedtuple with the following values. 
        statistic : float
            Normalized k-sample Anderson-Darling test statistic.
        critical_values : array
            The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
            0.5%, 0.1%.
        significance_level : float
            An approximate significance level at which the null hypothesis for the
            provided samples can be rejected. The value is floored / capped at
            0.1% / 25%.
    """

    # Removed the CDF values that are outside the min/max range
    cdf_at_truth_values_clean = _clean_cdf_values(cdf_at_truth_values, cdf_min, cdf_max)

    # Create a uniform "distribution" of values in the range of [cdf_min, cdf_max]
    uniform_yvals = np.linspace(cdf_min, cdf_max, len(cdf_at_truth_values_clean))

    return stats.anderson_ksamp([cdf_at_truth_values_clean, uniform_yvals])

def _clean_cdf_values(cdf_at_truth_values, cdf_min, cdf_max):
    """Remove and report any cdf(x) that are outside the min/max range.

    Args:
        cdf_at_truth_values [float]: The array of PIT values 
        cdf_min float: The minimum cdf(x) value to accept
        cdf_max float: The maximum cdf(x) value to accept

    Returns:
        pits_clean [float]: The list of PIT values within the min/max range.
    """
    # Create truth mask for pit values between cdf_min and pit max
    mask = (cdf_at_truth_values >= cdf_min) & (cdf_at_truth_values <= cdf_max)

    # Keep pit values that are within the min/max range
    pits_clean = cdf_at_truth_values[mask]

    # Determine how many pit values were dropped and warn the user.
    diff = len(cdf_at_truth_values) - len(pits_clean)
    if diff > 0:
        logging.warning("Removed %d PITs from the sample.", diff)

    return pits_clean
