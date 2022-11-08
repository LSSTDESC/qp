"""This module implements metric calculations that are independent of qp.Ensembles"""

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from qp.utils import safelog

def quick_anderson_darling(p_random_variables, scipy_distribution='norm'):
    """Calculate the Anderson-Darling statistic using scipy.stats.anderson for one CDF vs a scipy distribution.
    For more details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html

    Parameters
    ----------
    p_random_variables : np.array
        An array of random variables from the given distribution
    scipy_distribution : {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1'}, optional
        The type of distribution to test against.

    Returns
    -------
    [Result objects]
        A array of objects with attributes ``statistic``, ``critical_values``, and ``significance_level``.
    """
    return stats.anderson(p_random_variables, dist=scipy_distribution)

def quick_anderson_ksamp(p_random_variables, q_random_variables, **kwargs):
    """Calculate the k-sample Anderson-Darling statistic using scipy.stats.anderson_ksamp for two CDFs. 
    For more details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html

    Parameters
    ----------
    p_random_variables : np.array
        An array of random variables from the given distribution
    q_random_variables : np.array
        An array of random variables from the given distribution

    Returns
    -------
    [Result objects]
        A array of objects with attributes ``statistic``, ``critical_values``, and ``significance_level``.
    """
    return stats.anderson_ksamp([p_random_variables, q_random_variables], **kwargs)

def quick_cramer_von_mises(p_random_variables, q_cdf, **kwargs):
    """Calculate the Cramer von Mises statistic using scipy.stats.cramervonmises for each pair of distributions
    in two input Ensembles. For more details see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html


    Parameters
    ----------
    p_random_variables : np.array
        An array of random variables from the given distribution
    q_cdf : callable
        A function to calculate the CDF of a given distribution

    Returns
    -------
    [Result objects]
        A array of objects with attributes ``statistic`` and ``pvalue``.
    """

    return stats.cramervonmises(p_random_variables, q_cdf, **kwargs)

def quick_kld(p_eval, q_eval, dx=0.01):
    """
    Calculates the Kullback-Leibler Divergence between two evaluations of PDFs.

    Parameters
    ----------
    p_eval: numpy.ndarray, float
        evaluations of probability distribution whose distance _from_ `q` will be calculated
    q_eval: numpy.ndarray, float
        evaluations of probability distribution whose distance _to_ `p` will be calculated.
    dx: float
        resolution of integration grid

    Returns
    -------
    Dpq: float
        the value of the Kullback-Leibler Divergence from `q` to `p`
    """

    # safelog would be easy to isolate if array_metrics is ever extracted
    logquotient = safelog(p_eval) - safelog(q_eval)

    # Calculate the KLD from q to p
    Dpq = dx * np.sum(p_eval * logquotient, axis=-1)
    return Dpq

def quick_kolmogorov_smirnov(p_rvs, q_cdf, num_samples=100, **kwargs):
    """Calculate the Kolmogorov-Smirnov statistic using scipy.stats.kstest for each pair of distributions
    in two input Ensembles. For more details see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html 

    Parameters
    ----------
    p_rvs : callable
        A function to generate random variables for the given distribution
    q_cdf : callable
        A function to calculate the CDF of a given distribution
    num_samples : int, optional
        Number of samples to use in the calculation

    Returns
    -------
    [KstestResult]
        A array of KstestResult objects with attributes ``statistic`` and ``pvalue``.
    """

    return stats.kstest(p_rvs, q_cdf, N=num_samples, **kwargs)

def quick_moment(p_eval, grid_to_N, dx):
    """
    Calculates a moment of an evaluated PDF

    Parameters
    ----------
    p_eval: numpy.ndarray, float
        the values of a probability distribution
    grid: numpy.ndarray, float
        the grid upon which p_eval was evaluated
    dx: float
        the difference between regular grid points
    N: int
        order of the moment to be calculated

    Returns
    -------
    M: float
        value of the moment
    """
    M = np.dot(p_eval, grid_to_N) * dx
    return M

def quick_rmse(p_eval, q_eval, N):
    """
    Calculates the Root Mean Square Error between two evaluations of PDFs.

    Parameters
    ----------
    p_eval: numpy.ndarray, float
        evaluation of probability distribution function whose distance between
        its truth and the approximation of `q` will be calculated.
    q_eval: numpy.ndarray, float
        evaluation of probability distribution function whose distance between
        its approximation and the truth of `p` will be calculated.
    N: int
        number of points at which PDFs were evaluated

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`
    """
    # Calculate the RMS between p and q
    rms = np.sqrt(np.sum((p_eval - q_eval) ** 2, axis=-1) / N)
    return rms

def quick_rbpe(pdf_function, integration_bounds, limits=(np.inf, np.inf)):
    """
    Calculates the risk based point estimate of a qp.Ensemble object with npdf == 1.

    Parameters
    ----------
    pdf_function, python function
        The function should calculate the value of a pdf at a given x value
    integration_bounds, 2-tuple of floats
        The integration bounds - typically (ppf(0.01), ppf(0.99)) for the given distribution
    limits, tuple of floats
        The limits at which to evaluate possible z_best estimates.
        If custom limits are not provided then all potential z value will be
        considered using the scipy.optimize.minimize_scalar function.

    Returns
    -------
    rbpe: float
        The risk based point estimate of the provided ensemble.
    """

    def calculate_loss(x):
        return 1.0 - (1.0 / (1.0 + (pow((x / 0.15), 2))))

    lower = integration_bounds[0]
    upper = integration_bounds[1]

    def find_z_risk(zp):
        def integrand(z):
            return pdf_function(z) * calculate_loss((zp - z) / (1.0 + z))

        return quad(integrand, lower, upper)[0]

    if limits[0] == np.inf:
        return minimize_scalar(find_z_risk).x
    return minimize_scalar(
        find_z_risk, bounds=(limits[0], limits[1]), method="bounded"
    ).x
