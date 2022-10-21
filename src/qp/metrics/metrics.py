"""This module implements some performance metrics for distribution parameterization"""

from collections import namedtuple
from functools import partial

import numpy as np

import qp.metrics.array_metrics as array_metrics
from qp.utils import epsilon

Grid = namedtuple('Grid', ['grid_values', 'cardinality', 'resolution', 'limits'])

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
            limits: 2-tuple, the limits passed in and used in this function
    """
    cardinality = int((limits[-1] - limits[0]) / dx)
    grid_values = np.linspace(limits[0], limits[1], cardinality)
    resolution = (limits[-1] - limits[0]) / (cardinality - 1)

    return Grid(grid_values, cardinality, resolution, limits)

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
