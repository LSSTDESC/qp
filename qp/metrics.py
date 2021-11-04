"""This module implements some performance metrics for distribution parameterization"""

import numpy as np

from qp.utils import safelog, epsilon

def calculate_moment(p, N, limits, dx=0.01):
    """
    Calculates a moment of a qp.PDF object

    Parameters
    ----------
    p: qp.PDF object
        the PDF whose moment will be calculated
    N: int
        order of the moment to be calculated
    limits: tuple of floats
        endpoints of integration interval over which to calculate moments
    dx: float
        resolution of integration grid
    vb: Boolean
        print progress to stdout?

    Returns
    -------
    M: float
        value of the moment
    """
    # Make a grid from the limits and resolution
    d = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], d)
    dx = (limits[-1] - limits[0]) / (d - 1)
    # Evaluate the functions on the grid
    pe = p.gridded(grid)[1]
    # calculate the moment
    grid_to_N = grid ** N
    M = quick_moment(pe, grid_to_N, dx)
    return M

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

def calculate_kld(p, q, limits, dx=0.01):
    """
    Calculates the Kullback-Leibler Divergence between two qp.PDF objects.

    Parameters
    ----------
    p: PDF object
        probability distribution whose distance _from_ `q` will be calculated.
    q: PDF object
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
    TO DO: change this to calculate_kld
    TO DO: have this take number of points not dx!
    """
    # Make a grid from the limits and resolution
    N = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], N)
    dx = (limits[-1] - limits[0]) / (N - 1)
    # Evaluate the functions on the grid and normalize
    pe = p.gridded(grid)
    pn = pe[1]
    qe = q.gridded(grid)
    qn = qe[1]
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    # pn = pe / np.sum(pe)
    #denominator = max(np.sum(qe), epsilon)
    # qn = qe / np.sum(qe)#denominator
    # Compute the log of the normalized PDFs
    # logquotient = safelog(pn / qn)
    # logp = safelog(pn)
    # logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = quick_kld(pn, qn, dx=dx)# np.dot(pn * logquotient, np.ones(len(grid)) * dx)
    if Dpq.any() < 0.: #pragma: no cover
        print('broken KLD: '+str((Dpq, pn, qn, dx)))
        Dpq = epsilon*np.ones(Dpq.shape)
    return Dpq

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

    Notes
    -----
    TO DO: change this to quick_kld
    """
    logquotient = safelog(p_eval) - safelog(q_eval)
    # logp = safelog(pn)
    # logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = dx * np.sum(p_eval * logquotient, axis=-1)
    return Dpq

def calculate_rmse(p, q, limits, dx=0.01):
    """
    Calculates the Root Mean Square Error between two qp.PDF objects.

    Parameters
    ----------
    p: PDF object
        probability distribution function whose distance between its truth and the approximation of `q` will be calculated.
    q: PDF object
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
    # Make a grid from the limits and resolution
    N = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], N)
    dx = (limits[-1] - limits[0]) / (N - 1)
    # Evaluate the functions on the grid
    pe = p.gridded(grid)[1]
    qe = q.gridded(grid)[1]
    # Calculate the RMS between p and q
    rms = quick_rmse(pe, qe, N)# np.sqrt(dx * np.sum((pe - qe) ** 2))
    return rms

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
