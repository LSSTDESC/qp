import numpy as np

import qp

def calculate_moment(p, N, using=None, limits=None, dx=0.01, vb=False):
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
    if limits is None:
        limits = p.limits
    if using is None:
        using = p.first
    # Make a grid from the limits and resolution
    d = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], d)
    dx = (limits[-1] - limits[0]) / (d - 1)
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, using=using, vb=vb)[1]
    # pe = normalize_gridded(pe)[1]
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
    M = np.dot(grid_to_N, p_eval) * dx
    return M

def calculate_kld(p, q, limits=qp.utils.lims, dx=0.01, vb=False):
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
    vb: boolean
        report on progress to stdout?

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
    pe = p.evaluate(grid, vb=vb, norm=True)
    pn = pe[1]
    qe = q.evaluate(grid, vb=vb, norm=True)
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
    if Dpq < 0.:
        print(('broken KLD: '+str((Dpq, pn, qn, dx))))
        Dpq = qp.utils.epsilon
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
    logquotient = qp.utils.safelog(p_eval) - qp.utils.safelog(q_eval)
    # logp = safelog(pn)
    # logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = dx * np.sum(p_eval * logquotient)
    return Dpq

def calculate_rmse(p, q, limits=qp.utils.lims, dx=0.01, vb=False):
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
    vb: boolean
        report on progress to stdout?

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
    pe = p.evaluate(grid, vb=vb)[1]
    qe = q.evaluate(grid, vb=vb)[1]
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
    rms = np.sqrt(np.sum((p_eval - q_eval) ** 2) / N)
    return rms
