"""
.. module:: sparse_rep
.. moduleauthor:: Matias Carrasco Kind
.. moduleauthor:: Johann Cohen-Tanugi
The original SparsePZ code to be found at https://github.com/mgckind/SparsePz
This module reorganizes it for usage by DESC within qp, and is python3 compliant.
"""

__author__ = "Matias Carrasco Kind"

import numpy as np
from scipy.special import voigt_profile  # pylint: disable=no-name-in-module
from scipy import linalg as sla
from scipy import integrate as sciint


def shapes2pdf(wa, ma, sa, ga, meta, cut=1.0e-5):  # pylint: disable=too-many-arguments
    """return a pdf evaluated at the meta['xvals'] values for the
    given set of Voigt parameters"""
    # input : list of shape parameters for a single object
    x = meta["xvals"]
    pdf = np.zeros_like(x)
    for w, m, s, g in zip(wa, ma, sa, ga):
        pdft = voigt_profile(x - m, s, g)
        pdft = np.where(pdft >= cut, pdft, 0.0)
        pdft = w * pdft / sla.norm(pdft)
        pdf += pdft
    pdf = np.where(pdf >= cut, pdf, 0.0)
    return pdf / sciint.trapz(pdf, x)


def create_basis(metadata, cut=1.0e-5):
    """create the Voigt basis matrix out of a metadata dictionary"""
    mu = metadata["mu"]
    Nmu = metadata["dims"][0]
    sigma = metadata["sig"]
    Nsigma = metadata["dims"][1]
    Nv = metadata["dims"][2]
    xvals = metadata["xvals"]
    return create_voigt_basis(xvals, mu, Nmu, sigma, Nsigma, Nv, cut=cut)


def create_voigt_basis(xvals, mu, Nmu, sigma, Nsigma, Nv, cut=1.0e-5):  # pylint: disable=too-many-arguments
    """
    Creates a gaussian-voigt dictionary at the same resolution as the original PDF

    :param float xvals: the x-axis point values for the PDF
    :param float mu: [min_mu, max_mu], range of mean for gaussian
    :param int Nmu: Number of values between min_mu and max_mu
    :param float sigma: [min_sigma, max_sigma], range of variance for gaussian
    :param int Nsigma: Number of values between min_sigma and max_sigma
    :param Nv: Number of Voigt profiles per gaussian at given position mu and sigma
    :param float cut: Lower cut for gaussians

    :return: Dictionary as numpy array with shape (len(xvals), Nmu*Nsigma*Nv)
    :rtype: float

    """

    means = np.linspace(mu[0], mu[1], Nmu)
    sig = np.linspace(sigma[0], sigma[1], Nsigma)
    gamma = np.linspace(0, 0.5, Nv)
    NA = Nmu * Nsigma * Nv
    Npdf = len(xvals)
    A = np.zeros((Npdf, NA))
    kk = 0
    for i in range(Nmu):
        for j in range(Nsigma):
            for k in range(Nv):
                pdft = voigt_profile(xvals - means[i], sig[j], gamma[k])
                pdft = np.where(pdft >= cut, pdft, 0.0)
                A[:, kk] = pdft / sla.norm(pdft)
                kk += 1
    return A


def sparse_basis(dictionary, query_vec, n_basis, tolerance=None):
    """
    Compute sparse representation of a vector given Dictionary  (basis)
    for a given tolerance or number of basis. It uses Cholesky decomposition to speed the process and to
    solve the linear operations adapted from
    Rubinstein, R., Zibulevsky, M. and Elad, M., Technical Report - CS
    Technion, April 2008

    :param float dictionary: Array with all basis on each column,
           must has shape (len(vector), total basis) and each column must have euclidean l-2 norm equal to 1
    :param float query_vec: vector of which a sparse representation is desired
    :param int n_basis: number of desired basis
    :param float tolerance: tolerance desired if n_basis is not needed to be fixed, must input a large number
           for n_basis to assure achieving tolerance

    :return: indices, values (2 arrays one with the position and the second with the coefficients)
    """

    a_n = np.zeros(dictionary.shape[1])
    machine_eps = np.finfo(dictionary.dtype).eps
    alpha = np.dot(dictionary.T, query_vec)
    res = query_vec
    idxs = np.arange(dictionary.shape[1])  # keeping track of swapping
    L = np.zeros((n_basis, n_basis), dtype=dictionary.dtype)
    L[0, 0] = 1.0
    for n_active in range(n_basis):
        lam = np.argmax(abs(np.dot(dictionary.T, res)))
        if lam < n_active or alpha[lam] ** 2 < machine_eps:  # pragma: no cover
            n_active -= 1
            break
        if n_active > 0:  # pragma: no cover
            # Updates the Cholesky decomposition of dictionary
            L[n_active, :n_active] = np.dot(
                dictionary[:, :n_active].T, dictionary[:, lam]
            )
            sla.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                lower=True,
                overwrite_b=True,
            )
            v = sla.norm(L[n_active, :n_active]) ** 2
            if 1 - v <= machine_eps:
                print("Selected basis are dependent or normed are not unity")
                break
            L[n_active, n_active] = np.sqrt(1 - v)
        dictionary[:, [n_active, lam]] = dictionary[:, [lam, n_active]]
        alpha[[n_active, lam]] = alpha[[lam, n_active]]
        idxs[[n_active, lam]] = idxs[[lam, n_active]]
        # solves LL'x = query_vec as a composition of two triangular systems
        gamma = sla.cho_solve(
            (L[: n_active + 1, : n_active + 1], True),
            alpha[: n_active + 1],
            overwrite_b=False,
        )
        res = query_vec - np.dot(dictionary[:, : n_active + 1], gamma)
        if tolerance is not None and sla.norm(res) ** 2 <= tolerance:
            break
    a_n[idxs[: n_active + 1]] = gamma
    del dictionary
    # return a_n
    return idxs[: n_active + 1], gamma


def combine_int(Ncoef, Nbase):
    """
    combine index of base (up to 62500 bases) and value (16 bits integer with sign) in a 32 bit integer
    First half of word is for the value and second half for the index

    :param int Ncoef: Integer with sign to represent the value associated with a base, 
           this is a sign 16 bits integer
    :param int Nbase: Integer representing the base, unsigned 16 bits integer
    :return: 32 bits integer
    """
    return (Ncoef << 16) | Nbase


def get_N(longN):
    """
    Extract coefficients fro the 32bits integer,
    Extract Ncoef and Nbase from 32 bit integer
    return (longN >> 16), longN & 0xffff

    :param int longN: input 32 bits integer

    :return: Ncoef, Nbase both 16 bits integer
    """
    return (longN >> 16), (longN & (2**16 - 1))


def decode_sparse_indices(indices):
    """decode sparse indices into basis indices and weigth array"""
    Ncoef = 32001
    sp_ind = np.array(list(map(get_N, indices)))
    spi = sp_ind[:, 0, :]
    dVals = 1.0 / (Ncoef - 1)
    vals = spi * dVals
    vals[:, 0] = 1.0
    return sp_ind[:, 1, :], vals


def indices2shapes(sparse_indices, meta):
    """compute the Voigt shape parameters from the sparse index

    Parameters
    ----------
    sparse_index: `np.array`
        1D Array of indices for each object in the ensemble

    meta: `dict`
        Dictionary of metadata to decode the sparse indices
    """
    Nmu = meta["dims"][0]
    Nsigma = meta["dims"][1]
    Nv = meta["dims"][2]
    Ncoef = meta["dims"][3]
    mu = meta["mu"]
    sigma = meta["sig"]

    means_array = np.linspace(mu[0], mu[1], Nmu)
    sig_array = np.linspace(sigma[0], sigma[1], Nsigma)
    gam_array = np.linspace(0, 0.5, Nv)

    # split the sparse indices into pairs (weight, basis_index)
    # for each sparse index corresponding to one of the basis function
    sp_ind = np.array(list(map(get_N, sparse_indices)))

    spi = sp_ind[:, 0, :]
    dVals = 1.0 / (Ncoef - 1)
    vals = spi * dVals
    vals[:, 0] = 1.0

    Dind2 = sp_ind[:, 1, :]
    means = means_array[np.array(Dind2 / (Nsigma * Nv), int)]
    sigmas = sig_array[np.array((Dind2 % (Nsigma * Nv)) / Nv, int)]
    gammas = gam_array[np.array((Dind2 % (Nsigma * Nv)) % Nv, int)]

    return vals, means, sigmas, gammas


def build_sparse_representation(  # pylint: disable=too-many-arguments
    x,
    P,
    mu=None,
    Nmu=None,
    sig=None,
    Nsig=None,
    Nv=3,
    Nsparse=20,
    tol=1.0e-10,
    verbose=True,
):
    """compute the sparse representation of a set of pdfs evaluated on a common x array"""
    # Note : the range for gamma is fixed to [0, 0.5] in create_voigt_basis
    Ntot = len(P)
    if verbose:
        print("Total Galaxies = ", Ntot)
    dx = x[1] - x[0]
    if mu is None:
        mu = [min(x), max(x)]
    if Nmu is None:
        Nmu = len(x)
    if sig is None:
        max_sig = (max(x) - min(x)) / 12.0
        min_sig = dx / 6.0
        sig = [min_sig, max_sig]
    if Nsig is None:
        Nsig = int(np.ceil(2.0 * (max_sig - min_sig) / dx))

    if verbose:
        print("dx = ", dx)
        print("Nmu, Nsig, Nv = ", "[", Nmu, ",", Nsig, ",", Nv, "]")
        print("Total bases in dictionary", Nmu * Nsig * Nv)
        print("Nsparse (number of bases) = ", Nsparse)
        # Create dictionary
        print("Creating Dictionary...")

    A = create_voigt_basis(x, mu, Nmu, sig, Nsig, Nv)
    bigD = {}

    Ncoef = 32001
    AA = np.linspace(0, 1, Ncoef)
    Da = AA[1] - AA[0]

    bigD["xvals"] = x
    bigD["mu"] = mu
    bigD["sig"] = sig
    bigD["dims"] = [Nmu, Nsig, Nv, Ncoef, Nsparse]
    bigD["Ntot"] = Ntot
    if verbose:
        print("Creating Sparse representation...")

    Sparse_Array = np.zeros((Ntot, Nsparse), dtype="int")
    for k in range(Ntot):
        pdf0 = P[k]

        Dind, Dval = sparse_basis(A, pdf0, Nsparse, tolerance=tol)

        if len(Dind) < 1:  # pragma: no cover
            continue
        # bigD[k]['sparse'] = [Dind, Dval]
        if max(Dval) > 0:
            dval0 = Dval[0]
            Dvalm = Dval / np.max(Dval)
            index = np.array(list(map(round, (Dvalm / Da))), dtype="int")
            index0 = int(round(dval0 / Da))
            index[0] = index0
        else:
            index = np.zeros(len(Dind), dtype="int")  # pragma: no cover

        sparse_ind = np.array(list(map(combine_int, index, Dind)))
        Sparse_Array[k, 0 : len(sparse_ind)] = sparse_ind

        # swap back columns
        A[:, [Dind]] = A[:, [np.arange(len(Dind))]]

    if verbose:
        print("done")
    return Sparse_Array, bigD, A


def pdf_from_sparse(sparse_indices, A, xvals, cut=1.0e-5):
    """return the array of evaluations at xvals from the sparse indices"""
    indices, vals = decode_sparse_indices(sparse_indices)
    pdf_y = (A[:, indices] * vals).sum(axis=-1)
    pdf_y = np.where(pdf_y >= cut, pdf_y, 0.0)
    pdf_x = xvals
    norms = sciint.trapz(pdf_y.T, pdf_x)
    pdf_y /= norms
    return pdf_y
