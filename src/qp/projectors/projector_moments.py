import numpy as np
from ..ensemble import Ensemble
from multipledispatch import dispatch
from .projector_base import ProjectorBase
from numpy.linalg import eig, cholesky
from scipy.stats import multivariate_normal as mvn


class ProjectorMoments(ProjectorBase):
    """
    Projector for the moments model.
    The moments model assumes that meausred photometric distribution
    is Gaussian meaning that it can be fully described by its mean and
    covariance matrix. Conceptually, this is equavalent to a 
    Gaussian process regressio for a given p(z). The details can be found 
    in the paper: 2301.11978

    Some measured photometric distributions will possess non-invertible
    covariance matrices. If this is the case, ProjectorMoments will
    attempt regularize the covariance matrix by adding a small jitter
    to its eigen-values. If this fails, the covariance matrix will be
    diagonalized.
    """
    @dispatch()
    def __init__(self):
        self._project_base()
        self._project()

    @dispatch(np.ndarray, np.ndarray)
    def __init__(self, zs, pzs):
        self._project_base(zs, pzs)
        self._project()

    @dispatch(Ensemble)
    def __init__(self, ens):
        self._project_base(ens)
        self._project()

    def _project(self):
        self.pz_cov = self._get_cov()
        self.pz_chol = cholesky(self.pz_cov)

    def _get_cov(self):
        cov = np.cov(self.pzs, rowvar=False)
        if not self._is_pos_def(cov):
            print('Warning: Covariance matrix is not positive definite')
            print('The covariance matrix will be regularized')
            jitter = 1e-15 * np.eye(cov.shape[0])
            w, v = eig(cov+jitter)
            w = np.real(np.abs(w))
            v = np.real(v)
            cov = v @ np.diag(np.abs(w)) @ v.T
            cov = np.tril(cov) + np.triu(cov.T, 1)
            if not self._is_pos_def(cov):
                print('Warning: regularization failed')
                print('The covariance matrix will be diagonalized')
                cov = np.diag(np.diag(cov))
        return cov

    def _is_pos_def(self, A):
        return np.all(np.linalg.eigvals(A) > 0)

    def evaluate_model(self, pz, alpha):
        """
        Samples a photometric distribution 
        from a Gaussian distribution with mean
        and covariance measured from the data.
        """
        z = pz[0]
        pz = pz[1]
        return [z, pz + self.pz_chol @ alpha]

    def _get_prior(self):
        return mvn(np.zeros_like(self.pz_mean),
                   np.ones_like(self.pz_mean))

