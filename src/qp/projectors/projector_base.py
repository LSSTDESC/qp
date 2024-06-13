from ..ensemble import Ensemble
import numpy as np


class ProjectorBase(object):
    """
    Base class for projectors. Projectors are used to project the measured
    photometric distributions by RAIL onto the space of a given generative
    photometric model for inference. 
    This class is not meant to be used directly, 
    but to be subclassed by specific projectors. 
    The subclasses should implement the following methods:
    - evaluate_model: given a set of parameters, evaluate the model
    - get_prior: return the prior distribution of the model given
    the meadured photometric distributions.
    """
    def __init__(self, ens):
        self._project_base(ens)

    def _project_base(self, ens, z=None):
        if z is None:
            z = np.linspace(0, 1.5, 45)
        self.z = z
        nzs = ens.pdf(z)
        nzs = ens.objdata()['pdfs']
        self.ens = ens
        self.nzs = self._normalize(nzs)
        self.nz_mean = np.mean(self.nzs, axis=0)
        self.nz_cov = np.cov(self.nzs, rowvar=False)
        self.prior = None

    def _normalize(self, nzs):
        norms = np.sum(nzs, axis=1)
        nzs /= norms[:, None]  
        return nzs

    def evaluate_model(self, *args):
        """
        Evaluate the model at the given parameters.
        """
        raise NotImplementedError

    def get_prior(self):
        """
        Returns the calibrated prior distribution for the model 
        parameters given the measured photometric distributions.
        """
        if self.prior is None:
            self.prior = self._get_prior()
        return self.prior

    def sample_prior(self):
        """
        Draws a sample from the prior distribution.
        """
        prior = self.get_prior()
        return prior.rvs()

    def save_prior(self, mode="dist"):
        """
        Saves the prior distribution to a file.
        """
        prior = self.get_prior()
        if mode == "dist":
            return prior
        if mode == "file":
            prior = self.get_prior()
            np.save("prior_mean.npy", prior.mean)
            np.save("prior_cov.npy", prior.cov)
        else:
            raise NotImplementedError
