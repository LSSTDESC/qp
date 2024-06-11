from ..ensemble import Ensemble
import numpy as np
from multipledispatch import dispatch


class ProjectorBase(object):
    @dispatch()
    def __init__(self):
        self._project_base()
        self._project()

    @dispatch(np.ndarray, np.ndarray)
    def __init__(self, zs, pzs):
        self._project_base(zs, pzs)

    @dispatch(Ensemble)
    def __init__(self, ens):
        self._project_base(ens)

    @dispatch()
    def _project_base(self):
        raise NotImplementedError

    @dispatch(np.ndarray, np.ndarray)
    def _project_base(self, zs, pzs):
        self.pzs = self._normalize(pzs)
        self.z = zs
        self.pz_mean = np.mean(self.pzs, axis=0)
        self.prior = None

    @dispatch(Ensemble)
    def _project_base(self, ens, z=None):
        if z is None:
            z = np.linspace(0, 1.5, 45)
        self.z = z
        pzs = ens.pdf(z)
        pzs = ens.objdata()['pdfs']
        self.pzs = self._normalize(pzs)
        self.pz_mean = np.mean(self.pzs, axis=0)
        self.prior = None

    def _normalize(self, pzs):
        norms = np.sum(pzs, axis=1)
        pzs /= norms[:, None]  
        return pzs

    def evaluate_model(self, *args):
        raise NotImplementedError

    def get_prior(self):
        if self.prior is None:
            self.prior = self._get_prior()
        return self.prior

    def sample_prior(self):
        prior = self.get_prior()
        return prior.rvs()

    def save_prior(self):
        raise NotImplementedError
