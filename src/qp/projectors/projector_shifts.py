import numpy as np
from ..ensemble import Ensemble
from multipledispatch import dispatch
from .projector_base import ProjectorBase
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal


class ProjectorShifts(ProjectorBase):
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
        self.shift = self._find_shift()

    def evaluate_model(self, pz, shift):
        """
        Aplies a shift to the given p(z) distribution.
        This is done by interpolating the p(z) distribution
        at the shifted z values and then evaluating it at the
        original z values.
        """
        z = pz[0]
        pz = pz[1]
        z_shift = z + shift
        pz_shift = interp1d(z_shift, pz,
                            kind='linear',
                            fill_value='extrapolate')(z)
        return [z, pz_shift]

    def _find_shift(self):
        stds = np.std(self.pzs, axis=1)  # std of each pz
        s_stds = np.std(stds)            # std of the z-std
        m_stds = np.mean(stds)           # mean of the z-std
        return s_stds / m_stds

    def _get_prior(self):
        return multivariate_normal([0], [self.shift**2])
