import numpy as np
from ..ensemble import Ensemble
from .projector_base import ProjectorBase
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal as mvn


class ProjectorShifts(ProjectorBase):
    """
    Projector for the shifts model.
    The shift model assumes that all the variation in the measured
    photometric distributions can be described by a single shift in
    the position of the mean of a fiducial n(z) distribution.

    This shift is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The shift prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    """
    def __init__(self, ens):
        self._project_base(ens)
        self._project()

    def _project(self):
        self.shift = self._find_shift()

    def evaluate_model(self, nz, shift):
        """
        Aplies a shift to the given p(z) distribution.
        This is done by interpolating the p(z) distribution
        at the shifted z values and then evaluating it at the
        original z values.
        """
        z = nz[0]
        nz = nz[1]
        z_shift = z + shift
        pz_shift = interp1d(z_shift, nz,
                            kind='linear',
                            fill_value='extrapolate')(z)
        return [z, pz_shift]

    def _find_shift(self):
        stds = np.std(self.nzs, axis=1)  # std of each pz
        s_stds = np.std(stds)            # std of the z-std
        m_stds = np.mean(stds)           # mean of the z-std
        return s_stds / m_stds

    def _get_prior(self):
        return mvn([0], [self.shift**2])
