import numpy as np
from ..ensemble import Ensemble
from ..interp_pdf import interp
from .projector_base import ProjectorBase
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal as mvn


class ProjectorShiftsWidths(ProjectorBase):
    """
    Projector for the shifts and widths model.
    The shifts and widths model assumes that the variation in the measured
    photometric distributions can be captured by varying the mean and the
    standard deviation of a fiducial n(z) distribution.

    The calibration method was written by Tilman Tröster. 
    The shift prior is given by a Gaussian distributiob with zero mean
    standard deviation the standard deviation in the mean of 
    the measured photometric distributions.
    The width is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The width prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    This is similar to how the shift prior is calibrated in the shift model.
    """
    def __init__(self, ens):
        self._project_base(ens)
        self._project()

    def _project(self):
        self.shift = self._find_shift()
        self.width = self._find_width()

    def evaluate_model(self, nz, shift, width):
        """
        Aplies a shift and a width to the given p(z) distribution.
        This is done by evluating the n(z) distribution at
        p((z-mu)/width + mu + shift) where mu is the mean redshift
        of the fiducial n(z) distribution and the rescaling by the width.
        Finally the distribution is normalized.
        """
        z = nz[0]
        nz = nz[1]
        nz_i = interp(xvals=z, yvals=nz)
        mu = nz_i.mean().squeeze()

        pdf = nz_i.pdf((z-mu)/width + mu + shift)/width
        norm = np.sum(pdf)
        return [z, pdf/norm]

    def _find_shift(self):
        ms = np.mean(self.nzs, axis=1)  # mean of each nz
        ms_std = np.std(ms)             # std of the means
        return ms_std                   

    def _find_width(self):
        stds = np.std(self.nzs, axis=1)
        std_std = np.std(stds)
        std_mean = np.mean(stds) 
        width = std_std / std_mean
        return width

    def _get_prior(self):
        return mvn([0, 1], [self.shift**2,  self.width**2],
                    allow_singular=True)

    def _get_shift_prior(self):
        return mvn([0], [self.shift**2])

    def _get_width_prior(self):
        return mvn([1], [self.width**2])