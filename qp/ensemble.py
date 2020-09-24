"""Implemenation of an ensemble of distributions"""

import numpy as np
import pathos
from pathos.multiprocessing import ProcessingPool as Pool
# import psutil
import timeit
#import os
import sys
# import sqlalchemy
#import scipy.interpolate as spi
#import matplotlib.pyplot as plt


from .utils import lims as default_lims

from .utils import histogramize_dist, integrate_dist, mix_mod_fit_dist_grid, mix_mod_fit_dist_samples

from .metrics import quick_moment, quick_kld, quick_rmse


class Ensemble:
    """An object comprised of many qp.PDF objects to efficiently perform operations on all of them"""


    def __init__(self, dists, limits=None, procs=None):
        """Class constructor

        Creates an object comprised of many qp.PDF objects to efficiently
        perform operations on all of them

        Parameters
        ----------
        dist: list of scipy.stats.rv_continuous
            List containing the distributions
        limits : (float, float) or None
            Limits of support for the PDFs
        procs: int, optional
            limit the number of processors used, otherwise use all available

        Notes
        -----
        The qp.Ensemble object is a wrapper for a collection of qp.PDF
        objects enabling the methods of qp.PDF objects to be applied in parallel.  This is very much a work in progress!  The current version
        holds a list of qp.PDF objects in place.  (Ultimately, we would like the qp.Ensemble object to be a wrapper for a database of
        parameters corresponding to a large collection of PDFs.  The excessive
        quantities of commented code were building toward this ambitious goal
        but have been temporarily abandoned to meet a deadline.)
        TO DO: change dx --> dz (or delta)
        TO DO: standardize n/N
        TO DO: add an option to carry around ID numbers
        """
        start_time = timeit.default_timer()
        if procs is not None:
            self.n_procs = procs
        else:
            self.n_procs = pathos.helpers.cpu_count()
        self.pool = Pool(self.n_procs)
        print(('made the pool of '+str(self.n_procs)+' in '+str(timeit.default_timer() - start_time)))

        self.dists = dists
        self.dist_class = type(self.dists[0])

        if limits is None:
            self.limits = default_lims
        else:
            self.limits = limits

        self.gridded = None
        self.samples = None

    @classmethod
    def create(cls, ctor, data, limits=None, procs=None):
        """
        Makes a list of qp.PDF objects based on input
        """
        def make_dist_helper(sub_data):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('making pdf '+str(i)+'\n')
            return ctor(sub_data)

        if procs is not None:
            n_procs = procs
        else:
            n_procs = pathos.helpers.cpu_count()
        pool = Pool(n_procs)

        start_time = timeit.default_timer()
        dists = pool.map(make_dist_helper, data)
        print(('made the catalog in '+str(timeit.default_timer() - start_time)))
        return cls(dists, limits, n_procs)


    def sample(self, *args, size=1000):
        """
        Samples the pdf in given representation

        Parameters
        ----------
        size: int, optional
            number of samples to produce
        args: passed to the individual distritbutions

        Returns
        -------
        samples: ndarray
            array of sampled values

        Notes
        -----
        TODO: change syntax samps --> N
        """
        def sample_helper(dist):
            try:
                return dist.rvs(*args, size=size)
            except Exception:
                print(('ERROR: sampling failed because '+str(sys.exc_info()[0])))

        self.samples = self.pool.map(sample_helper, self.dists)
        return np.vstack(self.samples)


    def quantize(self, quants):
        """
        Computes an array of evenly-spaced quantiles for each PDF

        Parameters
        ----------
        quants: ndarray, float, optional
            array of quantile locations as decimals
        percent: float, optional
            the separation of the requested quantiles, in percent
        N: int, optional
            the number of quantiles to compute.
        infty: float, optional
            approximate value at which CDF=1.
        vb: boolean
            report on progress

        Returns
        -------
        self.quantiles: ndarray, tuple, ndarray, float
            array of tuples of the CDF values and the quantiles for each PDF
        """
        def quantize_helper(dist):
            return dist.ppf(q=quants)

        quantiles = self.pool.map(quantize_helper, self.dists)
        return np.vstack(quantiles)


    def histogramize(self, bins, normalize=False):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        binends: ndarray, float, optional
            Array of N+1 endpoints of N bins
        N: int, optional
            Number of bins if no binends provided
        binrange: tuple, float, optional
            Pair of values of endpoints of total bin range
        vb: boolean
            Report on progress

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        def histogram_helper(dist):
            try:
                return histogramize_dist(dist, bins, normalize)
            except Exception:
                print(('ERROR: histogramization failed because '+str(sys.exc_info()[0])))

        histogram = self.pool.map(histogram_helper, self.dists)
        histogram = np.swapaxes(np.array(histogram), 0, 1)
        histogram = (histogram[0][0], np.vstack(histogram[1]))

        return histogram

    def mix_mod_fit(self, comps=5):
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        comps: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation
        vb: boolean
            Report progress

        Returns
        -------
        self.mix_mod: list, qp.Composite objects
            list of qp.Composite objects approximating the PDFs

        Notes
        -----
        Currently only supports mixture of Gaussians
        """
        if self.gridded is not None:
            def mixmod_helper(grid_row):
                try:
                    return mix_mod_fit_dist_grid(grid_row, n_components=comps)
                except Exception:
                    print(('ERROR: mixture model fitting failed because '+str(sys.exc_info()[0])))
            mix_mod = self.pool.map(mixmod_helper, self.gridded[1])
        elif self.samples is not None:
            def mixmod_helper(sample_row):
                try:
                    return mix_mod_fit_dist_samples(sample_row, n_components=comps)
                except Exception:
                    print(('ERROR: mixture model fitting failed because '+str(sys.exc_info()[0])))
            mix_mod = self.pool.map(mixmod_helper, self.samples)
        else:
            raise RuntimeError("You must first either evaluate a grid or throw samples to construct a mix_mod representation")
        return mix_mod


    def evaluate(self, x):
        """
        Evaluates all PDFs

        Parameters
        ----------
        loc: float or ndarray, float
            location(s) at which to evaluate the pdfs
        using: string
            which parametrization to evaluate, defaults to initialization
        norm: boolean, optional
            True to normalize the evaluation, False if expected probability outside loc
        vb: boolean
            report on progress

        Returns
        -------
        self.gridded: tuple(string, tuple(ndarray, ndarray, float))
            tuple of string and tuple of grid and values of the PDFs (or their approximations) at the requested location(s), of shape (npdfs, nlocs)
        """
        def evaluate_helper(dist):
            try:
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('evaluating pdf '+str(i)+'\n')
                return dist.pdf(x)
            except Exception:
                print(('REAL ERROR: evaluation with failed because '+str(sys.exc_info()[0])))
            # return result
        self.gridded = self.pool.map(evaluate_helper, self.dists)
        self.gridded = (x, np.vstack(self.gridded))
        return self.gridded


    def integrate(self, limits):
        """
        Computes the integral under the ensemble of PDFs between the given limits.

        Parameters
        ----------
        limits: numpy.ndarray, tuple, float
            limits of integration, may be different for all PDFs in the ensemble
        using: string
            parametrization over which to approximate the integral
        dx: float, optional
            granularity of integral

        Returns
        -------
        integral: numpy.ndarray, float
            value of the integral
        """
        if len(np.shape(limits)) == 1:
            limits = [limits] * len(self.dists)
        def integrate_helper(dist, lims):
            try:
                return integrate_dist(dist, lims)
            except Exception:
                print(('ERROR: integration failed because '+str(sys.exc_info()[0])))

        integrals = self.pool.map(integrate_helper, self.dists, limits)
        return integrals


    def moment(self, N, limits=None, dx=0.01, vb=False):
        """
        Calculates a given moment for each PDF in the ensemble

        Parameters
        ----------
        N: int
            number of moment
        using: string
            which parametrization to use
        limits: tuple of floats, optional
            endpoints of integration interval in which to calculate moment
        dx: float
            resolution of integration grid
        vb: boolean
            print progress to stdout?

        Returns
        -------
        moments: numpy.ndarray, float
            moment values of each PDF under the using approximation or truth
        """
        if limits is None:
            limits = self.limits

        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        dx = (limits[-1] - limits[0]) / (D - 1)
        grid_to_N = grid ** N

        if self.gridded is not None and np.array_equal(self.gridded[0], grid):
            if vb:
                print('taking a shortcut')
            def moment_helper(grid_row):
                return quick_moment(grid_row, grid_to_N, dx)
            moments = self.pool.map(moment_helper, self.gridded[1])
        else:
            def moment_helper(dist):
                p_eval = dist.pdf(grid)
                return quick_moment(p_eval, grid_to_N, dx)
            moments = self.pool.map(moment_helper, self.dists)
        moments = np.array(moments)
        return moments


    def kld(self, other, limits=None, dx=0.01, vb=False):
        """
        Calculates the KLD for each PDF in the ensemble

        Parameters
        ----------
        using: string
            which parametrization to use
        limits: tuple of floats, optional
            endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid
        vb: boolean
            print progress to stdout?

        Returns
        -------
        klds: numpy.ndarray, float
            KLD values of each PDF under the using approximation relative to the truth
        """
        if other is None:
            print('Metrics can only be calculated relative another distribution.')
            return None

        if limits is None:
            limits = self.limits

        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        if self.gridded is not None and np.array_equal(self.gridded[0], grid):
            if vb:
                print('taking a shortcut')
            def kld_helper(grid_row, other_dist):
                P_eval = other_dist.pdf(grid)
                KL = quick_kld(P_eval, grid_row, dx=dx)
                return KL
            klds = self.pool.map(kld_helper, self.gridded[1], other.dists)
        else:
            def kld_helper(dist, other_dist):
                P_eval = other_dist.pdf(grid)
                Q_eval = dist.pdf(grid)
                KL = quick_kld(P_eval, Q_eval, dx=dx)
                return KL
            klds = self.pool.map(kld_helper, self.dists, other.dists)
        klds = np.array(klds)
        return klds


    def rmse(self, other, limits=None, dx=0.01, vb=False):
        """
        Calculates the RMSE for each PDF in the ensemble

        Parameters
        ----------
        using: string
            which parametrization to use
        limits: tuple of floats
            endpoints of integration interval in which to calculate RMSE
        dx: float
            resolution of integration grid
        vb: boolean
            print progress to stdout?

        Returns
        -------
        rmses: numpy.ndarray, float
            RMSE values of each PDF under the using approximation relative to the truth
        """
        if other is None:
            print('Metrics can only be calculated relative another distribution.')
            return None

        if limits is None:
            limits = self.limits

        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        dx = (limits[-1] - limits[0]) / (D - 1)

        if self.gridded is not None and np.array_equal(self.gridded[0], grid):
            if vb:
                print('taking a shortcut')
            def rmse_helper(grid_row, other_dist):
                P_eval = other_dist.pdf(grid)
                return quick_rmse(P_eval, grid_row, N=D)
            rmses = self.pool.map(rmse_helper, self.gridded[1], other.dists)
        else:
            def rmse_helper(dist, other_dist):
                P_eval = other_dist.pdf(grid)
                Q_eval = dist.pdf(grid)
                return quick_rmse(P_eval, Q_eval, N=D)
            rmses = self.pool.map(rmse_helper, self.dists, other.dists)

        rmses = np.array(rmses)
        return rmses
