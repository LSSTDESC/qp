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

import qp
#from . import utils as u
from .utils import infty as default_infty
#from .utils import epsilon as default_eps
from .utils import lims as default_lims

class Ensemble:

    def __init__(self, N, funcform=None, quantiles=None, histogram=None, gridded=None, samples=None, limits=None, scheme='linear', vb=False, procs=None):# where='ensemble.db', procs=None):#
        """
        Creates an object comprised of many qp.PDF objects to efficiently
        perform operations on all of them

        Parameters
        ----------
        N: int
            number of pdfs in the ensemble
        funcform: list of scipy.stats.rv_continuous objects or qp.composite objects, optional
            List of length (npdfs) containing the continuous, parametric forms of the PDFs
        quantiles: tuple of ndarrays, optional
            Pair of arrays of lengths (nquants) and (npdfs, nquants) containing
            shared CDF values and quantiles for each pdf
        histogram: tuple of ndarrays, optional
            Pair of arrays of lengths (nbins+1) and (npdfs, nbins) containing
            shared endpoints of bins and values in bins for each pdf
        gridded: tuple of ndarrays, optional
            Pair of arrays of lengths (npoints) and (npdfs, npoints) containing
            shared points at which pdfs are evaluated and the values of each
            pdf at those points
        samples: ndarray, optional
            Array of size (npdfs, nsamples) containing sampled values
        limits: tuple, float, optional
            shared limits of the space over which the PDFs are defined
        scheme: string, optional
            name of interpolation scheme to use.
        vb: boolean, optional
            report on progress
        where: string
            path to file corresponding to Ensemble, presumed to not yet exist
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

        self.n_pdfs = N
        self.pdf_range = list(range(N))

        if funcform is None and quantiles is None and histogram is None and gridded is None and samples is None:
            print('Warning: initializing an Ensemble object without inputs')
            return

        if limits is None:
            self.limits = default_lims
        else:
            self.limits = limits

        if funcform is None:
            self.mix_mod = [None] * N
        else:
            self.mix_mod = funcform
        if samples is None:
            self.samples = [None] * N
        else:
            self.samples = samples
        if quantiles is None:
            self.quantiles = [None] * N
        else:
            self.quantiles = [(quantiles[0], quantiles[1][i]) for i in self.pdf_range]
        if histogram is None:
            self.histogram = [None] * N
        else:
            self.histogram = [(histogram[0], histogram[1][i]) for i in self.pdf_range]
        if gridded is None:
            self.gridded = (None, [None] * N)
        else:
            self.gridded = (None, [(gridded[0], gridded[1][i]) for i in self.pdf_range])
        # self.mix_mod = None
        self.evaluated = None

        self.scheme = scheme

        self.make_pdfs()

        # self.stacked = {}
        self.klds = {}

    def make_pdfs(self):
        """
        Makes a list of qp.PDF objects based on input
        """
        def make_pdfs_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('making pdf '+str(i)+'\n')
            return qp.PDF( funcform=self.mix_mod[i],
                            quantiles=self.quantiles[i],
                            histogram=self.histogram[i],
                            gridded=self.gridded[-1][i], samples=self.samples[i], limits=self.limits,
                            scheme=self.scheme, vb=False)

        start_time = timeit.default_timer()
        self.pdfs = self.pool.map(make_pdfs_helper, self.pdf_range)
        print(('made the catalog in '+str(timeit.default_timer() - start_time)))


    def sample(self, samps=100, infty=default_infty, using=None, vb=False):
        """
        Samples the pdf in given representation

        Parameters
        ----------
        samps: int, optional
            number of samples to produce
        infty: float, optional
            approximate value at which CDF=1.
        using: string, optional
            Parametrization on which to interpolate, defaults to initialization
        vb: boolean
            report on progress

        Returns
        -------
        samples: ndarray
            array of sampled values

        Notes
        -----
        TODO: change syntax samps --> N
        """
        def sample_helper(i):
            try:
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('sampling pdf '+str(i)+'\n')
                return self.pdfs[i].sample(N=samps, infty=infty, using=using, vb=False)
            except Exception:
                print(('ERROR: sampling failed on '+str(i)+' because '+str(sys.exc_info()[0])))

        self.samples = self.pool.map(sample_helper, self.pdf_range)

        return self.samples

    def quantize(self, quants=None, N=None, limits=None, vb=False):
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
        def quantize_helper(i):
            # try:
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('quantizing pdf '+str(i)+'\n')
            return self.pdfs[i].quantize(quants=quants,
                                            N=N, limits=None, vb=vb)
            # except Exception:
                # print('ERROR: quantization failed on '+str(i)+' because '+str(sys.exc_info()[0]))

        self.quantiles = self.pool.map(quantize_helper, self.pdf_range)
        self.quantiles = np.swapaxes(np.array(self.quantiles), 0, 1)
        self.quantiles = (self.quantiles[0][0], self.quantiles[1])

        return self.quantiles

    def histogramize(self, binends=None, N=10, binrange=None, vb=False):
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
        def histogram_helper(i):
            try:
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('histogramizing pdf '+str(i)+'\n')
                return self.pdfs[i].histogramize(binends=binends, N=N,
                                                binrange=binrange, vb=False)
            except Exception:
                print(('ERROR: histogramization failed on '+str(i)+' because '+str(sys.exc_info()[0])))

        self.histogram = self.pool.map(histogram_helper, self.pdf_range)
        self.histogram = np.swapaxes(np.array(self.histogram), 0, 1)
        self.histogram = (self.histogram[0][0], self.histogram[1])

        return self.histogram

    def mix_mod_fit(self, comps=5, using=None, vb=False):
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
        def mixmod_helper(i):
            try:
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('fitting pdf '+str(i)+'\n')
                return self.pdfs[i].mix_mod_fit(n_components=comps, using=using, vb=False)
            except Exception:
                print(('ERROR: mixture model fitting failed on '+str(i)+' because '+str(sys.exc_info()[0])))

        self.mix_mod = self.pool.map(mixmod_helper, self.pdf_range)

        return self.mix_mod

    def evaluate(self, loc, using=None, norm=False, vb=False):
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
        def evaluate_helper(i):
            try:
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('evaluating pdf '+str(i)+'\n')
                return self.pdfs[i].evaluate(loc=loc, using=using, norm=norm, vb=vb)
            except Exception:
                print(('REAL ERROR: evaluation with '+using+' failed on '+str(i)+' because '+str(sys.exc_info()[0])))
            # return result
        self.gridded = self.pool.map(evaluate_helper, self.pdf_range)
        self.gridded = np.swapaxes(np.array(self.gridded), 0, 1)
        self.gridded = (using, (self.gridded[0][0], self.gridded[1]))

        return self.gridded[-1]

    def integrate(self, limits, using, dx=0.001):
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
            limits = [limits] * self.n_pdfs
        def integrate_helper(i):
            try:
                return self.pdfs[i].integrate(limits[i], using=using, dx=dx, vb=False)
            except Exception:
                print(('ERROR: integration failed on '+str(i)+' because '+str(sys.exc_info()[0])))

        integrals = self.pool.map(integrate_helper, self.pdf_range)

        return integrals

    def moment(self, N, using=None, limits=None, dx=0.01, vb=False):
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
        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        dx = (limits[-1] - limits[0]) / (D - 1)
        grid_to_N = grid ** N

        if self.gridded[0] == using and np.array_equal(self.gridded[-1][0], grid):
            if vb:
                print('taking a shortcut')
            def moment_helper(i):
                return qp.metrics.quick_moment(self.gridded[-1][-1][i], grid_to_N, dx)
        else:
            def moment_helper(i):
                p_eval = self.pdfs[i].evaluate(grid, using=using, vb=vb)[1]
                return qp.metrics.quick_moment(p_eval, grid_to_N, dx)

        moments = self.pool.map(moment_helper, self.pdf_range)

        moments = np.array(moments)
        return moments

    def kld(self, using=None, limits=None, dx=0.01, vb=False):
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
        if self.truth is None:
            print('Metrics can only be calculated relative to the truth.')
            return None

        def P_func(pdf):
            return qp.PDF(truth=pdf.truth, vb=False)

        if limits is None:
            limits = self.limits

        if using == 'quantiles':
            def Q_func(pdf):
                assert pdf.quantiles is not None
                return qp.PDF(quantiles=pdf.quantiles, limits=limits, vb=False)
        elif using == 'histogram':
            def Q_func(pdf):
                assert pdf.histogram is not None
                return qp.PDF(histogram=pdf.histogram, limits=limits, vb=False)
        elif using == 'samples':
            def Q_func(pdf):
                assert pdf.samples is not None
                return qp.PDF(samples=pdf.samples, limits=limits, vb=False)
        elif using == 'gridded':
            def Q_func(pdf):
                assert pdf.gridded is not None
                return qp.PDF(gridded=pdf.gridded, limits=limits, vb=False)
        else:
            print((using + ' not available; try a different parametrization.'))
            return None

        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        self.klds[using] = np.empty(self.n_pdfs)
        if self.gridded[0] == using and np.array_equal(self.gridded[-1][0], grid):
            if vb:
                print('taking a shortcut')
            def kld_helper(i):
                P_eval = P_func(self.pdfs[i]).evaluate(grid, using='truth', vb=vb, norm=True)[-1]
                KL = qp.metrics.quick_kld(P_eval, self.gridded[-1][-1][i], dx=dx)
                self.pdfs[i].klds[using] = KL
                self.klds[using][i] = KL
                return KL
        else:
            def kld_helper(i):
                P_eval = P_func(self.pdfs[i]).evaluate(grid, using='truth', vb=vb, norm=True)[-1]
                Q_eval = Q_func(self.pdfs[i]).evaluate(grid, vb=vb, using=using, norm=True)[-1]
                KL = qp.metrics.quick_kld(P_eval, Q_eval, dx=dx)
                self.pdfs[i].klds[using] = KL
                self.klds[using][i] = KL
                return KL

        klds = self.pool.map(kld_helper, self.pdf_range)
        klds = np.array(klds)

        return self.klds

    def rmse(self, using=None, limits=None, dx=0.01, vb=False):
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
        if self.truth is None:
            print('Metrics can only be calculated relative to the truth.')
            return None

        def P_func(pdf):
            return qp.PDF(truth=pdf.truth, vb=False)

        if limits is None:
            limits = self.limits

        if using == 'quantiles':
            def Q_func(pdfs):
                return qp.PDF(quantiles=pdf.quantiles, vb=False)
        elif using == 'histogram':
            def Q_func(pdfs):
                return qp.PDF(histogram=pdf.histogram, vb=False)
        elif using == 'samples':
            def Q_func(pdfs):
                return qp.PDF(samples=pdf.samples, vb=False)
        elif using == 'gridded':
            def Q_func(pdfs):
                return qp.PDF(quantiles=pdf.gridded, vb=False)
        else:
            print((using + ' not available; try a different parametrization.'))
            return None

        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        dx = (limits[-1] - limits[0]) / (D - 1)

        if self.gridded[0] == using and np.array_equal(self.gridded[-1][0], grid):
            if vb:
                print('taking a shortcut')
            def rmse_helper(i):
                return qp.metrics.quick_rmse(self.gridded[-1][-1][i], grid, dx=dx)
        else:
            def rmse_helper(i):
                P_eval = P_func(self.pdfs[i]).evaluate(grid, norm=True, vb=vb)[-1]
                Q_eval = Q_func(self.pdfs[i]).evaluate(grid, norm=True, vb=vb)[-1]
                return qp.metrics.quick_rmse(P_eval, Q_eval, dx=dx)

        rmses = self.pool.map(rmse_helper, self.pdf_range)

        rmses = np.array(rmses)

        return rmses

    # def stack(self, loc, using, vb=False):
    #     """
    #     Produces an average of the PDFs in the ensemble
    #
    #     Parameters
    #     ----------
    #     loc: ndarray, float or float
    #         location(s) at which to evaluate the PDFs
    #     using: string
    #         which parametrization to use for the approximation
    #     vb: boolean
    #         report on progress
    #
    #     Returns
    #     -------
    #     self.stacked: tuple, ndarray, float
    #         pair of arrays for locations where approximations were evaluated
    #         and the values of the stacked PDFs at those points
    #
    #     Notes
    #     -----
    #     Stacking refers to taking the sum of PDFs evaluated on a shared grid and normalizing it such that it integrates to unity.  This is equivalent to calculating an average probability (based on the PDFs in the ensemble) over the grid.  This probably should be done in a script and not by qp!  The right way to do it would be to call qp.Ensemble.evaluate() and sum those outputs appropriately.
    #     TO DO: make this do something more efficient for mixmod, grid, histogram, samples
    #     TO DO: enable stacking on irregular grid
    #     """
    #     loc_range = max(loc) - min(loc)
    #     delta = loc_range / len(loc)
    #     evaluated = self.evaluate(loc, using=using, norm=True, vb=vb)
    #     stack = np.mean(evaluated[1], axis=0)
    #     stack /= np.sum(stack) * delta
    #     assert(np.isclose(np.sum(stack) * delta, 1.))
    #     self.stacked[using] = (evaluated[0], stack)
    #     return self.stacked

# Note: A copious quantity of commented code has been removed in this commit!
# For future reference, it can still be found here:
#  https://github.com/aimalz/qp/blob/d8d145af9514e29c76e079e869b8b4923f592f40/qp/ensemble.py
# Critical additions still remain.  Metrics of individual qp.PDF objects collected in aggregate over a qp.Ensemble are still desired.
