import numpy as np
import bisect
import scipy.stats as sps
import scipy.interpolate as spi
import scipy.optimize as spo
import sklearn as skl
from sklearn import mixture

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['savefig.dpi'] = 250
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'

import qp
from qp.utils import infty as default_infty
from qp.utils import epsilon as default_eps
from qp.utils import lims as default_lims

class PDF(object):

    def __init__(self, funcform=None,
                 quantiles=None, histogram=None,
                 gridded=None, samples=None, limits=default_lims, scheme='linear',
                 vb=False):
        """
        An object representing a probability density function in
        various ways.

        Parameters
        ----------
        # truth: scipy.stats.rv_continuous object or qp.composite object, optional
        #     Continuous, parametric form of the PDF
        quantiles: tuple of ndarrays, optional
            Pair of arrays of lengths (nquants, nquants) containing CDF
            values and quantiles
        histogram: tuple of ndarrays, optional
            Pair of arrays of lengths (nbins+1, nbins) containing
            endpoints of bins and values in bins
        gridded: tuple of ndarrays, optional
            Pair of arrays of lengths (npoints, npoints) containing
            points at which function is evaluated and function values
            at those points
        samples: ndarray, optional
            Array of length nsamples containing sampled values
        limits: tuple, float, optional
            limits past which PDF is considered to be 0.
        scheme: string or int, optional
            name of interpolation scheme to use, or order of spline interpolation.
        vb: boolean
            report on progress to stdout?

        Notes
        -----
        TO DO: enable truth to be any parametrization
        TO DO: change dx --> dz (or delta)
        TO DO: consider changing quantiles input to just be the z-values since interpolation gives the p(z) values anyway
        """
        # self.truth = truth
        self.quantiles = quantiles
        self.histogram = qp.utils.normalize_histogram(histogram, vb=vb)
        self.samples = samples
        self.gridded = qp.utils.normalize_integral(qp.utils.normalize_gridded(gridded))
        self.mix_mod = funcform
        self.mixmod = self.mix_mod
        self.limits = limits
        self.scheme = scheme
        self.interpolator = [None, None]
        # if vb: print('interpolator set to '+str(self.interpolator))

        # if vb and self.quantiles is None and self.histogram is None and self.gridded is None and self.samples is None:# and self.truth is None:
        #     print 'Warning: initializing a PDF object without inputs'
        #     return

        # Record how the PDF object was initialized:
        if self.mix_mod is not None:
            self.initialized = self.mix_mod
            self.first = 'mix_mod'
        elif self.gridded is not None:
        # if self.gridded is not None:
            self.initialized = self.gridded
            self.first = 'gridded'
            self.limits = (min(self.limits[0], np.min(self.gridded[0])), max(self.limits[-1], np.max(self.gridded[0])))
        elif self.samples is not None:
            self.initialized = self.samples
            self.first = 'samples'
            self.limits = (min(self.limits[0], np.min(self.samples)), max(self.limits[-1], np.max(self.samples)))
        elif self.histogram is not None:
            self.initialized = self.histogram
            self.first = 'histogram'
            self.limits = (min(self.limits[0], np.min(self.histogram[0])), max(self.limits[-1], np.max(self.histogram[0])))
        elif self.quantiles is not None:
            self.initialized = self.quantiles
            self.first = 'quantiles'
            self.limits = (min(self.limits[0], np.min(self.quantiles[-1])), max(self.limits[-1], np.max(self.quantiles[-1])))
        else:
            print('Warning: initializing a PDF object without inputs')
            return

        # The most recent parametrization used is, at this point, the
        # first one:
        self.truth = self.first
        self.last = self.first
        # self.klds = {}

        return

    def evaluate(self, loc, using=None, norm=False, vb=False):
        """
        Evaluates the PDF (either the true version or the first
        approximation of it if no parametrization is specified)
        at the given location(s).

        Parameters
        ----------
        loc: float or ndarray
            location(s) at which to evaluate the pdf
        using: string
            which parametrization to evaluate, defaults to initialization
        norm: boolean, optional
            True to normalize the evaluation, False if expected probability outside loc
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        (loc, val): tuple, float or ndarray
            the input locations and the value of the PDF (or its approximation) at the requested location(s)
        """
        if using is None:
            using = self.last

        # if using == 'truth':
        #     if self.truth is not None:
        #         if vb: print 'Evaluating the true distribution.'
        #         val = self.truth.pdf(loc)
        #         self.evaluated = (loc, val)
        #     else:
        #         raise ValueError('true PDF is not set, use an approximation instead (the most recent one was '+self.last+')')
        if using == 'mix_mod':
            if self.mix_mod is None:
                self.mix_mod = self.mix_mod_fit(vb=vb)
            if vb: print('Evaluating the fitted mixture model distribution.')
            val = self.mix_mod.pdf(loc)
            self.evaluated = (loc, val)
        else:
            # if vb: print 'Evaluating a `'+self.scheme+'` interpolation of the '+using+' parametrization.'
            evaluated = self.approximate(loc, using=using, vb=vb)
            val = evaluated[1]

        gridded = qp.utils.normalize_gridded((loc, val))
        if norm:
            gridded = qp.utils.normalize_integral(gridded, vb=vb)

        return gridded

    def integrate(self, limits=None, dx=0.001, using=None, vb=False):
        """
        Computes the integral under the PDF between the given limits.

        Parameters
        ----------
        limits: tuple, float, optional
            limits of integration
        dx: float, optional
            granularity of integral
        using: string, optional
            parametrization over which to approximate the integral
        vb: Boolean
            print progress to stdout?

        Returns
        -------
        integral: float
            value of the integral
        """
        if limits is None:
            limits = self.limits
            if vb:
                print('Integrating over '+str(limits)+' because no limits provided')
        lim_range = limits[-1] - limits[0]
        fine_grid = np.arange(limits[0], limits[-1] + dx, dx)

        evaluated = self.evaluate(fine_grid, vb=vb, using=using)
        integral = np.sum(evaluated[1]) * dx

        return integral

    def quantize(self, quants=None, N=9, limits=None, vb=False):
        """
        Computes an array of evenly-spaced quantiles from the truth.

        Parameters
        ----------
        quants: ndarray, float, optional
            array of quantile locations as decimals
        N: int, optional
            number of regular quantiles to compute
        limits: tuple, float, optional
            approximate values at which CDF=0. and CDF=1.
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        self.quantiles: ndarray, float
            the quantile points.

        Notes
        -----
        Quantiles of a PDF could be a useful approximate way to store it.
        This method computes the quantiles from a truth distribution
        (other representations forthcoming)
        and stores them in the `self.quantiles` attribute.
        Uses the `.ppf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the inverse CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html#scipy.stats.rv_continuous.ppf>`_ for details.
        TO DO: reorder these checks
        TO DO: address what happens when limits are too restrictive
        """
        if quants is not None:
            quantpoints = quants
            N = len(quantpoints)
        else:
            quantum = 1. / float(N+1)
            quantpoints = np.linspace(0.+quantum, 1.-quantum, N)

        if vb:
            print("Calculating "+str(len(quantpoints))+" quantiles: "+str(quantpoints))

        if limits is None:
            limits = self.limits

        if self.mixmod is not None:
            if isinstance(self.mixmod, qp.composite):
                if type(self.scheme) != int:
                    order = 5
                else:
                    order = self.scheme

                extrapoints = np.concatenate((np.array([0.]), quantpoints, np.array([1.])))
                min_delta = np.min(extrapoints[1:] - extrapoints[:-1])

                grid = np.linspace(limits[0], limits[-1], N + 1)
                icdf = self.mixmod.cdf(grid)
                unit_ext = 1. / (order + 1.)
                low_extended = 0
                while icdf[0] >= quantpoints[0]:
                    low_extended += 1
                    subgrid = np.linspace(limits[0] - 1., limits[0] - unit_ext, order)
                    subcdf = self.mixmod.cdf(subgrid)
                    grid = np.concatenate((subgrid, grid))
                    icdf = np.concatenate((subcdf, icdf))
                    limits = (limits[0] - 1., limits[-1])
                    if vb:
                        print('lower limits extended '+str(low_extended)+' times')
                high_extended = 0
                while icdf[-1] <= quantpoints[-1]:
                    high_extended += 1
                    subgrid = np.linspace(limits[-1] + unit_ext, limits[-1] + 1., order)
                    subcdf = self.mixmod.cdf(subgrid)
                    grid = np.concatenate((grid, subgrid))
                    icdf = np.concatenate((icdf, subcdf))
                    limits = (limits[0], limits[-1] + 1.)
                    if vb:
                        print('upper_limits extended '+str(high_extended)+' times')
                new_deltas = icdf[1:] - icdf[:-1]
                expanded = 0
                while np.max(new_deltas) >= min_delta:
                    expanded += 1
                    where_wrong = np.where(new_deltas >= min_delta)[0]
                    flipped = np.flip(where_wrong, axis=0)
                    for i in flipped:
                        delta_i = new_deltas[i] / (order + 1.)
                        subgrid = np.linspace(grid[i] + delta_i, grid[i+1] - delta_i, order)
                        grid = np.sort(np.insert(grid, i, subgrid))
                        subcdf = self.mixmod.cdf(subgrid)
                        icdf = np.sort(np.insert(icdf, i, subcdf))
                    new_deltas = icdf[1:] - icdf[:-1]
                if vb:
                    print('grid expanded '+str(expanded)+' times')
                # locs = np.array([bisect.bisect_right(icdf[:-1], quantpoints[n]) for n in range(N)])
                i = np.min(np.where(icdf > default_eps**(1./order)))
                f = np.max(np.where(1.-icdf > default_eps**(1./order)))
                icdf = icdf[i:f+1]
                grid = grid[i:f+1]

                # if vb: print('about to interpolate the CDF: '+str((icdf, grid)))
                # if vb: print('made the interpolator')
                #quantiles self.truth.ppf(quantpoints, ivals=grid[locs])

                # alternate = spi.interp1d(x, y, kind='linear', bounds_error=False, fill_value=default_eps)
                # backup = qp.utils.make_kludge_interpolator((x, y), outside=default_eps)

                quantiles = np.flip(quantpoints, axis=0)
                try:
                    while (order>0) and (not np.array_equal(quantiles, np.sort(quantiles))):
                        if vb: print('order is '+str(order))
                        b = spi.InterpolatedUnivariateSpline(icdf, grid, k=order, ext=1, check_finite=True)
                        quantiles = b(quantpoints)
                        order -= 1
                    assert(not np.any(np.isnan(quantiles)))
                except AssertionError:
                    print('ERROR: splines failed because '+str(AssertionError)+', defaulting to optimization for '+str((icdf, grid)))
                    locs = np.array([bisect.bisect_right(icdf[:-1], quantpoints[n]) for n in range(N)])
                    quantiles = self.mixmod.ppf(quantpoints, ivals=grid[locs])
                    assert(not np.any(np.isnan(quantiles)))
                except Exception as e:
                    print('ERROR in `scipy.interpolate.InterpolatedUnivariateSpline`')
                if vb: print('output quantiles = '+str(quantiles))
            else:
                quantiles = self.mixmod.ppf(quantpoints)
        else:
            print('New quantiles can only be computed from a truth distribution in this version.')
            return

        # integrals = self.truth.cdf(quantiles)
        # assert np.isclose(integrals, quantpoints)
        assert(type(quantiles) is np.ndarray)
        self.quantiles = (quantpoints, quantiles)
        if vb:
            print("Resulting "+str(len(quantiles))+" quantiles: "+str(self.quantiles))
        self.limits = (min(limits[0], np.min(quantiles)), max(limits[-1], np.max(quantiles)))
        self.last = 'quantiles'
        return self.quantiles

    def histogramize(self, binends=None, N=10, binrange=None, vb=False):
        """
        Computes integrated histogram bin values from the truth via the CDF.

        Parameters
        ----------
        binends: ndarray, float, optional
            array of N+1 endpoints of N bins
        N: int, optional
            number of regular bins if no binends provided
        binrange: tuple, float, optional
            pair of values of endpoints of total bin range
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        self.histogram: tuple of ndarrays of floats
            Pair of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins

        Comments
        --------
        A histogram representation of a PDF is a popular approximate way
        to store it. This method computes some histogram bin heights
        from a truth distribution (other representations forthcoming)
        and stores them in the `self.histogram` attribute.
        Uses the `.cdf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.cdf.html#scipy.stats.rv_continuous.cdf>`_ for details.
        """
        if binends is None:
            if binrange is None:
                binrange = self.limits
            binends = np.linspace(binrange[0], binrange[-1], N+1)
        else:
            N = len(binends) - 1

        histogram = np.zeros(N)
        if vb: print('Calculating histogram: ', binends)
        if self.mixmod is not None:
            cdf = self.mixmod.cdf(binends)
            heights = cdf[1:] - cdf[:-1]
            histogram = qp.utils.normalize_histogram((binends, heights), vb=vb)
            # for b in range(N):
            #     histogram[b] = (cdf[b+1]-cdf[b])/(binends[b+1]-binends[b])
        else:
            print('New histograms can only be computed from a mixmod format in this version.')
            return

        if vb: print('Result: ', histogram)
        self.histogram = histogram
        self.last = 'histogram'
        return self.histogram

    def mix_mod_fit(self, n_components=5, using=None, vb=False):
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        n_components: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation
        vb: boolean
            Report progress on stdout?

        Returns
        -------
        self.mix_mod: qp.composite object
            the qp.composite object approximating the PDF

        Notes
        -----
        Currently only supports mixture of Gaussians
        TO DO: change syntax n_components --> N
        """
        comp_range = list(range(n_components))
        if using == None:
            using = self.first

        if using == 'gridded':
            if self.gridded is not None:
                (x, y) = self.gridded
                ival_weights = np.ones(n_components) / n_components
                ival_means = min(x) + (max(x) - min(x)) * np.arange(n_components) / n_components
                ival_stdevs = np.sqrt((max(x) - min(x)) * np.ones(n_components) / n_components)
                ivals = np.array([ival_weights, ival_means, ival_stdevs]).T.flatten()
                def gmm(x, *args):
                    y = 0.
                    args = np.array(args).reshape((n_components, 3))
                    for c in comp_range:
                        # index = c * n_components
                        y += args[c][0] *  sps.norm(loc = args[c][1], scale = args[c][2]).pdf(x)
                    return y
                low_bounds = np.array([np.zeros(n_components), min(x) * np.ones(n_components), np.ones(n_components) * (max(x) - min(x)) / len(x)]).T.flatten()
                high_bounds = np.array([np.ones(n_components), max(x) * np.ones(n_components), np.ones(n_components) * (max(x) - min(x))]).T.flatten()
                popt, pcov = spo.curve_fit(gmm, self.gridded[0], self.gridded[1], ivals, bounds = (low_bounds, high_bounds))
                popt = popt.reshape((n_components, 3)).T
                weights = popt[0]
                means = popt[1]
                stdevs = popt[2]
            else:
                print('No gridded parametrization available.  Try using a different format.')
                return
        else:
            if self.samples is None:
                self.samples = self.sample(using=using, vb=vb)

            estimator = skl.mixture.GaussianMixture(n_components=n_components)
            estimator.fit(self.samples.reshape(-1, 1))

            weights = estimator.weights_
            means = estimator.means_[:, 0]
            stdevs = np.sqrt(estimator.covariances_[:, 0, 0])

        if vb:
            print('weights, means, stds = '+str((weights, means, stdevs)))

        components = []
        for i in comp_range:
            mix_mod_dict = {}
            function = sps.norm(loc = means[i], scale = stdevs[i])
            coefficient = weights[i]
            mix_mod_dict['function'] = function
            mix_mod_dict['coefficient'] = coefficient
            components.append(mix_mod_dict)

        if vb:
            statement = ''
            for c in comp_range:
                statement += str(weights[c])+r'$\cdot\mathcal{N}($'+str(means[c])+r','+str(stdevs[c])+r')\n'
            print(statement)
        self.mix_mod = qp.composite(components)
        return self.mix_mod

    def sample(self, N=1000, infty=default_infty, using=None, vb=False):
        """
        Samples the pdf in given representation

        Parameters
        ----------
        N: int, optional
            number of samples to produce
        infty: float, optional
            approximate value at which CDF=1.
        using: string, optional
            Parametrization on which to interpolate, defaults to initialization
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        samples: ndarray
            array of sampled values

        Notes
        -----
        TO DO: all formats should use rejection sampling
        TO DO: change infty to upper and lower bounds to use for quantiles
        TO DO: check for existence of parametrization before using it
        """
        if using is None:
            using = self.first

        if vb: print('Sampling from '+using+' parametrization.')

        # if using == 'truth':
        #     samples = self.truth.rvs(size=N)
        if using == 'mix_mod':
            samples = self.mix_mod.rvs(size=N)

        elif using == 'gridded':
            interpolator = self.interpolate(using = 'gridded', vb=vb)[0]
            # (xlims, ylims) = self.evaluate(self.limits, using='gridded', vb=vb)
            (xmin, xmax) = (min(self.gridded[0]), max(self.gridded[0]))
            (ymin, ymax) = (min(self.gridded[1]), max(self.gridded[1]))
            (xran, yran) = (xmax - xmin, ymax - ymin)
            samples = []
            while len(samples) < N:
                (x, y) = (xmin + xran * np.random.uniform(), ymin + yran * np.random.uniform())
                if y < interpolator(x):
                    samples.append(x)

        else:
            if using == 'quantiles':
                # First find the quantiles if none exist:
                if self.quantiles is None:
                    self.quantiles = self.quantize(vb=vb)

                # (x, y) = qp.utils.evaluate_quantiles(self.quantiles, vb=vb)
                (endpoints, weights) = qp.utils.normalize_quantiles(self.quantiles, vb=vb)
                # endpoints = np.insert(self.quantiles[1], [0, -1], self.limits)
                # weights = qp.utils.evaluate_quantiles(self.quantiles)[1]# self.evaluate((endpoints[1:]+endpoints[:-1])/2.)
                # interpolator = self.interpolate(using='quantiles', vb=False)

            if using == 'histogram':
                # First find the histogram if none exists:
                if self.histogram is None:
                    self.histogram = self.histogramize(vb=vb)

                endpoints = self.histogram[0]
                weights = self.histogram[1]

            ncats = len(weights)
            cats = list(range(ncats))
            sampbins = [0]*ncats
            for item in range(N):
                sampbins[qp.utils.choice(cats, weights)] += 1
            samples = []*N
            for c in cats:
                for n in range(sampbins[c]):
                    samples.append(np.random.uniform(low=endpoints[c], high=endpoints[c+1]))

        if vb: print('Sampled values: ', samples)
        self.samples = np.array(samples)
        self.limits = (min(self.limits[0], np.min(self.samples)), max(self.limits[-1], np.max(self.samples)))
        self.last = 'samples'
        return self.samples

    def interpolate(self, using=None, vb=False):
        """
        Constructs an `interpolator` function based on the parametrization.

        Parameters
        ----------
        using: string, optional
            parametrization on which to interpolate, defaults to initialization
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        interpolator
            an interpolator object

        Notes
        -----
        The `interpolator` object is a function that is used by the
        `approximate` method. It employs
        [`scipy.interpolate.interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
        to carry out the interpolation for the gridded format, using the internal
        `self.scheme` attribute to choose the interpolation scheme.  For quantile interpolation, it uses a `scipy.interpolate.InterpolatedUnivariateSpline` object, with self.scheme being the integer order of the spline.
        TO DO: store the interpolators separately with using tags
        """
        if using is None:
            using = self.last

        if using == 'truth' or using == 'mix_mod':
            interpolator = self.mix_mod.pdf
            print('A functional form needs no interpolation.  Try converting to an approximate parametrization first.')
            # return

        if using == 'quantiles':
            # First find the quantiles if none exist:
            if self.quantiles is None:
                self.quantiles = self.quantize(vb=vb)

            if type(self.scheme) != int:
                order = min(5, len(self.quantiles[0]))
            else:
                order = self.scheme

            if vb: print('input quantiles are '+str(self.quantiles[1]))
            # (x, y) = qp.utils.evaluate_quantiles(self.quantiles, vb=vb)
            # if vb: print('evaluated quantile PDF: '+str((x, y)))
            # [x_crit_lo, x_crit_hi] = [x[0], x[-1]]
            # [y_crit_lo, y_crit_hi] = [y[0], y[-1]]
            (x, y) = qp.utils.normalize_quantiles(self.quantiles, vb=vb)
            if vb: print('complete evaluated quantile PDF: '+str((x, y)))

            z = np.insert(self.quantiles[1], 0, min(x))
            z = np.append(z, max(x))
            q = np.insert(self.quantiles[0], 0, 0.)
            q = np.append(q, 1.)

            # knots, coeffs, degree = spi.splrep(z, q, k=order, s=0)
            #
            # def inside(xi):
            #     yi = spi.splev(xi, (knots, coeffs, degree), der=1)
            #     coeffs[yi<0]
            [x_crit_lo, x_crit_hi] = [self.quantiles[1][0], self.quantiles[1][-1]]
            [y_crit_lo, y_crit_hi] = [-1., -1.]

            try:
                while (order>0) and ((y_crit_lo <= 0.) or (y_crit_hi <= 0.)):
                    if vb: print('order is '+str(order))
                    inside = spi.InterpolatedUnivariateSpline(z, q, k=order, ext=1).derivative()
                    [y_crit_lo, y_crit_hi] = inside([x_crit_lo, x_crit_hi])
                    order -= 1
                assert((y_crit_lo > 0.) and (y_crit_hi > 0.))
            except AssertionError:
                print('ERROR: spline tangents '+str((y_crit_lo, y_crit_hi))+'<0')
                if type(self.scheme) == str:
                    scheme = self.scheme
                else:
                    scheme = 'linear'
                if vb: print('defaulting to '+scheme+' interpolation')
                inside_int = spi.interp1d(z, q, kind=scheme, bounds_error=False, fill_value=default_eps)
                derivative = (q[1:] - q[:-1]) / (z[1:] - z[:-1])
                derivative = np.insert(derivative, 0, default_eps)
                derivative = np.append(derivative, default_eps)
                def inside(xf):
                    nx = len(xf)
                    yf = np.ones(nx) * default_eps
                    for n in range(nx):
                        i = bisect.bisect_left(z, xf[n])
                        yf[n] = derivative[i]
                    return(yf)
                [y_crit_lo, y_crit_hi] = inside([x_crit_lo, x_crit_hi])
                assert((y_crit_lo > 0.) and (y_crit_hi > 0.))

            def quantile_interpolator(xf):
                yf = np.ones(np.shape(xf)) * default_eps
                in_inds = ((xf >= self.quantiles[1][0]) & (xf <= self.quantiles[1][-1])).nonzero()[0]
                lo_inds = ((xf < self.quantiles[1][0]) & (xf >= z[0])).nonzero()[0]
                hi_inds = ((xf > self.quantiles[1][-1]) & (xf <= z[-1])).nonzero()[0]
                if vb:
                    print('divided into '+str((lo_inds, in_inds, hi_inds)))

                try:
                    yf[in_inds] = inside(xf[in_inds])
                    assert(np.all(yf >= default_eps))
                    if vb:
                        print('Created a k=`'+str(order)+'`B-spline interpolator for the '+using+' parametrization.')
                except AssertionError:
                    print('ERROR: spline interpolation failed with '+str((xf[in_inds], yf[in_inds])))
                    try:
                        alternate = spi.interp1d(x, y, kind='linear', bounds_error=False, fill_value=default_eps)
                        yf[in_inds] = alternate(xf[in_inds])
                        assert(np.all(yf >= default_eps))
                        if vb:
                            print('Created a linear interpolator for the '+using+' parametrization.')
                    except AssertionError:
                        print('ERROR: linear interpolation failed for the '+using+' parametrization with '+str((xf[in_inds], yf[in_inds])))
                        backup = qp.utils.make_kludge_interpolator((x, y), threshold=default_eps)
                        yf[in_inds] = backup(xf[in_inds])
                        if vb:
                            print('Doing linear interpolation by hand for the '+using+' parametrization.')
                        assert(np.all(yf >= default_eps))
                if vb:
                    print('evaluated inside '+str((xf[in_inds], yf[in_inds])))

                try:
                    tan_lo = y_crit_lo / (x_crit_lo - z[0])
                    yf[lo_inds] = tan_lo * (xf[lo_inds] - z[0])# yf[in_inds[0]] / (xf[in_inds[0]] - z[0])
                    assert(np.all(yf >= default_eps))
                    if vb:
                        print('evaluated below '+str((xf[lo_inds], yf[lo_inds])))
                except AssertionError:
                    print('ERROR: linear extrapolation below failed with '+str((xf[lo_inds], yf[lo_inds]))+' via '+str((tan_lo, x_crit_lo, z[0])))

                try:
                    tan_hi = y_crit_hi / (z[-1] - x_crit_hi)
                    yf[hi_inds] = tan_hi * (z[-1] - xf[hi_inds])# yf[in_inds[-1]] * (xf[hi_inds] - z[-1]) / (xf[in_inds[-1]] - z[-1])
                    assert(np.all(yf >= default_eps))
                    if vb:
                        print('evaluated above '+str((xf[hi_inds], yf[hi_inds])))
                except AssertionError:
                    print('ERROR: linear extrapolation above failed with '+str((xf[hi_inds], yf[hi_inds]))+' via '+str((tan_hi, z[-1], x_crit_hi)))

                return(yf)
            # if vb:
            #     print(tck)

            #still not enforcing integration at ends
            # def quantile_interpolator(xf):
            #     yf = np.ones(len(xf)) * default_eps
            #     subset = ((xf>z[0]) == (xf<z[-1]))
            #     yf[subset] = b(xf[subset])
            #     return(yf)
            interpolator = quantile_interpolator

        if using == 'histogram':
            # First find the histogram if none exists:
            if self.histogram is None:
                self.histogram = self.histogramize(vb=vb)

            extra_y = np.insert(self.histogram[1], 0, default_eps)
            extra_y = np.append(extra_y, default_eps)

            def histogram_interpolator(xf):
                nx = len(xf)
                yf = np.ones(nx) * default_eps
                for n in range(nx):
                    i = bisect.bisect_left(self.histogram[0], xf[n])
                    yf[n] = extra_y[i]
                return(yf)

            #(x, y) = qp.utils.evaluate_histogram(self.histogram)

            interpolator = histogram_interpolator#qp.utils.evaluate_histogram()

            if vb:
                print('Created a piecewise constant interpolator for the '+using+' parametrization.')

        if using == 'samples':
            # First sample if not already done:
            if self.samples is None:
                self.samples = self.sample(vb=vb)

            # Well that's weird!  Samples does much better with KDE than linear interpolation.  I guess that shouldn't be surprising.
            def samples_interpolator(xf):
                kde = sps.gaussian_kde(self.samples)# , bw_method=bandwidth)
                yf = kde(xf)
                return (yf)
            interpolator = samples_interpolator
            if vb:
                print('Created a KDE interpolator for the '+using+' parametrization.')

            # (x, y) = qp.evaluate_samples(self.samples)
            # interpolator = spi.interp1d(x, y, kind=self.scheme, bounds_error=False, fill_value=default_eps)
            # if vb: print('interpolator support between '+str(min(x))+' and '+str(max(x))+' with extrapolation of '+str(default_eps))

        if using == 'gridded':
            if self.gridded is None:
                print('Interpolation from a gridded parametrization requires a previous gridded parametrization.')
                return
            (x, y) = self.gridded

            interpolator = spi.interp1d(x, y, kind=self.scheme, bounds_error=False, fill_value=default_eps)

            if vb:
                print('Created a `'+self.scheme+'` interpolator for the '+using+' parametrization.')

        self.interpolator = [interpolator, using]

        return self.interpolator

    def approximate(self, points, using=None, scheme=None, vb=False):
        """
        Interpolates the parametrization to get an approximation to the density.

        Parameters
        ----------
        points: ndarray
            the value(s) at which to evaluate the interpolated function
        using: string, optional
            approximation parametrization
        scheme: int or string, optional
            interpolation scheme, from the [`scipy.interpolate.interp1d`
            options](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html) or order of spline interpolation.
            If passed as `None`, the internal `self.scheme` attribute
            is used - this defaults to `linear` in the constructor.
            Otherwise, this attribute is reset to the one chosen.
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        points: ndarray, float
            the input grid upon which to interpolate
        interpolated: ndarray, float
            the interpolated points.

        Notes
        -----
        Extrapolation is via the `scheme` while values are positive;
        otherwise, extrapolation returns 0.

        Example:
            x, y = p.approximate(np.linspace(-1., 1., 100))
        """
        # First, reset the interpolation scheme if one is passed
        # explicitly:
        if scheme is not None:
            self.scheme = scheme

        # Now make the interpolation, using the current scheme:
        if self.interpolator[-1] == using:
            interpolator = self.interpolator[0]
        else:
            [interpolator, using] = self.interpolate(using=using, vb=vb)

        # if vb: print('interpolating over '+str(points)+' using '+using)
        # try:
        points.sort()
        interpolated = interpolator(points)
        # except:
        #     print('error in '+using+' interpolation of '+str(points))
        interpolated = qp.utils.normalize_gridded((points, interpolated))
        # interpolated[interpolated<0.] = 0.

        return interpolated#(points, interpolated)

    def plot(self, limits=None, loc='plot.pdf', vb=False):
        """
        Plots the PDF, in various ways.

        Parameters
        ----------
        limits: tuple, float, optional
            limits over which plot should be made
        loc: string, optional
            filepath/name for saving the plot, optional
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----
        What this method plots depends on what information about the PDF
        is stored in it: the more properties the PDF has,
        the more exciting the plot!
        """
        if limits is None:
            limits = self.limits
        extrema = limits

        colors = {}
        colors['truth'] = 'k'
        colors['mix_mod'] = 'k'
        colors['gridded'] = 'k'
        colors['quantiles'] = 'blueviolet'
        colors['histogram'] = 'darkorange'
        colors['samples'] = 'forestgreen'

        styles = {}
        styles['truth'] = '-'
        styles['mix_mod'] = ':'
        styles['gridded'] = '--'
        styles['quantiles'] = '--'#(0,(5,10))
        styles['histogram'] = ':'#(0,(3,6))
        styles['samples'] = '-.'#(0,(1,2))

        x = np.linspace(self.limits[0], self.limits[-1], 100)
        if self.mixmod is not None:
            [min_x, max_x] = [self.mixmod.ppf(np.array([0.001])), self.mixmod.ppf(np.array([0.999]))]
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            [min_x, max_x] = extrema
            x = np.linspace(min_x, max_x, 100)
            y = self.mixmod.pdf(x)
            plt.plot(x, y, color=colors['truth'], linestyle=styles['truth'], lw=5.0, alpha=0.25, label='True PDF')
            if vb:
                print('Plotted truth.')

        if self.mix_mod is not None:
            [min_x, max_x] = [self.mix_mod.ppf(np.array([0.001])), self.mix_mod.ppf(np.array([0.999]))]
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            [min_x, max_x] = extrema
            x = np.linspace(min_x, max_x, 100)
            y = self.mix_mod.pdf(x)
            plt.plot(x, y, color=colors['mix_mod'], linestyle=styles['mix_mod'], lw=2.0, alpha=1.0, label='Mixture Model PDF')
            if vb:
                print('Plotted mixture model.')

        if self.quantiles is not None:
            # (z, p) = self.evaluate(self.quantiles[1], using='quantiles', vb=vb)
            # print('first: '+str((z,p)))
            (x, y) = qp.utils.normalize_quantiles(self.quantiles)
            if vb:
                print('second: '+str((x, y)))
            [min_x, max_x] = [min(x), max(x)]
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            [min_x, max_x] = extrema
            x = np.linspace(min_x, max_x, 100)
            if vb:
                print('third: '+str(x))
            (grid, qinterpolated) = self.approximate(x, vb=vb, using='quantiles')
            plt.scatter(self.quantiles[1], np.zeros(np.shape(self.quantiles[1])), color=colors['quantiles'], marker='|', s=100, label='Quantiles', alpha=0.75)
            # plt.vlines(z, np.zeros(len(self.quantiles[1])), p, color=colors['quantiles'], linestyle=styles['quantiles'], lw=1.0, alpha=1.0, label='Quantiles')
            plt.plot(grid, qinterpolated, color=colors['quantiles'], lw=2.0, alpha=1.0, linestyle=styles['quantiles'], label='Quantile Interpolated PDF')
            if vb:
                print('Plotted quantiles.')

        if self.histogram is not None:
            [min_x, max_x] = [min(self.histogram[0]), max(self.histogram[0])]
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            [min_x, max_x] = extrema
            x = np.linspace(min_x, max_x, 100)
            # plt.vlines(self.histogram[0], self.histogram[0][:-1],
            #            self.histogram[0][1:], color=colors['histogram'], linestyle=styles['histogram'], lw=1.0, alpha=1., label='histogram bin ends')
            plt.scatter(self.histogram[0], np.zeros(np.shape(self.histogram[0])), color=colors['histogram'], marker='|', s=100, label='Histogram Bin Ends', alpha=0.75)
            (grid, hinterpolated) = self.approximate(x, vb=vb,
                                                     using='histogram')
            plt.plot(grid, hinterpolated, color=colors['histogram'], lw=2.0, alpha=1.0,
                     linestyle=styles['histogram'],
                     label='Histogram Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print('Plotted histogram.')

        if self.gridded is not None:
            [min_x, max_x] = [min(self.gridded[0]), max(self.gridded[0])]
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            [min_x, max_x] = extrema
            (x, y) = self.gridded
            plt.plot(x, y, color=colors['gridded'], lw=1.0, alpha=0.5,
                     linestyle=styles['gridded'], label='Gridded PDF')
            if vb:
                print('Plotted gridded.')

        if self.samples is not None:
            [min_x, max_x] = [min(self.samples), max(self.samples)]
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            [min_x, max_x] = extrema
            x = np.linspace(min_x, max_x, 100)
            plt.scatter(self.samples, np.zeros(np.shape(self.samples)), color=colors['samples'], marker='|', s=100, label='Samples', alpha=0.75)
            (grid, sinterpolated) = self.approximate(x, vb=vb,
                                                     using='samples')
            plt.plot(grid, sinterpolated, color=colors['samples'], lw=2.0,
                        alpha=1.0, linestyle=styles['samples'],
                        label='Samples Interpolated PDF')
            if vb:
                print('Plotted samples')

        plt.xlim(extrema[0], extrema[-1])
        plt.legend(fontsize='large')
        plt.xlabel(r'$z$', fontsize=16)
        plt.ylabel(r'$p(z)$', fontsize=16)
        plt.tight_layout()
        plt.savefig(loc, dpi=250)

        return

    # def kld(self, using=None, limits=None, dx=0.01):
    #     """
    #     Calculates Kullback-Leibler divergence of quantile approximation from truth.
    #
    #     Parameters
    #     ----------
    #     using: string
    #         parametrization to use
    #     limits: tuple of floats, optional
    #         endpoints of integration interval in which to calculate KLD
    #     dx: float
    #         resolution of integration grid
    #
    #     Returns
    #     -------
    #     KL: float
    #         value of Kullback-Leibler divergence from approximation to truth
    #         if truth is available; otherwise nothing.
    #
    #     Notes
    #     -----
    #     Example::
    #         d = p.kld(limits=(-1., 1.), dx=1./100))
    #     """
    #     print('This function is deprecated; use `qp.utils.calculate_kld`.')
    #     return
    #     # print('This function is deprecated; use `qp.utils.calculate_kl_divergence`.')
    #     # return
    #     # if self.truth is None:
    #     #     print('Truth not available for comparison.')
    #     #     return
    #     # else:
    #     #     if using is None:
    #     #         using = self.last
    #     #     if limits is None:
    #     #         limits = self.limits
    #     #     D = int((limits[-1] - limits[0]) / dx)
    #     #     grid = np.linspace(limits[0], limits[1], D)
    #     #     KL = qp.utils.quick_kl_divergence(self.evaluate(grid, using='truth'), self.evaluate(grid, using=using), dx=dx)
    #     #     self.klds[using] = KL
    #     #     return(KL)
    #
    # def rmse(self, limits=(0., 1.), dx=0.01):
    #     """
    #     Calculates root mean square difference between quantile approximation
    #     and truth.
    #
    #     Parameters
    #     ----------
    #     limits: tuple of floats
    #         endpoints of integration interval in which to calculate KLD
    #     dx: float
    #         resolution of integration grid
    #
    #     Returns
    #     -------
    #     RMS: float
    #         value of root mean square difference between approximation of truth
    #         if truth is available; otherwise nothing.
    #
    #     Notes
    #     -----
    #     Example::
    #         d = p.rms(limits=(-1., 1.), dx=1./100))
    #     """
    #     print('This function is deprecated; use `qp.utils.calculate_rmse`.')
    #     return
    #     # if self.truth is None:
    #     #     print('Truth not available for comparison.')
    #     #     return
    #     # else:
    #     #     RMS = qp.utils.calculate_rms(self, self, limits=limits, dx=dx)
    #     #     return(RMS)
