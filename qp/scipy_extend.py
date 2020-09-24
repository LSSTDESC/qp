"""This module implements continous distributions that inherit from the
`scipy.stats.rv_continuous` class

If you would like to add a sub-class, please read the instructions on subclassing
here:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html


Open questions:
1) At this time the normalization is not enforced for many of the PDF types.  It is assumed that
the user values give correct normalization.  We should think about this more.

2) At this time for most of the distributions, only the _pdf function is overridden.  This is all that
is required to inherit from `scipy.stats.rv_continuous`; however, providing implementations of some of
_logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf could speed the code up a lot is some cases.

"""


import numpy as np

from scipy.stats import rv_continuous, gaussian_kde
from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import minimize


class interp_dist(rv_continuous):
    """Interpolator based distribution

    Notes
    -----
    This implements a PDF using a set of interpolated values.

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the PDF.
    """

    name = 'interp_dist'

    def __init__(self, xvals, yvals, *args, **kwargs):
        """Class Constructor

        Parameters
        ----------
        xvals : `np.array`
            x-values used to define the PDF interpolation grid
        xvals : `np.array`
            y-values used to define the PDF interpolation grid
        *args, **kwds : passed to the base class constructor
        """
        self._xvals = xvals
        self._yvals = yvals
        self._interp_kind = kwargs.get('kind', 'linear')
        self._interp1d = interp1d(xvals, yvals, kind=self._interp_kind, bounds_error=False, fill_value=0.)
        super(interp_dist).__init__(*args, **kwargs)

    @property
    def xvals(self):
        """Return the x-values used to define the PDF interpolation grid"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-values used to define the PDF interpolation grid"""
        return self._yvals

    def _pdf(self, x, *args):
        return self._interp1d(x)

    def _updated_ctor_param(self):
        """
        Set the xvals and yvals as additional constructor arguments
        """
        dct = super(interp_dist)._updated_ctor_param()
        dct['xvals'] = self._xvals
        dct['yvals'] = self._yvals
        return dct


class spline_dist(rv_continuous):
    """UnivariateSpline based distribution

    Notes
    -----
    This implements a PDF using a spline.

    It simply takes a set of x and y values and uses `scipy.interpolate.UnivariateSpline` to
    build the PDF.
    """

    name = 'spline_dist'

    def __init__(self, xvals, yvals, *args, **kwargs):
        """Class Constructor

        Parameters
        ----------
        xvals : `np.array`
            x-values used to define the PDF interpolation grid
        xvals : `np.array`
            y-values used to define the PDF interpolation grid
        *args, **kwds : passed to the base class constructor
        """
        self._xvals = xvals
        self._yvals = yvals
        self._spline = UnivariateSpline(xvals, yvals, ext=1, s=0.01)
        super(spline_dist).__init__(*args, **kwargs)

    @property
    def xvals(self):
        """Return the x-values used to define the PDF interpolation grid"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-values used to define the PDF interpolation grid"""
        return self._yvals

    def _pdf(self, x, *args):
        return self._spline(x)

    def _updated_ctor_param(self):
        """
        Set the xvals and yvals as additional constructor arguments
        """
        dct = super(spline_dist)._updated_ctor_param()
        dct['xvals'] = self._xvals
        dct['yvals'] = self._yvals
        return dct


class intspline_dist(rv_continuous):
    """InterpolatedUnivariateSpline based distribution

    Notes
    -----
    This implements a PDF using a spline.

    It simply takes a set of x and y values and uses `scipy.interpolate.InterpolatedUnivariateSpline` to
    build the PDF.

    This is a subclass of `scipy.interpolate.UnivariateSpline` where the spline knot-placement is chosen to force
    the curve to go through the points provide on construction.

    """

    name = 'intspline_dist'

    def __init__(self, xvals, yvals, *args, **kwargs):
        """Class Constructor

        Parameters
        ----------
        xvals : `np.array`
            x-values used to define the PDF interpolation grid
        xvals : `np.array`
            y-values used to define the PDF interpolation grid
        *args, **kwds : passed to the base class constructor
        """
        self._xvals = xvals
        self._yvals = yvals
        self._spline = InterpolatedUnivariateSpline(xvals, yvals, ext=1)
        super(intspline_dist).__init__(*args, **kwargs)

    @property
    def xvals(self):
        """Return the x-values used to define the PDF interpolation grid"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-values used to define the PDF interpolation grid"""
        return self._yvals

    def _pdf(self, x, *args):
        return self._spline(x)

    def _updated_ctor_param(self):
        """
        Set the xvals and yvals as additional constructor arguments
        """
        dct = super(intspline_dist)._updated_ctor_param()
        dct['xvals'] = self._xvals
        dct['yvals'] = self._yvals
        return dct


class kde_dist(rv_continuous):
    """Kernal density estimation based distribution

    Notes
    -----
    This implements a PDF using a kernal density estimation obtained from a set of samples.

    This simply takes the samples constructs a `scipy.stats.gaussian_kde` to serve as the PDF
    """

    name = 'kde_dist'

    def __init__(self, samples, *args, **kwargs):
        """Class Constructor

        Parameters
        ----------
        samples : `np.array`
            Values sampled from the PDF.
        *args, **kwds : passed to the base class constructor
        """
        self._samples = samples
        self._kde = gaussian_kde(self._samples)
        super(kde_dist).__init__(*args, **kwargs)

    @property
    def samples(self):
        """Return the samples used to build the distribution"""
        return self._samples

    def _pdf(self, x, *args):
        return self._kde(x)

    def _updated_ctor_param(self):
        """
        Set the xvals and yvals as additional constructor arguments
        """
        dct = super(kde_dist)._updated_ctor_param()
        dct['samples'] = self._samples
        return dct


class quantile_dist(rv_continuous):
    """Quantile based distribution

    Notes
    -----
    This implements a PDF using an interpolated spline that passes through a set of quantiles provided

    It simply takes quantiles values and the parameter values at which they are obtained and uses `scipy.interpolate.InterpolatedUnivariateSpline` to
    build the CDF.

    It also builds the PDF as the derivate of the CDF.
    """

    name = 'quantile_dist'

    def __init__(self, quantiles, par_values, *args, **kwargs):
        """Class Constructor

        Parameters
        ----------
        quantiles : `np.array`
            Quantile values used to define the CDF
        par_values : `np.array`
            Parameter values at which those quantiles are reached
        *args, **kwds : passed to the base class constructor
        """
        self._quantiles = quantiles
        self._par_values = par_values
        self._cdf_spline = InterpolatedUnivariateSpline(self._par_values, self._quantiles, ext=1)
        self._pdf_spline = self._cdf_spline.derivative()
        super(quantile_dist).__init__(*args, **kwargs)

    @property
    def quantiles(self):
        """Return the quantile values used to define the CDF"""
        return self._quantiles

    @property
    def par_values(self):
        """Return the parameter values at which those quantiles are reached"""
        return self._par_values

    def _cdf(self, x, *args):
        return self._cdf_spline(x)

    def _pdf(self, x, *args):
        return self._pdf_spline(x)

    def _updated_ctor_param(self):
        """
        Set the quantiles as additional constructor arguments
        """
        dct = super(quantile_dist)._updated_ctor_param()
        dct['par_values'] = self._par_values
        dct['quantiles'] = self._quantiles
        return dct


class sum_dist(rv_continuous):
    """Distribution that is a sum of distributions"""

    name = 'sum_dist'

    def __init__(self, coefs, dists, *args, **kwargs):
        """
        A probability distribution that is a linear combination of scipy.stats.rv_continuous objects

        Parameters
        ----------
        coefs: list or tuple of floats
            Component coefficients
        dists: list or tuple of `scipy.stats.rv_continuous` objects
            Component distributions
        """
        assert len(coefs) == len(dists)
        self._coefs = np.array(coefs)
        self._dists = dists
        # Does this need to be normalized somehow?
        # self._coefs /= self._coefs.sum()
        super(sum_dist).__init__(*args, **kwargs)

    @property
    def coefs(self):
        """Return the component coefficients"""
        return self.coefs

    @property
    def dists(self):
        """Return the component distributions"""
        return self._dists

    def _pdf(self, x, *args):
        """
        Evaluates the composite PDF at locations

        Parameters
        ----------
        x: float or numpy.ndarray, float
            value(s) at which to evaluate the PDF

        Returns
        -------
        p: float or numpy.ndarray, float
            value(s) of the PDF at x
        """
        p = np.zeros(np.shape(x))
        for coef, dist in zip(self._coefs, self._dists):
            p += coef * dist.pdf(x)
        return p

    def _cdf(self, x, *args):
        """
        Evaluates the composite CDF at locations

        Parameters
        ----------
        x: float or numpy.ndarray, float
            value(s) at which to evaluate the CDF

        Returns
        -------
        p: float or numpy.ndarray, float
            value(s) of the CDF at x
        """
        p = np.zeros(np.shape(x))
        for coef, dist in zip(self._coefs, self._dists):
            p += coef * dist.cdf(x, *args)
        return p


    def _rvs(self, *args):
        """
        Samples the composite probability distribution

        Parameters
        ----------
        size: int
            number of samples to take

        Returns
        -------
        x: numpy.ndarray, float
            samples from the PDF
        """
        groups = np.random.choice(np.arange(len(self._coefs)), self._size, p=self._coefs)
        u, counts = np.unique(groups, return_counts=True)
        samples = np.empty(0)
        for idx, c_counts in zip(u, counts):
            samples = np.append(samples, self._dists[idx].rvs(*args, size=c_counts))
        return np.array(samples).flatten()


    def _ppf(self, q, *args):
        """
        Evaluates the composite PPF at locations

        Parameters
        ----------
        cdf: float or numpy.ndarray, float
            value(s) at which to find quantiles
        ival: float or numpy.ndarray, float
            initial guesses for quantiles

        Returns
        -------
        x: float or numpy.ndarray, float
            quantiles
        """
        N = np.shape(q)[0]
        x = np.zeros(N)

        if args:
            x0s = args[0]
        else:
            all_cdfs = np.zeros(N)
            for dist in self._dists:
                all_cdfs += dist.ppf(q)
            x0s = all_cdfs / len(self._dists)

        for i, (x0, qq) in enumerate(zip(x0s, q)):
            def ppf_helper(x):
                return np.absolute(qq - self.cdf(x))
            res = minimize(ppf_helper, x0, method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5}, tol=1e-8)
                    # res = op.basinhopping(ppf_helper, xs0[n])#, method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5})
            x[i] += res.x
            # if vb:
            #     print(res.message, res.success)

        return x


    def _updated_ctor_param(self):
        """
        Set the xvals and yvals as additional constructor arguments
        """
        dct = super(sum_dist)._updated_ctor_param()
        dct['coefs'] = self._coefs
        dct['dists'] = self._dists
        return dct
