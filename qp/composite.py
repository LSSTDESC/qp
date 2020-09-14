import numpy as np
import scipy as sp
from scipy import stats as sps
import scipy.optimize as op

import qp

class composite(object):

    def __init__(self, components, vb=False):
        """
        A probability distribution that is a linear combination of scipy.stats.rv_continuous objects

        Parameters
        ----------
        components: list or tuple, dicts
            aggregation of dicts defining component functions and their coefficients
        vb: boolean
            report on progress to stdout?

        Notes
        -----
        TO DO: change x --> z
        """
        self.components = components
        self.n_components = len(self.components)
        self.component_range = list(range(self.n_components))

        coefficients = np.array([component['coefficient'] for component in self.components])
        self.coefficients = coefficients / np.sum(coefficients)
        self.functions = np.array([component['function'] for component in self.components])

    def pdf(self, xs):
        """
        Evaluates the composite PDF at locations

        Parameters
        ----------
        xs: float or numpy.ndarray, float
            value(s) at which to evaluate the PDF

        Returns
        -------
        ps: float or numpy.ndarray, float
            value(s) of the PDF at xs
        """
        p = np.zeros(np.shape(xs))
        for c in self.component_range:
            p += self.coefficients[c] * self.functions[c].pdf(xs)
        return p

    def cdf(self, xs):
        """
        Evaluates the composite CDF at locations

        Parameters
        ----------
        xs: float or numpy.ndarray, float
            value(s) at which to evaluate the CDF

        Returns
        -------
        ps: float or numpy.ndarray, float
            value(s) of the CDF at xs
        """
        ps = np.zeros(np.shape(xs))
        for c in self.component_range:
            ps += self.coefficients[c] * self.functions[c].cdf(xs)
        return ps

    def rvs(self, size):
        """
        Samples the composite probability distribution

        Parameters
        ----------
        size: int
            number of samples to take

        Returns
        -------
        xs: numpy.ndarray, float
            samples from the PDF
        """
        groups = np.random.choice(self.component_range, size, p=self.coefficients)
        u, counts = np.unique(groups, return_counts=True)
        samples = np.empty(0)
        for i in range(len(u)):
            samples = np.append(samples, self.functions[u[i]].rvs(counts[i]))
        return np.array(samples).flatten()

    def ppf(self, cdfs, ivals=None):
        """
        Evaluates the composite PPF at locations

        Parameters
        ----------
        cdfs: float or numpy.ndarray, float
            value(s) at which to find quantiles
        ivals: float or numpy.ndarray, float
            initial guesses for quantiles

        Returns
        -------
        xs: float or numpy.ndarray, float
            quantiles
        """
        N = np.shape(cdfs)[0]
        xs = np.zeros(N)

        if ivals is not None:
            xs0 = ivals
        else:
            all_cdfs = np.zeros(N)
            for c in self.component_range:
                all_cdfs += self.functions[c].ppf(cdfs)
            xs0 = all_cdfs / self.n_components

        for n in range(N):
            def ppf_helper(x):
                return np.absolute(cdfs[n] - self.cdf(x))
            res = op.minimize(ppf_helper, xs0[n], method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5}, tol=1e-8)
                    # res = op.basinhopping(ppf_helper, xs0[n])#, method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5})
            xs[n] += res.x
            # if vb:
            #     print(res.message, res.success)

        return xs
