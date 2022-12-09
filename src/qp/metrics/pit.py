import logging
import numpy as np
from scipy import stats
import qp
from qp.metrics.metrics import calculate_outlier_rate
from qp.metrics.array_metrics import quick_anderson_ksamp, quick_cramer_von_mises, quick_kolmogorov_smirnov

DEFAULT_QUANTS = np.linspace(0, 1, 100)

class PIT():
    """PIT(qp_ens, true_vals, eval_grid=DEFAULT_QUANTS)
    Probability Integral Transform

    Parameters
    ----------
    qp_ens : Ensemble
        A collection of N distribution objects
    true_vals : [float]
        An array-like sequence of N float values representing the known true value for each distribution
    eval_grid : [float], optional
        A strictly increasing array-like sequence in the range [0,1], by default DEFAULT_QUANTS

    Returns
    -------
    PIT object
        An object with an Ensemble containing the PIT distribution, and a full set of PIT samples.
    """

    def __init__(self, qp_ens, true_vals, eval_grid=DEFAULT_QUANTS):
        """We will create a quantile Ensemble to store the PIT distribution, but also store the
        full set of PIT samples as ancillary data of the (single PDF) ensemble.

        Parameters
        ----------
        qp_ens : Ensemble
            A collection of N distribution objects
        true_vals : [float]
            An array-like sequence of N float values representing the known true value for each distribution
        eval_grid : [float], optional
            A strictly increasing array-like sequence in the range [0,1], by default DEFAULT_QUANTS
        """

        self._true_vals = true_vals

        # For each distribution in the Ensemble, calculate the CDF where x = known_true_value
        self._pit_samps = np.array([qp_ens[i].cdf(self._true_vals[i])[0][0] for i in range(len(self._true_vals))])

        n_pit = np.min([len(self._pit_samps), len(eval_grid)])
        if n_pit < len(eval_grid):
            logging.warning('Number of pit samples is smaller than the evaluation grid size. Will create a new evaluation grid with size = number of pit samples')
            eval_grid = np.linspace(0, 1, n_pit)

        data_quants = np.quantile(self._pit_samps, eval_grid)
        self._pit = qp.Ensemble(qp.quant_piecewise, data=dict(quants=eval_grid, locs=np.atleast_2d(data_quants)))

    @property
    def pit_samps(self):
        """Returns the PIT samples. i.e. ``CDF(true_vals)`` for each distribution in the Ensemble used to initialize the PIT object.

        Returns
        -------
        np.array
            An array of floats
        """
        return self._pit_samps

    @property
    def pit(self):
        """Return the PIT Ensemble object

        Returns
        -------
        qp.Ensemble
            An Ensemble containing 1 qp.quant_piecewise distribution.
        """
        return self._pit

    def calculate_pit_meta_metrics(self):
        """Convenience method that will calculate all of the PIT meta metrics and return them 
            as a dictionary.

        Returns
        -------
        dictionary
            The collection of PIT statistics
        """
        pit_meta_metrics = {}

        pit_meta_metrics['ad'] = self.evaluate_PIT_anderson_ksamp()
        pit_meta_metrics['cvm'] = self.evaluate_PIT_CvM()
        pit_meta_metrics['ks'] = self.evaluate_PIT_KS()
        pit_meta_metrics['outlier_rate'] = self.evaluate_PIT_outlier_rate()

        return pit_meta_metrics

    def evaluate_PIT_anderson_ksamp(self, pit_min=0., pit_max=1.):
        """Use scipy.stats.anderson_ksamp to compute the Anderson-Darling statistic
        for the cdf(truth) values by comparing with a uniform distribution between 0 and 1.
        Up to the current version (1.9.3), scipy.stats.anderson does not support
        uniform distributions as reference for 1-sample test, therefore we create a uniform
        "distribution" and pass it as the second value in the list of parameters to the scipy
        implementation of k-sample Anderson-Darling.
        For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html

        Parameters
        ----------
        pit_min : float, optional
            Minimum PIT value to accept, by default 0.
        pit_max : float, optional
            Maximum PIT value to accept, by default 1.

        Returns
        -------
        [Result objects]
            A array of objects with attributes `statistic`, `critical_values`, and `significance_level`.
        For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html
        """
        # Removed the CDF values that are outside the min/max range
        trimmed_pit_values = self._trim_pit_values(pit_min, pit_max)

        uniform_yvals = np.linspace(pit_min, pit_max, len(trimmed_pit_values))

        return quick_anderson_ksamp(trimmed_pit_values, uniform_yvals)

    def evaluate_PIT_CvM(self):
        """Calculate the Cramer von Mises statistic using scipy.stats.cramervonmises using self._pit_samps
        compared to a uniform distribution. For more details see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html

        Returns
        -------
        [Result objects]
            A array of objects with attributes `statistic` and `pvalue`
            For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html
        """
        return quick_cramer_von_mises(self._pit_samps, stats.uniform.cdf)

    def evaluate_PIT_KS(self):
        """Calculate the Kolmogorov-Smirnov statistic using scipy.stats.kstest. For more details see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html

        Returns
        -------
        [Return Object]
            A array of objects with attributes `statistic` and `pvalue`.
            For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
        """
        return quick_kolmogorov_smirnov(self._pit_samps, stats.uniform.cdf)

    def evaluate_PIT_outlier_rate(self, pit_min=0.0001, pit_max=0.9999):
        """Compute fraction of PIT outliers by evaluating the CDF of the distribution in the PIT Ensemble
        at `pit_min` and `pit_max`.

        Parameters
        ----------
        pit_min : float, optional
            Lower bound for outliers, by default 0.0001
        pit_max : float, optional
            Upper bound for outliers, by default 0.9999

        Returns
        -------
        float
            The percentage of outliers in this distribution given the min and max bounds.
        """
        return calculate_outlier_rate(self._pit, pit_min, pit_max)[0]

    def _trim_pit_values(self, cdf_min, cdf_max):
        """Remove and report any cdf(x) that are outside the min/max range.

        Parameters
        ----------
        cdf_min : float
            The minimum cdf(x) value to accept
        cdf_max : float
            The maximum cdf(x) value to accept

        Returns
        -------
        [float]
            The list of PIT values within the min/max range.
        """
        # Create truth mask for pit values between cdf_min and pit max
        mask = (self._pit_samps >= cdf_min) & (self._pit_samps <= cdf_max)

        # Keep pit values that are within the min/max range
        pits_clean = self._pit_samps[mask]

        # Determine how many pit values were dropped and warn the user.
        diff = len(self._pit_samps) - len(pits_clean)
        if diff > 0:
            logging.warning("Removed %d PITs from the sample.", diff)

        return pits_clean
