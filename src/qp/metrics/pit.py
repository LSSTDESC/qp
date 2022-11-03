import logging
import numpy as np
from scipy import stats
import qp
from qp.metrics.metrics import calculate_outlier_rate
from qp.metrics.array_metrics import quick_anderson_ksamp

DEFAULT_QUANTS = np.linspace(0, 1, 100)

class PIT():
    """ Probability Integral Transform """

    def __init__(self, qp_ens, ztrue, eval_grid=DEFAULT_QUANTS):
        """We will create a quantile Ensemble to store the PIT distribution, but also store the
        full set of PIT samples as ancillary data of the (single PDF) ensemble.

        Args:
            qp_ens Ensemble: A collection of N distribution objects
            ztrue [float]: An array-like sequence of N float values representing the known true value for each distribution
            eval_grid ([float], optional): A strictly increasing array-like sequence in the range [0,1]. Defaults to DEFAULT_QUANTS.
        """

        self._ztrue = ztrue

        # For each distribution in the Ensemble, calculate the CDF where x = known_true_value
        self._pit_samps = np.array([qp_ens[i].cdf(self._ztrue[i])[0][0] for i in range(len(self._ztrue))])

        n_pit = np.min([len(self._pit_samps), len(eval_grid)])
        if n_pit < len(eval_grid):
            logging.warning('Number of pit samples is smaller than the evaluation grid size. Will create a new evaluation grid with size = number of pit samples')
            eval_grid = np.linspace(0, 1, n_pit)

        data_quants = np.quantile(self._pit_samps, eval_grid)
        self._pit = qp.Ensemble(qp.quant_piecewise, data=dict(quants=eval_grid, locs=np.atleast_2d(data_quants)))

    @property
    def pit_samps(self):
        """Return the samples used to compute the PIT"""
        return self._pit_samps

    @property
    def pit(self):
        """Return the PIT Ensemble object"""
        return self._pit

    def calculate_pit_meta_metrics(self):
        """Convenience method that will calculate all of the PIT meta metrics and return them 
            as a dictionary.

        Returns:
            dict: The collection of PIT statistics
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

        Args:
            pit_min (float, optional): Minimum PIT value to accept. Defaults to 0..
            pit_max (float, optional): Maximum PIT value to accept. Defaults to 1..

        Returns:
        output [Objects]: A array of objects with attributes `statistic`, `critical_values`, and `significance_level`.
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

        Returns:
            output [Objects]: A array of objects with attributes `statistic` and `pvalue`
            For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html
        """
        return stats.cramervonmises(self._pit_samps, 'uniform')

    def evaluate_PIT_KS(self):
        """Calculate the Kolmogorov-Smirnov statistic using scipy.stats.kstest for each pair of distributions
            in two input Ensembles. For more details see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html

        Args:
            p np.array of floats: A gridded array representing the CDF of a given distribution
            q np.array of floats: A second gridded array representing the CDF of a given distribution

        Returns:
            output [KstestResult]: A array of named 2-tuples.
            For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
        """
        return stats.kstest(self._pit_samps, 'uniform')

    def evaluate_PIT_outlier_rate(self, pit_min=0.0001, pit_max=0.9999):
        """Compute fraction of PIT outliers by evaluating the CDF of the distribution in the PIT Ensemble
            at `pit_min` and `pit_max`. 

        Args:
            pit_min (float, optional): Lower bound for outliers. Defaults to 0.0001.
            pit_max (float, optional): Upper bound for outliers. Defaults to 0.9999.


        Returns:
            float: The percentage of outliers in this distribution given the min and max bounds.
        """
        return calculate_outlier_rate(self._pit, pit_min, pit_max)

    def _trim_pit_values(self, cdf_min, cdf_max):
        """Remove and report any cdf(x) that are outside the min/max range.

        Args:
            cdf_min float: The minimum cdf(x) value to accept
            cdf_max float: The maximum cdf(x) value to accept

        Returns:
            pits_clean [float]: The list of PIT values within the min/max range.
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
