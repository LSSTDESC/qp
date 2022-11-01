import logging
import numpy as np
from scipy import stats
import qp
from qp.metrics.metrics import calculate_outlier_rate, calculate_anderson_ksamp


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
        """_summary_

        Args:
            pit_min (float, optional): _description_. Defaults to 0..
            pit_max (float, optional): _description_. Defaults to 1..

        Returns:
            _type_: _description_
        """
        return calculate_anderson_ksamp(self._pit_samps, pit_min, pit_max)

    def evaluate_PIT_CvM(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return stats.cramervonmises(self._pit_samps, 'uniform')

    def evaluate_PIT_KS(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return stats.kstest(self._pit_samps, 'uniform')

    def evaluate_PIT_outlier_rate(self, pit_min=0.0001, pit_max=0.9999):
        """Compute fraction of PIT outliers

        Args:
            pit_min (float, optional): _description_. Defaults to 0.0001.
            pit_max (float, optional): _description_. Defaults to 0.9999.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        return calculate_outlier_rate(self._pit, pit_min, pit_max)
