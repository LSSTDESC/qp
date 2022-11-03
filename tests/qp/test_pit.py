import unittest
import numpy as np
import qp
from qp import interp_gen
from qp.ensemble import Ensemble
from qp.metrics.pit import PIT


# constants for tests
NMAX = 2.5
NPDF = 399
ZGRID = np.linspace(0, NMAX, 301)

ADVAL_ALL = 82.51480
ADVAL_CUT = 1.10750
CVMVAL = 20.63155
KSVAL = 0.367384
OUTRATE = 0.0

# These constants retained for future use
# CDEVAL = -4.31200
# SIGIQR = 0.0045947
# BIAS = -0.00001576
# SIGMAD = 0.0046489

class PitTestCase(unittest.TestCase):
    """ Test cases for PIT metric. """

    def setUp(self):
        np.random.seed(87)
        self.true_zs = np.random.uniform(high=NMAX, size=NPDF)

        locs = np.expand_dims(self.true_zs + np.random.normal(0.0, 0.01, NPDF), -1)
        scales = np.ones((NPDF, 1)) * 0.1 + np.random.uniform(size=(NPDF, 1)) * .05
        self.n_ens = Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))

        self.grid_ens = self.n_ens.convert_to(interp_gen, xvals=ZGRID)

    def test_pit_metrics(self):
        """Base test of PIT metric generation"""
        quant_grid = np.linspace(0, 1, 101)
        pit_obj = PIT(self.grid_ens, self.true_zs, quant_grid)

        pit_samples = pit_obj.pit_samps
        self.assertTrue(len(pit_samples) == 399)

        pit_ensemble = pit_obj.pit
        self.assertTrue(pit_ensemble.npdf == 1)

        meta_metrics = pit_obj.calculate_pit_meta_metrics()

        ad_stat = meta_metrics['ad'].statistic
        assert np.isclose(ad_stat, ADVAL_ALL)

        cut_ad_stat = pit_obj.evaluate_PIT_anderson_ksamp(pit_min=0.6, pit_max=0.9).statistic
        assert np.isclose(cut_ad_stat, ADVAL_CUT)

        cvm_stat = meta_metrics['cvm'].statistic
        assert np.isclose(cvm_stat, CVMVAL)

        ks_stat = meta_metrics['ks'].statistic
        assert np.isclose(ks_stat, KSVAL)

        assert np.isclose(meta_metrics['outlier_rate'], OUTRATE)

    def test_pit_metric_small_eval_grid(self):
        """Test PIT metric warning message when number of pit samples is smaller than the evaluation grid"""
        with self.assertLogs(level='WARNING') as log:
            quant_grid = np.linspace(0, 1, 1000)
            _ = PIT(self.grid_ens, self.true_zs, quant_grid)
            self.assertIn('Number of pit samples is smaller', log.output[0])
