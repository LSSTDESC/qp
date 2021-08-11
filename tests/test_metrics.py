"""
Unit tests for PDF class
"""

import unittest
import qp

from qp import test_funcs


class MetricTestCase(unittest.TestCase):
    """ Tests for the metrics """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm'])  #pylint: disable=no-member
        self.ens_n_shift = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm_shifted'])  #pylint: disable=no-member

    def tearDown(self):
        """ Clean up any mock data files created by the tests. """

    def test_kld(self):
        """ Test the calculate_kld method """
        _ = qp.metrics.calculate_kld(self.ens_n, self.ens_n_shift, limits=(0.,2.5))

    def test_rmse(self):
        """ Test the calculate_rmse method """
        _ = qp.metrics.calculate_rmse(self.ens_n, self.ens_n_shift, limits=(0.,2.5))


if __name__ == '__main__':
    unittest.main()
