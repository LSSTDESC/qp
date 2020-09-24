"""
Unit tests for PDF class
"""
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

class PDFTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass

    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass

    def test_wide_separation_quantiles(self):
        """
        When the two modes of a composite PDF are widely separated,
        the quantiles can be misestimated. That's OK (ish) but the
        KLD should not be NaN.
        """
        # Create a pathological PDF:
        component_1 = {}
        component_1['function'] = sps.norm(loc=0.4, scale=0.001)
        component_1['coefficient'] = 0.1
        component_2 = {}
        component_2['function'] = sps.norm(loc=3.5, scale=0.001)
        component_2['coefficient'] = 0.9
        dist_info = [component_1, component_2]
        dist = qp.composite(dist_info)
        test_limits = (0., 5.)
        P = qp.PDF(funcform=dist, limits=test_limits)
        # Quantile approximate:
        Q = qp.PDF(quantiles=P.quantize(N=10, limits=test_limits), limits=test_limits)
        # Compute KLD:
        KLD = qp.metrics.calculate_kld(P, Q, limits=test_limits)
        self.assertFalse(np.isnan(KLD))


if __name__ == '__main__':
    unittest.main()
