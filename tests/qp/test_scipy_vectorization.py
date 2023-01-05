"""
Unit tests for PDF class
"""

import unittest
import qp
import qp.metrics
import numpy as np

from qp import test_funcs

import time

class ScipyVectorizationTests(unittest.TestCase):
    """ Tests for the metrics """

    @classmethod
    def setUpClass(cls):
        """
        Make any objects that are used in multiple tests.
        """
        cls.random_state = 9999
        cls.rvs_size = 100
        t1 = time.perf_counter()
        # ! Consider changing this to the gamma distribution - norm is symmetric. trunc_norm is a good option too.
        cls.ens_n = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm'])  #pylint: disable=no-member
        t2 = time.perf_counter()
        print(f'Setup time is {t2-t1}')
        print(f'Ensemble contains {cls.ens_n.npdf} distributions.')
        print(f'For each statistic {cls.rvs_size} random variables will be generated.')

    @classmethod
    def tearDownClass(cls):
        """ Clean up any mock data files created by the tests. """

    def test_ad_results(self):
        """This test compares the results of the Anderson-Darling evaluation against a known output"""
        expected_output = [1.41296516, 2.18126088, 0.59390624, 0.43942853, 0.67658563, 0.38376997,
        0.66460796, 2.72899571, 1.4542696, 1.34378282, 1.19221443]

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        test_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n,
            fit_metric='ad',
            num_samples=rvs_size,
            _random_state=random_state
        )

        for test, expected in zip(test_output, expected_output):
            assert np.isclose(test, expected)

    def test_call_copied_method_ad_versus_single_distribution_ensemble(self):
        """For Anderson-Darling stat stat for ensemble vs. ensemble-with-1-distribution"""
        expected_output = [1.41296516e+00, 5.94997248e+01, 1.76866646e+02, 3.76671747e+02,
        6.17435009e+02, 9.61477755e+02, 1.24161826e+03, 2.33034254e+03, 1.87698332e+03,
        2.44635392e+03, 4.10601407e+03]

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        test_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n[0],
            fit_metric='ad',
            num_samples=rvs_size,
            _random_state=random_state
        )

        for test, expected in zip(test_output, expected_output):
            assert np.isclose(test, expected)

    def test_cvm_results(self):
        """This test compares the results of the Cramer-von Mises evaluation against a known output"""
        expected_output = [0.22630139, 0.27090351, 0.09884569, 0.04506071, 0.10731862, 0.05631584,
            0.07646541, 0.52543176, 0.23656815, 0.21094075, 0.22384716]

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        test_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n,
            fit_metric='cvm',
            num_samples=rvs_size,
            _random_state=random_state
        )

        for test, expected in zip(test_output, expected_output):
            assert np.isclose(test, expected)

    def test_call_copied_method_cvm_versus_single_distribution_ensemble(self):
        """For Cramer von Mises stat stat for ensemble vs. ensemble-with-1-distribution"""
        expected_output = [0.22630139, 4.90875135, 12.18815523, 17.88155095, 21.76168586, 24.17512056,
        22.69008766, 28.436041, 23.2801267, 23.87779491, 28.99530345]

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        test_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n[0],
            fit_metric='cvm',
            num_samples=rvs_size,
            _random_state=random_state
        )

        for test, expected in zip(test_output, expected_output):
            assert np.isclose(test, expected)

    def test_ks_results(self):
        """This test compares the results of the Kolmogorov-Smirnov evaluation against a known output"""
        expected_output = [0.11739344, 0.10533858, 0.07950318, 0.06216017, 0.08471774, 0.05786293,
        0.08342168, 0.14441362, 0.09896913, 0.11243258, 0.09189444]

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        test_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n,
            fit_metric='ks',
            num_samples=rvs_size,
            _random_state=random_state
        )

        for test, expected in zip(test_output, expected_output):
            assert np.isclose(test, expected)

    def test_call_copied_method_ks_versus_single_distribution_ensemble(self):
        """Test Kolmogorov-Smirnov stat for ensemble vs. ensemble-with-1-distribution"""
        expected_output = [0.11739344, 0.38809819, 0.51939786, 0.70188153, 0.76502763, 0.80196938,
        0.79053856, 0.8996195, 0.80379571, 0.82881527, 0.9095734]

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        test_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n[0],
            fit_metric='ks',
            num_samples=rvs_size,
            _random_state=random_state
        )

        for test, expected in zip(test_output, expected_output):
            assert np.isclose(test, expected)
