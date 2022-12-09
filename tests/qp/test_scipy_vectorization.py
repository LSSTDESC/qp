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

    @unittest.skip("The two implementations of Anderson-Darling will converge for very large values of self.rvs_size. See docstring for more info.")
    def test_copied_ad(self):
        """For Anderson Darling stat, test the parity between Scipy 1.9 code and
        Scipy 1.10.0dev code that was copied over into qp. Note that due to slightly different
        implementations there will not be exact agreement of results for a finite number
        of random variates.
        However, as self.rvs_size is increased, the results begin to converge.
        Because there isn't parity for a finite number of random variates, this test is
        disabled for automation."""
        print('Anderson Darling comparison')
        t1 = time.perf_counter()

        gof_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n,
            fit_metric='ad',
            num_samples=self.rvs_size,
            _random_state=self.random_state
        )

        t2 = time.perf_counter()
        gof2_run_time = t2 - t1
        print(f"GOF 2 run time: {gof2_run_time}")
        print(gof_output)

        t1 = time.perf_counter()

        adr_output = qp.metrics.calculate_anderson_darling(self.ens_n, 'norm', _random_state=self.random_state)
        t2 = time.perf_counter()

        ad_run_time = t2 - t1
        print(f"Scipy 1.9 AD run time: {ad_run_time}")
        print([a.statistic for a in adr_output])

        print(f"Speed increase: {ad_run_time/gof2_run_time} x")

        for gof, adr in zip(gof_output, adr_output):
            self.assertEqual(gof, adr.statistic)

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

    def test_copied_cvm(self):
        """For Cramer von Mises stat, test the parity between Scipy 1.9 code and
        Scipy 1.10.0dev code that was copied over into qp"""

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        t1 = time.perf_counter()
        gof_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n,
            fit_metric='cvm',
            num_samples=rvs_size,
            _random_state=random_state
        )
        t2 = time.perf_counter()
        gof_run_time = t2 - t1
        print(f"GOF run time: {gof_run_time}")

        t1 = time.perf_counter()
        cvm_output = qp.metrics.calculate_cramer_von_mises(self.ens_n, self.ens_n, num_samples=rvs_size, _random_state=random_state)
        t2 = time.perf_counter()
        cvm_run_time = t2 - t1
        print(f"Scipy 1.9 AD run time: {cvm_run_time}")
        print(f"Speed increase: {cvm_run_time/gof_run_time} x")

        for gof, cvm in zip(gof_output, cvm_output):
            self.assertEqual(gof, cvm.statistic)

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

    @unittest.skip("The Scipy 1.9 implementation of Dplus and Dminus do not handle nested arrays of cdf values. See docstring for more info.")
    def test_copied_ks(self):
        """For Kolmogorov-Smirnov stat, test the parity between Scipy 1.9 code and
        Scipy 1.10.0dev code that was copied over into qp. The implementation of _compute_dplus
        and _compute_dminus do not handle nested arrays of cdf values - such as those returned
        by Ensemble.cdf(x).

        See the definitions here: https://github.com/scipy/scipy/blob/main/scipy/stats/_stats_py.py#L7565
        Note that `n = len(cdfvals)` will return the length of the outer most array.
        Thus, if cdfvals = ndarray([[1,2,3,4]]), n will equal 1. The vectorized
        implementation in Scipy 1.10 handles this nesting correctly.

        This test should not be run with automation, but can be used to check the correctness
        of the vectorized Scipy 1.10 code **if** a temporary patch is applied to the
        Ensemble.cdf method as follows:
        -    return self._frozen.cdf(x)
        +    return np.squeeze(self._frozen.cdf(x))

        This is very hacky, but this comparison test should be considered temporary
        until Scipy 1.10 is released. Afterward, we can remove this test and related code.
        """

        # Specify these here, so that changes in setupClass don't affect the results
        rvs_size = 100
        random_state = 9999

        t1 = time.perf_counter()
        gof_output = qp.metrics.calculate_goodness_of_fit(
            self.ens_n,
            self.ens_n,
            fit_metric='ks',
            num_samples=rvs_size,
            _random_state=random_state
        )
        t2 = time.perf_counter()
        gof_run_time = t2 - t1
        print(f"GOF run time: {gof_run_time}")

        t1 = time.perf_counter()
        ks_output = qp.metrics.calculate_kolmogorov_smirnov(self.ens_n, self.ens_n, num_samples=rvs_size, _random_state=random_state)
        t2 = time.perf_counter()
        ks_run_time = t2 - t1
        print(f"Scipy 1.9 AD run time: {ks_run_time}")
        print(f"Speed increase: {ks_run_time/gof_run_time} x")

        for gof, ks in zip(gof_output, ks_output):
            self.assertEqual(gof, ks.statistic)

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
