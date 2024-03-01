# pylint: disable=no-member
# pylint: disable=protected-access

"""
Unit tests for PDF class
"""
import unittest

import numpy as np

import qp
import qp.metrics
from qp import test_funcs
from qp.metrics.array_metrics import quick_rbpe
from qp.metrics.concrete_metric_classes import (BrierMetric, KLDMetric,
                                                MomentMetric, OutlierMetric,
                                                RBPEMetric, RMSEMetric)
from qp.metrics.metrics import (
    _calculate_grid_parameters, _check_ensemble_is_not_nested,
    _check_ensembles_are_same_size,
    _check_ensembles_contain_correct_number_of_distributions, calculate_brier,
    calculate_goodness_of_fit, calculate_kld, calculate_moment,
    calculate_outlier_rate, calculate_rbpe, calculate_rmse)
from qp.utils import epsilon


class MetricTestCase(unittest.TestCase):
    """Tests for the metrics"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = test_funcs.build_ensemble(qp.stats.norm_gen.test_data["norm"])
        self.ens_n_shift = test_funcs.build_ensemble(
            qp.stats.norm_gen.test_data["norm_shifted"]
        )
        self.ens_n_multi = test_funcs.build_ensemble(
            qp.stats.norm_gen.test_data["norm_multi_d"]
        )

        locs = 2 * (np.random.uniform(size=(10, 1)) - 0.5)
        scales = 1 + 0.2 * (np.random.uniform(size=(10, 1)) - 0.5)
        self.ens_n_plus_one = qp.Ensemble(
            qp.stats.norm, data=dict(loc=locs, scale=scales)
        )

        bins = np.linspace(-5, 5, 11)
        self.ens_s = self.ens_n.convert_to(qp.spline_gen, xvals=bins, method="xy")

    def test_calculate_grid_parameters(self):
        """Given a small, simple input, ensure that the grid parameters are correct."""
        limits = (0, 1)
        dx = 1.0 / 11
        grid_params = _calculate_grid_parameters(limits, dx)
        assert grid_params.cardinality == 11
        assert grid_params.resolution == 0.1
        assert grid_params.grid_values[0] == limits[0]
        assert grid_params.grid_values[-1] == limits[-1]
        assert grid_params.grid_values.size == grid_params.cardinality
        assert grid_params.hist_bin_edges[0] == limits[0] - grid_params.resolution / 2
        assert grid_params.hist_bin_edges[-1] == limits[-1] + grid_params.resolution / 2
        assert grid_params.hist_bin_edges.size == grid_params.cardinality + 1

    def test_calculate_grid_parameters_larger_range(self):
        """Test that a large range in limits and small delta returns expected results"""
        limits = (-75, 112)
        dx = 0.042
        grid_params = _calculate_grid_parameters(limits, dx)
        assert grid_params.grid_values[0] == limits[0]
        assert grid_params.grid_values[-1] == limits[-1]
        assert grid_params.grid_values.size == grid_params.cardinality
        assert grid_params.hist_bin_edges[0] == limits[0] - grid_params.resolution / 2
        assert grid_params.hist_bin_edges[-1] == limits[-1] + grid_params.resolution / 2
        assert grid_params.hist_bin_edges.size == grid_params.cardinality + 1

    def test_kld(self):
        """Test the calculate_kld method. Ensure that the class wrapped version
        returns the same results."""
        limits = (0.0, 2.5)
        kld = calculate_kld(self.ens_n, self.ens_n_shift, limits=limits)
        assert np.all(kld == 0.0)

        kld_class = KLDMetric(limits=limits)
        kld_class.initialize()
        result = kld_class.evaluate(self.ens_n, self.ens_n_shift)
        kld_class.finalize()
        assert np.all(kld == result)

    def test_calculate_moment(self):
        """Base case test. Ensure that the class wrapped
        version returns the same results."""
        moment = 1
        limits = (-2.0, 2.0)
        result = calculate_moment(self.ens_n, moment, limits)

        self.assertTrue(result is not None)

        moment_class = MomentMetric(moment_order=moment, limits=limits)
        moment_class.initialize()
        class_result = moment_class.evaluate(self.ens_n)
        moment_class.finalize()
        assert np.all(result == class_result)

    def test_kld_alternative_ensembles(self):
        """Test the calculate_kld method against different types of ensembles"""
        bins = np.linspace(-5, 5, 11)
        quants = np.linspace(0.01, 0.99, 7)

        ens_h = self.ens_n[0].convert_to(qp.hist_gen, bins=bins)
        ens_h_shift = self.ens_n_shift[0].convert_to(qp.hist_gen, bins=bins)

        ens_i = self.ens_n[0].convert_to(qp.interp_gen, xvals=bins)
        ens_i_shift = self.ens_n_shift[0].convert_to(qp.interp_gen, xvals=bins)

        ens_s = self.ens_n[0].convert_to(qp.spline_gen, xvals=bins, method="xy")
        ens_s_shift = self.ens_n_shift[0].convert_to(
            qp.spline_gen, xvals=bins, method="xy"
        )

        ens_q = self.ens_n[0].convert_to(qp.quant_gen, quants=quants)
        ens_q_shift = self.ens_n_shift[0].convert_to(qp.quant_gen, quants=quants)

        kld_histogram = calculate_kld(ens_h, ens_h_shift, limits=(0.0, 2.5))
        assert np.all(kld_histogram == 0.0)

        kld_interp = calculate_kld(ens_i, ens_i_shift, limits=(0.0, 2.5))
        assert np.all(kld_interp == 0.0)

        kld_spline = calculate_kld(ens_s, ens_s_shift, limits=(0.0, 2.5))
        assert np.all(kld_spline == 0.0)

        kld_quants = calculate_kld(ens_q, ens_q_shift, limits=(0.0, 2.5))
        assert np.all(kld_quants == 0.0)

    def test_kld_different_shapes(self):
        """Ensure that the kld function fails when trying to compare ensembles of different sizes."""
        with self.assertRaises(ValueError) as context:
            calculate_kld(self.ens_n, self.ens_n_plus_one, limits=(0.0, 2.5))

        self.assertTrue(
            "Cannot calculate KLD between two ensembles with different shapes"
            in str(context.exception)
        )

    def test_broken_kld(self):
        """Test to see if the calculate_kld function responds appropriately to negative results"""
        kld = calculate_kld(self.ens_n, self.ens_s, limits=(0.0, 2.5))

        assert np.all(kld == epsilon)

    def test_rmse(self):
        """Test the calculate_rmse method. Ensure that the class wrapped version
        returns the same results."""
        limits = (0.0, 2.5)
        rmse = calculate_rmse(self.ens_n, self.ens_n_shift, limits=limits)
        assert np.all(rmse == 0.0)

        rmse_class = RMSEMetric(limits=limits)
        rmse_class.initialize()
        result = rmse_class.evaluate(self.ens_n, self.ens_n_shift)
        rmse_class.finalize()
        assert np.all(rmse == result)

    def test_rmse_alternative_ensembles(self):
        """Test the calculate_rmse method against different types of ensembles"""
        bins = np.linspace(-5, 5, 11)
        quants = np.linspace(0.01, 0.99, 7)

        ens_h = self.ens_n[0].convert_to(qp.hist_gen, bins=bins)
        ens_h_shift = self.ens_n_shift[0].convert_to(qp.hist_gen, bins=bins)

        ens_i = self.ens_n[0].convert_to(qp.interp_gen, xvals=bins)
        ens_i_shift = self.ens_n_shift[0].convert_to(qp.interp_gen, xvals=bins)

        ens_s = self.ens_n[0].convert_to(qp.spline_gen, xvals=bins, method="xy")
        ens_s_shift = self.ens_n_shift[0].convert_to(
            qp.spline_gen, xvals=bins, method="xy"
        )

        ens_q = self.ens_n[0].convert_to(qp.quant_gen, quants=quants)
        ens_q_shift = self.ens_n_shift[0].convert_to(qp.quant_gen, quants=quants)

        rmse_histogram = calculate_rmse(ens_h, ens_h_shift, limits=(0.0, 2.5))
        assert np.all(rmse_histogram == 0.0)

        rmse_interp = calculate_rmse(ens_i, ens_i_shift, limits=(0.0, 2.5))
        assert np.all(rmse_interp == 0.0)

        rmse_spline = calculate_rmse(ens_s, ens_s_shift, limits=(0.0, 2.5))
        assert np.all(rmse_spline == 0.0)

        rmse_quants = calculate_rmse(ens_q, ens_q_shift, limits=(0.0, 2.5))
        assert np.all(rmse_quants == 0.0)

    def test_rmse_different_shapes(self):
        """Ensure that the rmse function fails when trying to compare ensembles of different sizes."""
        with self.assertRaises(ValueError) as context:
            calculate_rmse(self.ens_n, self.ens_n_plus_one, limits=(0.0, 2.5))

        self.assertTrue(
            "Cannot calculate RMSE between two ensembles with different shapes"
            in str(context.exception)
        )

    def test_rbpe(self):
        """Test the risk_based_point_estimate method. Ensure that the class wrapped
        version returns the same results."""
        limits = (0.0, 2.5)
        rbpe = calculate_rbpe(self.ens_n, limits=limits)
        assert np.all(rbpe >= 0.0)
        assert np.all(rbpe <= 2.5)

        rbpe_class = RBPEMetric(limits=limits)
        rbpe_class.initialize()
        result = rbpe_class.evaluate(self.ens_n)
        rbpe_class.finalize()
        assert np.all(rbpe == result)

    def test_rbpe_no_limits(self):
        """Test the risk_based_point_estimate method when the user doesn't provide a set of limits"""
        rbpe = calculate_rbpe(self.ens_n)
        assert np.all(rbpe >= -2.0)

    def test_rbpe_alternative_ensembles(self):
        """Test the risk_based_point_estimate method against different types of ensembles"""
        bins = np.linspace(-5, 5, 11)
        quants = np.linspace(0.01, 0.99, 7)

        ens_h = self.ens_n[0].convert_to(qp.hist_gen, bins=bins)
        ens_i = self.ens_n[0].convert_to(qp.interp_gen, xvals=bins)
        ens_s = self.ens_n[0].convert_to(qp.spline_gen, xvals=bins, method="xy")
        ens_q = self.ens_n[0].convert_to(qp.quant_gen, quants=quants)
        ens_m = self.ens_n[0].convert_to(qp.mixmod_gen, samples=1000, ncomps=3)

        rbpe_histogram = calculate_rbpe(ens_h, limits=(0.0, 2.5))
        assert np.all(rbpe_histogram >= 0.0)
        assert np.all(rbpe_histogram <= 2.5)

        rbpe_interp = calculate_rbpe(ens_i, limits=(0.0, 2.5))
        assert np.all(rbpe_interp >= 0.0)
        assert np.all(rbpe_interp <= 2.5)

        rbpe_spline = calculate_rbpe(ens_s, limits=(0.0, 2.5))
        assert np.all(rbpe_spline >= 0.0)
        assert np.all(rbpe_spline <= 2.5)

        rbpe_quants = calculate_rbpe(ens_q, limits=(0.0, 2.5))
        assert np.all(rbpe_quants >= 0.0)
        assert np.all(rbpe_quants <= 2.5)

        rbpe_mixmod = calculate_rbpe(ens_m, limits=(0.0, 2.5))
        assert np.all(rbpe_mixmod >= 0.0)
        assert np.all(rbpe_mixmod <= 2.5)

    def test_quick_rbpe(self):
        """Test the quick_rbpe method"""

        def eval_pdf_at_z(z):
            return self.ens_n[0].pdf(z)[0][0]

        integration_bounds = (
            self.ens_n[0].ppf(0.01)[0][0],
            self.ens_n[0].ppf(0.99)[0][0],
        )
        rbpe = quick_rbpe(eval_pdf_at_z, integration_bounds, limits=(0.0, 2.5))
        assert np.all(rbpe >= 0.0)
        assert np.all(rbpe <= 2.5)

    def test_quick_rbpe_no_limits(self):
        """Test the quick_rbpe method"""

        def eval_pdf_at_z(z):
            return self.ens_n[0].pdf(z)[0][0]

        integration_bounds = (
            self.ens_n[0].ppf(0.01)[0][0],
            self.ens_n[0].ppf(0.99)[0][0],
        )
        rbpe = quick_rbpe(eval_pdf_at_z, integration_bounds)
        assert np.all(rbpe >= -2.0)

    def test_rbpe_multiple_pdfs(self):
        """Ensure that calculate_rbpe function fails when working with multi-dimensional Ensembles."""
        with self.assertRaises(ValueError) as context:
            _ = calculate_rbpe(self.ens_n_multi, limits=(0.0, 2.5))

        error_msg = "quick_rbpe only handles Ensembles with a single PDF"
        self.assertTrue(error_msg in str(context.exception))

    def test_calculate_brier(self):
        """Base test case of Ensemble-based brier metric. Ensure that the class wrapped
        version returns the same results."""
        truth = 2 * (np.random.uniform(size=(11, 1)) - 0.5)
        limits = [-2.0, 2]
        result = calculate_brier(self.ens_n, truth, limits)
        self.assertTrue(result is not None)

        brier_class = BrierMetric(limits=limits)
        brier_class.initialize()
        class_result = brier_class.evaluate(self.ens_n, truth)
        assert np.all(result == class_result)

        brier_class_accumulate = BrierMetric(limits=limits)
        brier_class_accumulate.initialize()
        sum_tuple = brier_class_accumulate.accumulate(self.ens_n, truth)
        accumulated_result = brier_class_accumulate.finalize([sum_tuple])
        assert np.all(result == accumulated_result)

    def test_calculate_brier_mismatched_number_of_truths(self):
        """Expect an exception when number of truth values doesn't match number of distributions"""
        truth = 2 * (np.random.uniform(size=(10, 1)) - 0.5)
        limits = [-2.0, 2]
        with self.assertRaises(ValueError) as context:
            _ = calculate_brier(self.ens_n, truth, limits)

        error_msg = "Number of distributions in the Ensemble"
        self.assertTrue(error_msg in str(context.exception))

    def test_calculate_brier_truth_outside_of_limits(self):
        """Expect an exception when truth is outside of the limits"""
        truth = 2 * (np.random.uniform(size=(10, 1)) - 0.5)
        truth = np.append(truth, 100)
        limits = [-2.0, 2]
        with self.assertRaises(ValueError) as context:
            _ = calculate_brier(self.ens_n, truth, limits)

        error_msg = "Input truth values exceed the defined limits"
        self.assertTrue(error_msg in str(context.exception))

    def test_calculate_outlier_rate(self):
        """Base case test. Ensure that the class wrapped
        version returns the same results."""
        output = calculate_outlier_rate(self.ens_n[0])
        self.assertTrue(len(output) == 1)
        self.assertTrue(np.isclose(output[0], 0.012436869911068668))

        outlier_class = OutlierMetric()
        outlier_class.initialize()
        class_results = outlier_class.evaluate(self.ens_n[0])
        outlier_class.finalize()
        assert np.all(output == class_results)

    def test_calculate_outlier_rate_with_bounds(self):
        """Include min/max bounds for outlier rate"""
        output = calculate_outlier_rate(self.ens_n[0], -10, 10)
        self.assertTrue(len(output) == 1)
        self.assertTrue(np.isclose(output[0], 0))

    def test_calculate_outlier_rate_many_distributions(self):
        """Check that the outlier rate is correctly calculated for an Ensemble with many distributions"""
        output = calculate_outlier_rate(self.ens_n)
        self.assertTrue(len(output) == 11)

    def test_check_ensembles_are_same_size(self):
        """Test that no Value Error is raised when the ensembles are the same size"""
        try:
            _check_ensembles_are_same_size(self.ens_n, self.ens_n_shift)
        except ValueError:
            self.fail("Unexpectedly raised ValueError")

    def test_check_ensembles_are_same_size_asserts(self):
        """Test that a Value Error is raised when the ensembles are not the same size"""
        with self.assertRaises(ValueError) as context:
            _check_ensembles_are_same_size(self.ens_n, self.ens_n_plus_one)

        error_msg = "Input ensembles should have the same number of distributions"
        self.assertTrue(error_msg in str(context.exception))

    def test_check_ensemble_is_not_nested_with_flat_ensemble(self):
        """Test that no ValueError is raised when a flat Ensemble is passed in"""
        try:
            _check_ensemble_is_not_nested(self.ens_n)
        except ValueError:
            self.fail("Unexpectedly raised ValueError")

    def test_check_ensemble_is_not_nested_with_nested_ensemble(self):
        """Test that a ValueError is raised when a nested Ensemble is passed in"""
        with self.assertRaises(ValueError) as context:
            _check_ensemble_is_not_nested(self.ens_n_multi)

        error_msg = (
            "Each element in the input Ensemble should be a single distribution."
        )
        self.assertTrue(error_msg in str(context.exception))

    def test_check_ensembles_contain_correct_number_of_distributions_base(self):
        """Test base case of two ensembles the same size"""
        try:
            _check_ensembles_contain_correct_number_of_distributions(
                estimate=self.ens_n, reference=self.ens_n
            )
        except ValueError:
            self.fail("Unexpectedly raised ValueError")

    def test_check_ensembles_contain_correct_number_of_distributions_both_size_1(self):
        """Test base case of two ensembles with 1 distribution each"""
        try:
            _check_ensembles_contain_correct_number_of_distributions(
                estimate=self.ens_n[0], reference=self.ens_n[1]
            )
        except ValueError:
            self.fail("Unexpected raised ValueError")

    def test_check_ensembles_contain_correct_number_of_distributions_reference_has_1(
        self,
    ):
        """Test case of reference ensemble has 1 distribution, estimate has N"""
        try:
            _check_ensembles_contain_correct_number_of_distributions(
                estimate=self.ens_n, reference=self.ens_n[1]
            )
        except ValueError:
            self.fail("Unexpected raised ValueError")

    def test_check_ensembles_contain_correct_number_of_distributions_with_different_sizes(
        self,
    ):
        """Test the case of reference with M distributions, and estimate with N distributions"""
        with self.assertRaises(ValueError) as context:
            _check_ensembles_contain_correct_number_of_distributions(
                estimate=self.ens_n, reference=self.ens_n_plus_one
            )

        error_msg = "`reference` should contain either"
        self.assertTrue(error_msg in str(context.exception))

    def test_calculate_goodness_of_fit_with_unsupported_fit_metric(self):
        """Test that a KeyError is raised if an unsupported fit_metric is requested"""
        with self.assertRaises(KeyError) as context:
            calculate_goodness_of_fit(self.ens_n, self.ens_n, "xx")

        error_msg = "`fit_metric` should be one of"
        self.assertTrue(error_msg in str(context.exception))


if __name__ == "__main__":
    unittest.main()
