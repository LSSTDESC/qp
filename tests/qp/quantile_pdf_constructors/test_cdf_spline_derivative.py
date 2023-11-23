import unittest

import numpy as np

import qp
from qp.quantile_pdf_constructors import (AbstractQuantilePdfConstructor,
                                          CdfSplineDerivative)


class CdfSplineDerivativeTestCase(unittest.TestCase):
    """Tests for the CDF Spline Derivative PDF constructor for quantile parameterization."""

    def setUp(self):
        self.single_norm = qp.stats.norm(loc=3, scale=0.5)  # pylint: disable=no-member
        self.many_norm = qp.stats.norm(  # pylint: disable=no-member
            loc=np.array([[1], [2.5], [3]]), scale=np.array([[0.25], [0.5], [0.1]])
        )

        self.user_defined_quantiles = np.linspace(0.001, 0.999, 16)
        self.user_defined_grid = np.linspace(0, 5, 100)
        self.pdf_constructor = CdfSplineDerivative

    def test_instantiation(self):
        """Base case make sure that we can instantiate the class"""
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        assert isinstance(pdf_constructor, AbstractQuantilePdfConstructor)

    def test_instantiation_for_multiple_distributions(self):
        """Base case make sure that we can instantiate a pdf reconstructor for multiple distributions"""
        user_defined_locations = self.many_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        assert isinstance(pdf_constructor, AbstractQuantilePdfConstructor)

    def test_debug(self):
        """Ensure that debug returns expected values before `prepare_constructor` has been run"""
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        (
            debug_quantiles,
            debug_locations,
            debug_interp_functions,
        ) = pdf_constructor.debug()

        print(debug_quantiles)
        assert np.all(np.isclose(debug_quantiles, self.user_defined_quantiles))
        assert np.all(np.isclose(debug_locations, user_defined_locations))
        self.assertIsNone(debug_interp_functions)

    def test_basic_construct_pdf(self):
        """Base case to ensure that `construct_pdf` method runs with minimum arguments for
        single distribution case.  Want to verify only that the machinery is working,
        and the result is not just zeros.
        """
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        results = pdf_constructor.construct_pdf(self.user_defined_grid)
        self.assertIsNot(np.sum(results), 0.0)

    def test_basic_construct_pdf_for_multiple_distributions(self):
        """Base case to ensure that `construct_pdf` method runs with minimum arguments
        for many-distribution case.  Want to verify only that the machinery is working,
        and the result is not just zeros.
        """
        user_defined_locations = self.many_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        results = pdf_constructor.construct_pdf(self.user_defined_grid)
        self.assertIsNot(np.sum(results), 0.0)

    def test_passing_different_spline_orders(self):
        """Verify that passing a new spline_order value to `prepare_constructor` results in a change."""
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        default_spline_order = pdf_constructor.construct_pdf(self.user_defined_grid)

        # change the spline order - expect the reconstructed pdf to be different than
        # the one based on the default spline order (3)
        pdf_constructor.prepare_constructor(spline_order=2)
        different_spline_order = pdf_constructor.construct_pdf(self.user_defined_grid)

        assert np.any(np.not_equal(default_spline_order, different_spline_order))

    def test_basic_construct_pdf_for_subset_of_multiple_distributions(self):
        """Base case to ensure that `construct_pdf` method runs with minimum arguments
        for many-distribution case when passing in a `row` value. Want to verify only
        that the machinery is working, and that the size of the output matches expectations
        """
        user_defined_locations = self.many_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(
            quantiles=self.user_defined_quantiles, locations=user_defined_locations
        )
        user_defined_rows = [0, 1]
        results = pdf_constructor.construct_pdf(
            self.user_defined_grid, row=user_defined_rows
        )
        self.assertEqual(len(results), 2)
