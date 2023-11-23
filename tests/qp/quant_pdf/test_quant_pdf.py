"""
Unit tests for quant_pdf class
"""
import logging
import unittest

import numpy as np

import qp
from qp.quantile_pdf_constructors import AbstractQuantilePdfConstructor


class QuantPdfTestCase(unittest.TestCase):
    """Class to test quant_pdf qp.Ensemble functionality"""

    def test_quant_get_default_pdf_constructor_name(self):
        """Test that the getter for pdf constructor name works"""
        quantiles = np.linspace(0.001, 0.999, 16)
        locations = np.linspace(0, 5, 16)
        quant_dist = qp.quant(quants=quantiles, locs=locations)
        self.assertEqual(quant_dist.dist.pdf_constructor_name, "piecewise_linear")

    def test_quant_get_default_pdf_constructor(self):
        """Test that the getter for pdf constructor returns an AbstractQuantilePdfConstructor"""
        quantiles = np.linspace(0.001, 0.999, 16)
        locations = np.linspace(0, 5, 16)
        quant_dist = qp.quant(quants=quantiles, locs=locations)
        assert isinstance(
            quant_dist.dist.pdf_constructor, AbstractQuantilePdfConstructor
        )

    def test_quant_change_pdf_constructor(self):
        """Test that changing the pdf constructor works as expected"""
        quantiles = np.linspace(0.001, 0.999, 16)
        locations = np.linspace(0, 5, 16)
        quant_dist = qp.quant(quants=quantiles, locs=locations)
        quant_dist.dist.pdf_constructor_name = "piecewise_constant"
        self.assertEqual(quant_dist.dist.pdf_constructor_name, "piecewise_constant")

    def test_quant_change_pdf_constructor_raises(self):
        """Verify that attempting to change the pdf constructor to one that
        isn't in the dictionary, will raise an error."""
        quantiles = np.linspace(0.001, 0.999, 16)
        locations = np.linspace(0, 5, 16)
        quant_dist = qp.quant(quants=quantiles, locs=locations)
        with self.assertRaises(ValueError):
            quant_dist.dist.pdf_constructor_name = "drewtonian"

    def test_quant_change_pdf_constructor_warns(self):
        """Verify that attempting to change the pdf constructor to the one
        currently being used will log a warning."""
        quantiles = np.linspace(0.001, 0.999, 16)
        locations = np.linspace(0, 5, 16)
        quant_dist = qp.quant(quants=quantiles, locs=locations)
        with self.assertLogs(level=logging.WARNING) as log:
            quant_dist.dist.pdf_constructor_name = "piecewise_linear"
            self.assertIn("Already using", log.output[0])
