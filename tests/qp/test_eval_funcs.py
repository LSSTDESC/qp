"""
Unit tests for PDF class
"""
import unittest

import numpy as np

import qp


class EvalFuncsTestCase(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Tests of evaluations and interpolation functions"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.xpts = np.linspace(0, 3, 7)
        self.hpdfs = np.random.random((10, 50))  # pylint: disable=no-member
        self.hbins = np.linspace(0, 5, 51)
        self.hbins2 = np.linspace(0, 5, 51) + np.expand_dims(
            np.linspace(0.1, 1.0, 10), -1
        )

        self.xvals = np.linspace(0, 5, 50)
        self.xvals2 = np.linspace(0, 5, 50) + np.expand_dims(
            np.linspace(0.1, 1.0, 10), -1
        )
        self.yvals1d = self.hpdfs[0]

        self.rows = np.expand_dims(np.arange(10), -1)
        self.grid = self.xpts * np.ones((10, 7))

        self.range_grid = (self.rows * np.ones((10), int)).astype(int)

    def tearDown(self):
        "Clean up any mock data files created by the tests."

    def _check_interface_function(self, ifunc, xvals, yvals, **kwargs):
        v0 = ifunc(self.xpts, self.rows, xvals, yvals, **kwargs)
        v1 = ifunc(self.grid.flatten(), self.rows.flatten(), xvals, yvals, **kwargs)
        v2 = ifunc(self.grid, self.rows, xvals, yvals, **kwargs)
        _ = ifunc(self.xpts, np.arange(7), xvals, yvals, **kwargs)

        assert np.allclose(v0, v1)
        assert np.allclose(v0, v2)

    def test_evaluate_hist_x_multi_y(self):
        """Test the evaluate_hist_x_multi_y function"""
        self._check_interface_function(
            qp.utils.evaluate_hist_x_multi_y, self.hbins, self.hpdfs
        )

    def test_evaluate_hist_multi_x_multi_y(self):
        """Test the evaluate_hist_multi_x_multi_y function"""
        self._check_interface_function(
            qp.utils.evaluate_hist_multi_x_multi_y, self.hbins2, self.hpdfs
        )

    def test_interpolate_x_multi_y(self):
        """Test the interpolate_x_multi_y"""
        self._check_interface_function(
            qp.utils.interpolate_x_multi_y,
            self.xvals,
            self.hpdfs,
            bounds_error=False,
            fill_value=0,
        )

    def test_interpolate_multi_x_multi_y(self):
        """Test the interpolate_multi_x_multi_y"""
        self._check_interface_function(
            qp.utils.interpolate_multi_x_multi_y,
            self.xvals2,
            self.hpdfs,
            bounds_error=False,
            fill_value=0,
        )

    def test_interpolate_multi_x_y(self):
        """Test the interpolate_multi_x_y"""
        self._check_interface_function(
            qp.utils.interpolate_multi_x_y,
            self.xvals2,
            self.yvals1d,
            bounds_error=False,
            fill_value=0,
        )


if __name__ == "__main__":
    unittest.main()
