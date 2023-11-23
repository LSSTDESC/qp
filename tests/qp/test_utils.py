"""
Unit tests for PDF class
"""
import unittest

import numpy as np

import qp
from qp import test_data


class UtilsTestCase(unittest.TestCase):
    """Test the utility functions"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """

    def tearDown(self):
        """Clean up any mock data files created by the tests."""

    @staticmethod
    def test_profile():
        """Test the utils.profile function"""
        npdf = 100
        x_bins = test_data.XBINS
        x_cents = qp.utils.edge_to_center(x_bins)
        nbin = x_cents.size
        x_data = (np.ones((npdf, 1)) * x_cents).T
        c_vals = np.linspace(0.5, 2.5, nbin)
        y_data = np.expand_dims(c_vals, -1) * (
            0.95 + 0.1 * np.random.uniform(size=(nbin, npdf))
        )
        pf_1 = qp.utils.profile(x_data.flatten(), y_data.flatten(), x_bins, std=False)
        pf_2 = qp.utils.profile(x_data.flatten(), y_data.flatten(), x_bins, std=True)
        qp.test_funcs.assert_all_close(
            pf_1[0], c_vals, atol=0.02, test_name="profile_mean"
        )
        qp.test_funcs.assert_all_close(pf_1[0], pf_2[0], test_name="profile_check")
        qp.test_funcs.assert_all_close(
            pf_1[1], c_vals / npdf * np.sqrt(12), atol=0.2, test_name="profile_std"
        )
        qp.test_funcs.assert_all_close(pf_1[1], 0.1 * pf_2[1], test_name="profile_err")

    def test_sparse(self):
        """Test the sparse representation"""

        xvals = np.linspace(0, 1, 101)
        # assert basic construction
        A = qp.sparse_rep.create_voigt_basis(xvals, (0, 1), 11, (0.01, 0.5), 10, 10)
        self.assertEqual(A.shape, (101, 1100))
        # check consistency of a constrained case od voigt basis
        pdf0 = np.exp(-((xvals - 0.5) ** 2) / (2.0 * 0.01)) / (np.sqrt(2 * np.pi) * 0.1)
        pdf2 = qp.sparse_rep.shapes2pdf(
            [
                1,
            ],
            [
                0.5,
            ],
            [
                0.1,
            ],
            [
                0,
            ],
            dict(xvals=xvals),
            cut=1.0e-7,
        )
        self.assertTrue(np.allclose(pdf2, pdf0))
        A = qp.sparse_rep.create_voigt_basis(xvals, (0.5, 0.5), 1, (0.1, 0.1), 1, 1)
        pdf1 = np.squeeze(A) * np.sqrt((pdf0**2).sum())
        self.assertTrue(np.allclose(pdf1, pdf0))
        # NSparse set to 2 so that unit testing goes through more code in sparse_basis
        ALL, bigD, _ = qp.sparse_rep.build_sparse_representation(
            xvals, [pdf0], (0.5, 0.5), 1, (0.1, 0.1), 1, 1, 2
        )
        va, ma, sa, ga = qp.sparse_rep.indices2shapes(ALL, bigD)
        self.assertEqual(va[:, 1], 0.0)
        self.assertEqual([va[:, 0], ma[:, 0], sa[:, 0], ga[:, 0]], [1.0, 0.5, 0.1, 0.0])
        # check default values
        ALL, bigD, A = qp.sparse_rep.build_sparse_representation(xvals, [pdf0])
        self.assertEqual(bigD["mu"], [min(xvals), max(xvals)])
        self.assertEqual(bigD["dims"][0], len(xvals))
        pdf_rec = qp.sparse_rep.pdf_from_sparse(ALL, A, xvals)
        self.assertTrue(np.allclose(pdf_rec[:, 0], pdf0, atol=1.5e-2))


if __name__ == "__main__":
    unittest.main()
