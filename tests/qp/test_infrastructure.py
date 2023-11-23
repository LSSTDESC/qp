"""
Unit tests for PDF class
"""
import os
import unittest

import qp
from qp import test_data
from qp.test_funcs import build_ensemble


class InfrastructureTestCase(unittest.TestCase):
    """Class with tests of infrastructure"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.files = []

    def tearDown(self):
        "Clean up any mock data files created by the tests."
        for ff in self.files:
            os.unlink(ff)

    @staticmethod
    def test_print_factory():
        """Test the print_factory method"""
        qp.instance().pretty_print()

    @staticmethod
    def test_slice_dict():
        """Test the slice_dict method"""
        orig_dict = dict(loc=test_data.LOC, scale=test_data.SCALE, scalar=1)
        sliced = qp.dict_utils.slice_dict(orig_dict, 1)
        assert sliced["loc"] == test_data.LOC[1]
        assert sliced["scale"] == test_data.SCALE[1]
        assert sliced["scalar"] == 1

    @staticmethod
    def test_print_dict_shape():
        """Test the print_dict_shape method"""
        test_dict = dict(loc=test_data.LOC, scale=test_data.SCALE)
        qp.dict_utils.print_dict_shape(test_dict)

    @staticmethod
    def test_get_val_or_default():
        """Test the get_val_or_default method"""
        test_dict = dict(key=1)
        test_dict[None] = 2
        assert qp.dict_utils.get_val_or_default(test_dict, "key") == 1
        assert qp.dict_utils.get_val_or_default(test_dict, "nokey") == 2
        assert qp.dict_utils.get_val_or_default(test_dict, None) == 2
        assert qp.dict_utils.set_val_or_default(test_dict, "key", 5) == 1

        test_dict.pop(None)
        assert qp.dict_utils.get_val_or_default(test_dict, "nokey") is None

    def test_is_qp_file(self):
        fname = "norm_ensemble.hdf5"
        norm_test_data = qp.stats.norm_gen.test_data["norm"]  # pylint: disable=no-member
        ens_norm = build_ensemble(norm_test_data)
        ens_norm.write_to(fname)
        self.files.append(fname)
        assert qp.instance().is_qp_file(fname)
        assert not qp.instance().is_qp_file("test_pit.py")


if __name__ == "__main__":
    unittest.main()
