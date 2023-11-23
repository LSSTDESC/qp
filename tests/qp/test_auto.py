"""
Unit tests for PDF class
"""
import unittest
from functools import partial

import qp
from qp import test_funcs


class PDFTestCase(unittest.TestCase):
    """Class to manage automatically generated tests for qp distributions"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """

    def tearDown(self):
        "Clean up any mock data files created by the tests."

    @classmethod
    def auto_add_class(cls, test_class, ens_list):
        """Add tests as member functions to a class"""
        for key, val in test_class.test_data.items():
            test_pdf = val.pop("test_pdf", True)
            if test_pdf:
                kw_test_pdf = dict(
                    short=val.pop("short", False),
                    check_props=val.pop("check_props", True),
                )
                the_pdf_func = partial(
                    test_funcs.run_pdf_func_tests, test_class, val, **kw_test_pdf
                )
                setattr(cls, "test_pdf_%s" % key, the_pdf_func)
            test_persist = val.pop("test_persist", True)
            if test_persist:
                the_persist_func = partial(test_funcs.run_persist_func_tests, val)
                setattr(cls, "test_persist_%s" % key, the_persist_func)
            test_convert = val.pop("test_convert", True)
            if "convert_data" not in val:
                test_convert = False
            if test_convert:
                kw_test_convert = dict(
                    atol_diff=val.pop("atol_diff", 1e-2),
                    atol_diff2=val.pop("atol_diff2", 1e-2),
                )
                for i, ens in enumerate(ens_list):
                    the_convert_func = partial(
                        test_funcs.run_convert_tests,
                        ens_orig=ens,
                        gen_class=test_class,
                        test_data=val,
                        **kw_test_convert,
                    )
                    setattr(cls, "test_convert_%s_%i" % (key, i), the_convert_func)
            test_plot = val.pop("test_plot", True)
            if test_plot:
                kw_test_plot = dict(do_samples=val.pop("do_samples", False))
                the_plot_func = partial(
                    test_funcs.run_plotting_func_tests, val, **kw_test_plot
                )
                setattr(cls, "test_plotting_%s" % key, the_plot_func)

    @classmethod
    def auto_add(cls, class_list, ens_orig):
        """Add tests as member functions to a class"""
        for test_class in class_list:
            try:
                test_class.make_test_data()
            except AttributeError:
                pass
            if hasattr(test_class, "test_data"):
                cls.auto_add_class(test_class, ens_orig)


ENS_ORIG = test_funcs.build_ensemble(
    qp.stats.norm_gen.test_data["norm"]  # pylint: disable=no-member
)
ENS_MULTI = test_funcs.build_ensemble(
    qp.stats.norm_gen.test_data["norm"]  # pylint: disable=no-member
)
TEST_CLASSES = qp.instance().values()

PDFTestCase.auto_add(TEST_CLASSES, [ENS_ORIG, ENS_MULTI])


if __name__ == "__main__":
    unittest.main()
