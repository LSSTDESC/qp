"""
Unit tests for PDF class
"""
import sys
import os
import unittest
from functools import partial
import qp

from qp import test_data
from qp import test_funcs


class PDFTestCase(unittest.TestCase):
    
    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass

    @classmethod
    def auto_add_class(cls, test_class, ens_orig):
        """Add tests as member functions to a class"""
        for key, val in test_class.test_data.items():
            test_pdf = val.pop('test_pdf', True)
            if test_pdf:
                kw_test_pdf = dict(short=val.pop('short', False), check_props=val.pop('check_props', True))
                the_pdf_func = partial(test_funcs.run_pdf_func_tests, test_class, val, **kw_test_pdf)
                setattr(cls, 'test_pdf_%s' % key, the_pdf_func)
            test_persist = val.pop('test_persist', True)
            if test_persist:
                the_persist_func = partial(test_funcs.run_persist_func_tests, val)
                setattr(cls, 'test_persist_%s' % key, the_persist_func)
            test_convert = val.pop('test_convert', True)
            if 'convert_data' not in val:
                test_convert = False
            if test_convert:
                kw_test_convert = dict(atol_diff2=val.pop('atol_diff2', 1e-2))
                the_convert_func = partial(test_funcs.run_convert_tests, ens_orig=ens_orig, gen_class=test_class, test_data=val, **kw_test_convert)
                setattr(cls, 'test_convert_%s' % key, the_convert_func)
            test_plot = val.pop('test_plot', True)
            if test_plot:
                kw_test_plot = dict(do_samples=val.pop('do_samples', False))
                the_plot_func = partial(test_funcs.run_plotting_func_tests, val, **kw_test_plot)
                setattr(cls, 'test_plotting_%s' % key, the_plot_func)
            
    @classmethod
    def auto_add(cls, class_list, ens_orig):
        """Add tests as member functions to a class"""
        for test_class in class_list:
            if hasattr(test_class, 'test_data'):
                cls.auto_add_class(test_class, ens_orig)
        

ENS_ORIG = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm'])
TEST_CLASSES = qp.instance().values()

PDFTestCase.auto_add(TEST_CLASSES, ENS_ORIG)

    
if __name__ == '__main__':
    unittest.main()
