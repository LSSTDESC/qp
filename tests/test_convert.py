"""
Unit tests for converting PDF classes
"""

import numpy as np
from scipy import stats

import unittest

import qp

class ConvertTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass

    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass


    def do_conversions(self, norm_dist):
        """
        """
        xvals = np.linspace(-4., 4., 33)
        quantiles = np.linspace(0.001, 0.999, 21)
        spline = qp.qp_convert(norm_dist, qp.spline_dist, xvals=xvals)
        intspline = qp.qp_convert(norm_dist, qp.intspline_dist, xvals=xvals)
        interp = qp.qp_convert(norm_dist, qp.interp_dist, xvals=xvals)
        kde = qp.qp_convert(norm_dist, qp.kde_dist)
        quant = qp.qp_convert(norm_dist, qp.quantile_dist, quantiles=quantiles)
        hist = qp.qp_convert(norm_dist, stats.rv_histogram, bins=xvals)
        mix_mod =  qp.qp_convert(norm_dist, qp.sum_dist, ncomps=5, nsamples=1000)

        pdfs = [norm_dist, spline, intspline, interp, kde, quant, hist, mix_mod]
        labels = ['truth', 'spline', 'intspline', 'interp', 'kde', 'quantiles', 'hist', 'mixmod']
        return pdfs, labels
        
    
    def test_convert(self):
        """
        """
        norm_dist = stats.norm(loc=0.0, scale=1.0)
        pdfs, labels = self.do_conversions(norm_dist)


    def test_plots(self):
        """
        """
        norm_dist = stats.norm(loc=0.0, scale=1.0)
        pdfs, labels = self.do_conversions(norm_dist)
        fig, axes =  qp.make_figure_axes(limits=(-6., 6.))

        xvals_plot = np.linspace(-6., 6., 121)

        for pdf, label in zip(pdfs, labels):
            #qp.plot_pdf_on_axes(axes, pdf, xvals_plot, label=label)
            qp.qp_plot_native(axes, pdf, label=label)

        leg = fig.legend()
        fig.savefig('testfig.pdf')
        
        

if __name__ == '__main__':
    unittest.main()
