"""
Unit tests for converting PDF classes
"""

import numpy as np
from scipy import stats

import unittest

import qp

class EnsembleTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass

    def tearDown(self):
        """
        Clean up any mock data files created by the tests.
        """
        pass

    
    def test_ensemble(self):
        """
        """
        n = 100
        means = stats.uniform.rvs(size=n)
        scales = stats.uniform.rvs(size=n)

        quantiles = np.linspace(0.001, 0.999)
        limits = (-6., 6.,)
        bins = np.linspace(limits[0], limits[1], 121)
        means2 = means + 0.05 * stats.uniform.rvs(size=n)
        scales2 = scales * (1. * 0.05 * stats.uniform.rvs(size=n))

        dists = [ stats.norm(loc=mean, scale=scale) for (mean, scale) in zip(means, scales) ]
        ens = qp.Ensemble(dists)

        dists2 = [ stats.norm(loc=mean, scale=scale) for (mean, scale) in zip(means2, scales2) ]
        ens2 = qp.Ensemble(dists2)

        yvals = ens.evaluate(bins)
        integrals = ens.integrate(limits)
        samples = ens.sample()        
        qvalues = ens.quantize(quantiles)

        hists = ens.histogramize(bins)

        m1 = ens.moment(1)
        m2 = ens.moment(2)

        kld = ens.kld(ens2)
        rmse = ens.rmse(ens2)
        
        
        

if __name__ == '__main__':
    unittest.main()
