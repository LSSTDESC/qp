import unittest
import numpy as np

import qp
from qp.metrics.concrete_metric_classes import CDELossMetric
from qp.metrics.point_estimate_metric_classes import (PointBias,
                                                      PointOutlierRate,
                                                      PointSigmaIQR,
                                                      PointSigmaMAD,
                                                      PointStatsEz)

# values for metrics
OUTRATE = 0.0
CDEVAL = -4.31200
SIGIQR = 0.0045947
BIAS = -0.00001576
OUTRATE = 0.0
SIGMAD = 0.0046489

def construct_test_ensemble():
    np.random.seed(87)
    nmax = 2.5
    NPDF = 399
    true_zs = np.random.uniform(high=nmax, size=NPDF)
    locs = np.expand_dims(true_zs + np.random.normal(0.0, 0.01, NPDF), -1)
    true_ez = (locs.flatten() - true_zs) / (1.0 + true_zs)
    scales = np.ones((NPDF, 1)) * 0.1 + np.random.uniform(size=(NPDF, 1)) * 0.05

    # pylint: disable=no-member
    n_ens = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))
    zgrid = np.linspace(0, nmax, 301)
    grid_ens = n_ens.convert_to(qp.interp_gen, xvals=zgrid)
    return zgrid, true_zs, grid_ens, true_ez


#generator that yields chunks from estimate and reference
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class test_point_metrics(unittest.TestCase):

    def test_point_metrics(self):
        """Basic tests for the various point estimate metrics"""
        zgrid, zspec, pdf_ens, true_ez = construct_test_ensemble()
        zb = pdf_ens.mode(grid=zgrid).flatten()

        ez = PointStatsEz().evaluate(zb, zspec)
        assert np.allclose(ez, true_ez, atol=1.0e-2)

        # grid limits ez vals to ~10^-2 tol

        sig_iqr = PointSigmaIQR().evaluate(zb, zspec)
        assert np.isclose(sig_iqr, SIGIQR)

        bias = PointBias().evaluate(zb, zspec)
        assert np.isclose(bias, BIAS)

        out_rate = PointOutlierRate().evaluate(zb, zspec)
        assert np.isclose(out_rate, OUTRATE)

        sig_mad = PointSigmaMAD().evaluate(zb, zspec)
        assert np.isclose(sig_mad, SIGMAD)

    def test_point_metrics_digest(self):
        """Basic tests for the various point estimate metrics when using the
        t-digest approximation."""

        zgrid, zspec, pdf_ens, true_ez = construct_test_ensemble()
        zb = pdf_ens.mode(grid=zgrid).flatten()

        configuration = {'tdigest_compression': 5000}
        point_sigma_iqr = PointSigmaIQR(**configuration)
        centroids = point_sigma_iqr.accumulate(zb, zspec)
        sig_iqr = point_sigma_iqr.finalize([centroids])
        assert np.isclose(sig_iqr, SIGIQR, atol=1.0e-4)

        zb_iter = chunker(zb, 100)
        zspec_iter = chunker(zspec, 100)
        
        sig_iqr_v2 = point_sigma_iqr.eval_from_iterator(zb_iter, zspec_iter)

        point_bias = PointBias(**configuration)
        centroids = point_bias.accumulate(zb, zspec)
        bias = point_bias.finalize([centroids])
        assert np.isclose(bias, BIAS)

        point_outlier_rate = PointOutlierRate(**configuration)
        centroids = point_outlier_rate.accumulate(zb, zspec)
        out_rate = point_outlier_rate.finalize([centroids])
        assert np.isclose(out_rate, OUTRATE)

        point_sigma_mad = PointSigmaMAD(**configuration)
        centroids = point_sigma_mad.accumulate(zb, zspec)
        sig_mad = point_sigma_mad.finalize([centroids])
        assert np.isclose(sig_mad, SIGMAD, atol=1e-5)

        configuration = {'tdigest_compression': 5000, 'num_bins': 1_000}
        point_sigma_mad = PointSigmaMAD(**configuration)
        centroids = point_sigma_mad.accumulate(zb, zspec)
        sig_mad = point_sigma_mad.finalize([centroids])
        assert np.isclose(sig_mad, SIGMAD, atol=1e-4)



    def test_cde_loss_metric(self):
        """Basic test to ensure that the CDE Loss metric class is working."""
        zgrid, zspec, pdf_ens, _ = construct_test_ensemble()
        cde_loss_class = CDELossMetric(zgrid)
        result = cde_loss_class.evaluate(pdf_ens, zspec)
        assert np.isclose(result, CDEVAL)

        chunk_output = cde_loss_class.accumulate(pdf_ens, zspec)
        chunked_result = cde_loss_class.finalize([chunk_output])

        assert np.isclose(chunked_result, CDEVAL)
