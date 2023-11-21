import numpy as np
import qp

from qp.metrics.point_estimate_metric_classes import (
    PointStatsEz,
    PointBias,
    PointOutlierRate,
    PointSigmaIQR,
    PointSigmaMAD,
)

from qp.metrics.concrete_metric_classes import CDELossMetric

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


def test_point_metrics():
    """Basic tests for the various point estimate metrics
    """
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


def test_cde_loss_metric():
    """Basic test to ensure that the CDE Loss metric class is working.
    """
    zgrid, zspec, pdf_ens, _ = construct_test_ensemble()
    cde_loss_class = CDELossMetric(zgrid)
    result = cde_loss_class.evaluate(pdf_ens, zspec)
    assert np.isclose(result, CDEVAL)
