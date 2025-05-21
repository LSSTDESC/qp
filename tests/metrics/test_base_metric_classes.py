# pylint: disable=protected-access

import pytest

from qp.metrics.base_metric_classes import (DistToDistMetric,
                                            DistToPointMetric,
                                            PointToDistMetric,
                                            PointToPointMetric,
                                            SingleEnsembleMetric)


def test_single_ensemble_metrics():
    """Test single ensemble basic properties"""
    limits = (0.01, 2.99)
    dx = 0.001
    single_ens_metric = SingleEnsembleMetric(limits, dx)
    assert single_ens_metric._limits == limits
    assert single_ens_metric._dx == dx
    assert SingleEnsembleMetric.uses_distribution_for_estimate() is True
    assert SingleEnsembleMetric.uses_distribution_for_reference() is False
    assert SingleEnsembleMetric.uses_point_for_estimate() is False
    assert SingleEnsembleMetric.uses_point_for_reference() is False


def test_single_ensemble_raises_unimplemented():
    """Test single ensemble raises exceptions"""
    limits = (0.01, 2.99)
    dx = 0.001
    single_ens_metric = SingleEnsembleMetric(limits, dx)

    pytest.raises(NotImplementedError, single_ens_metric.evaluate, estimate=None)


def test_dist_to_dist_metrics():
    """Test dist to dist basic properties"""
    limits = (0.01, 2.99)
    dx = 0.001
    dist_to_dist_metric = DistToDistMetric(limits, dx)
    assert dist_to_dist_metric._limits == limits
    assert dist_to_dist_metric._dx == dx
    assert DistToDistMetric.uses_distribution_for_estimate() is True
    assert DistToDistMetric.uses_distribution_for_reference() is True
    assert DistToDistMetric.uses_point_for_estimate() is False
    assert DistToDistMetric.uses_point_for_reference() is False


def test_dist_to_dist_raises_unimplemented():
    """Test dist to dist raises exceptions"""
    limits = (0.01, 2.99)
    dx = 0.001
    dist_to_dist_metric = DistToDistMetric(limits, dx)

    pytest.raises(
        NotImplementedError, dist_to_dist_metric.evaluate, estimate=None, reference=None
    )


def test_dist_to_point_metrics():
    """Test dist to point basic properties"""
    limits = (0.01, 2.99)
    dx = 0.001
    dist_to_point_metric = DistToPointMetric(limits, dx)
    assert dist_to_point_metric._limits == limits
    assert dist_to_point_metric._dx == dx
    assert DistToPointMetric.uses_distribution_for_estimate() is True
    assert DistToPointMetric.uses_distribution_for_reference() is False
    assert DistToPointMetric.uses_point_for_estimate() is False
    assert DistToPointMetric.uses_point_for_reference() is True


def test_dist_to_point_raises_unimplemented():
    """Test dist to point raises exceptions"""
    limits = (0.01, 2.99)
    dx = 0.001
    dist_to_point_metric = DistToPointMetric(limits, dx)

    pytest.raises(
        NotImplementedError,
        dist_to_point_metric.evaluate,
        estimate=None,
        reference=None,
    )


def test_point_to_point_metrics():
    """Test point to point basic properties"""
    limits = (0.01, 2.99)
    dx = 0.001
    point_to_point_metric = PointToPointMetric(limits, dx)
    assert point_to_point_metric._limits == limits
    assert point_to_point_metric._dx == dx
    assert PointToPointMetric.uses_distribution_for_estimate() is False
    assert PointToPointMetric.uses_distribution_for_reference() is False
    assert PointToPointMetric.uses_point_for_estimate() is True
    assert PointToPointMetric.uses_point_for_reference() is True


def test_point_to_point_raises_unimplemented():
    """Test point to point raises exceptions"""
    limits = (0.01, 2.99)
    dx = 0.001
    point_to_point_metric = PointToPointMetric(limits, dx)

    pytest.raises(
        NotImplementedError,
        point_to_point_metric.evaluate,
        estimate=None,
        reference=None,
    )


def test_point_to_dist_metrics():
    """Test point to dist basic properties"""
    limits = (0.01, 2.99)
    dx = 0.001
    point_to_dist_metric = PointToDistMetric(limits, dx)
    assert point_to_dist_metric._limits == limits
    assert point_to_dist_metric._dx == dx
    assert PointToDistMetric.uses_distribution_for_estimate() is False
    assert PointToDistMetric.uses_distribution_for_reference() is True
    assert PointToDistMetric.uses_point_for_estimate() is True
    assert PointToDistMetric.uses_point_for_reference() is False


def test_point_to_dist_raises_unimplemented():
    """Test point to dist raises exceptions"""
    limits = (0.01, 2.99)
    dx = 0.001
    point_to_dist_metric = PointToDistMetric(limits, dx)

    pytest.raises(
        NotImplementedError,
        point_to_dist_metric.evaluate,
        estimate=None,
        reference=None,
    )
