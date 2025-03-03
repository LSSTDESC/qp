# pylint: disable=no-member
# pylint: disable=protected-access

"""
Unit tests for PDF class
"""
import unittest

import qp
import qp.metrics


class MetricFactoryTestCase(unittest.TestCase):
    """Tests for the metric factory"""

    def setUp(self):
        """
        Setup an objects that are used in multiple tests.
        """
        qp.metrics.update_metrics()

    def test_print_metrics(self):
        """Test printing the metrics."""
        qp.metrics.print_metrics()
        qp.metrics.print_metrics(force_update=True)

    def test_list_metrics(self):
        """Test printing the metrics."""
        the_list = qp.metrics.list_metrics()
        assert the_list
        the_list = qp.metrics.list_metrics(force_update=True)
        assert the_list

    def test_create_metrics(self):
        """Test creating all the metrics"""
        all_metric_names = qp.metrics.list_metrics(force_update=True)
        for metric_name in all_metric_names:
            a_metric = qp.metrics.create_metric(metric_name)
            assert a_metric.metric_name == metric_name
        a_metric = qp.metrics.create_metric("outlier", force_update=True)
        assert a_metric.metric_name == "outlier"

    def test_bad_metric_name(self):
        """Catch error on making a bad metric"""
        with self.assertRaises(KeyError):
            qp.metrics.create_metric("Bad Metric")


if __name__ == "__main__":
    unittest.main()
