import numpy as np
from qp.metrics.base_metric_classes import (
    MetricOutputType,
    PointToPointMetric,
)
from pytdigest import TDigest
from functools import reduce
from operator import add

class EvaluateFromIteratorMixin:
    def eval_from_iterator(self, estimate, reference):
        self.initialize()
        for estimate, reference in zip(estimate, reference):
            centroids = self.accumulate(estimate, reference)
        return self.finalize([centroids])

class PointToPointMetricTDigester(PointToPointMetric, EvaluateFromIteratorMixin):

    def __init__(self, tdigest_compression: int = 1000, **kwargs) -> None:
        super().__init__()
        self._tdigest_compression = tdigest_compression

    def initialize(self):
        pass

    def evaluate(self, estimate, reference, tdigest_compression=None):
        self._do_type_check(estimate, reference)
        if tdigest_compression is None:
            tdigest_compression = self._tdigest_compression
        self._evaluate(estimate, reference, tdigest_compression)

    def _do_type_check(self, estimate, reference):
        #check if estimate is an iterator
        if not hasattr(estimate, '__iter__'):
            raise ValueError('estimate should be an iterator')  #pragma: no cover

        #check if reference is an iterator
        if not hasattr(reference, '__iter__'):
            raise ValueError('reference should be an iterator') #pragma: no cover

    def _evaluate(self, estimate, reference, tdigest_compression):
        return self.eval_from_iterator(estimate, reference, tdigest_compression)

    def accumulate(self, estimate, reference):
        """This function compresses the input into a TDigest and returns the
        centroids.

        Parameters
        ----------
        estimate : Numpy 1d array
            Point estimate values
        reference : Numpy 1d array
            True values

        Returns
        -------
        Numpy 2d array
            The centroids of the TDigest. Roughly approximates a histogram with
            centroid locations and weights.
        """
        ez = (estimate - reference) / (1.0 + reference)
        digest = TDigest.compute(ez, compression=self._tdigest_compression)
        centroids = digest.get_centroids()
        return centroids

    def finalize(self, centroids: np.ndarray = []):
        """This function combines all the centroids that were calculated for the
        input estimate and reference subsets and returns the resulting TDigest
        object.

        Parameters
        ----------
        centroids : Numpy 2d array, optional
            The output collected from prior calls to `accumulate`, by default []

        Returns
        -------
        float
            The result of the specific metric calculation defined in the subclasses
            `compute_from_digest` method.
        """
        digests = (
            TDigest.of_centroids(np.array(centroid), compression=self._tdigest_compression)
            for centroid in centroids
        )
        digest = reduce(add, digests)

        return self._compute_from_digest(digest)

    def _compute_from_digest(self, digest):  #pragma: no cover
        raise NotImplementedError


class PointSigmaIQR_digest(PointToPointMetricTDigester):
    """Calculate sigmaIQR with t-digest approximation"""

    metric_name = "point_stats_iqr_tdigest"
    metric_output_type = MetricOutputType.single_value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _compute_from_digest(self, digest):
        x75, x25 = digest.inverse_cdf([0.75,0.25])
        iqr = x75 - x25
        sigma_iqr = iqr / 1.349
        return sigma_iqr


class PointBias(PointToPointMetricTDigester):
    """calculates the bias of the point stats ez samples with t-digest approximation.
    In keeping with the Science Book, this is just the median of the ez values.
    """

    metric_name = "point_bias"
    metric_output_type = MetricOutputType.single_value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute_from_digest(self, digest):
        return digest.inverse_cdf(0.50)