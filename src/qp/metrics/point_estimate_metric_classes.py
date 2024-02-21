import numpy as np
from qp.metrics.base_metric_classes import (
    MetricOutputType,
    PointToPointMetric,
)
from pytdigest import TDigest
from functools import reduce
from operator import add


class PointToPointMetricDigester(PointToPointMetric):

    def __init__(self, tdigest_compression: int = 1000, **kwargs) -> None:
        super().__init__()
        self._tdigest_compression = tdigest_compression

    def initialize(self):
        pass

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

        return self.compute_from_digest(digest)

    def compute_from_digest(self, digest):  #pragma: no cover
        raise NotImplementedError


class PointStatsEz(PointToPointMetric):
    """Copied from PZDC1paper repo. Adapted to remove the cut based on
    magnitude."""

    metric_name = "point_stats_ez"

    #! This doesn't seem quiet correct, perhaps we need a `single_value_per_input_element` ???
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, estimate, reference):
        """A calculation that takes in vectors of the estimated point values
        the known true values.

        Parameters
        ----------
        estimate : Numpy 1d array
            Point estimate values
        reference : Numpy 1d array
            True values

        Returns
        -------
        Numpy 1d array
            The result of calculating (estimate-reference)/(1+reference)
        """

        return (estimate - reference) / (1.0 + reference)


class PointSigmaIQR(PointToPointMetricDigester):
    """Calculate sigmaIQR"""

    metric_name = "point_stats_iqr"
    metric_output_type = MetricOutputType.single_value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, estimate, reference):
        """Calculate the width of the e_z distribution
        using the Interquartile range

        Parameters
        ----------
        estimate : Numpy 1d array
            Point estimate values
        reference : Numpy 1d array
            True values

        Returns
        -------
        float
            The interquartile range.
        """
        ez = (estimate - reference) / (1.0 + reference)
        x75, x25 = np.percentile(ez, [75.0, 25.0])
        iqr = x75 - x25
        sigma_iqr = iqr / 1.349
        return sigma_iqr

    def compute_from_digest(self, digest):
        x75, x25 = digest.inverse_cdf([0.75,0.25])
        iqr = x75 - x25
        sigma_iqr = iqr / 1.349
        return sigma_iqr


class PointBias(PointToPointMetricDigester):
    """calculates the bias of the point stats ez samples.
    In keeping with the Science Book, this is just the median of the ez values.
    """

    metric_name = "point_bias"
    metric_output_type = MetricOutputType.single_value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, estimate, reference):
        """The point bias, or median of the point stats ez samples.

        Parameters
        ----------
        estimate : Numpy 1d array
            Point estimate values
        reference : Numpy 1d array
            True values

        Returns
        -------
        float
            Median of the ez values
        """
        return np.median((estimate - reference) / (1.0 + reference))

    def compute_from_digest(self, digest):
        return digest.inverse_cdf(0.50)


class PointOutlierRate(PointToPointMetricDigester):
    """Calculates the catastrophic outlier rate, defined in the
    Science Book as the number of galaxies with ez larger than
    max(0.06,3sigma).  This keeps the fraction reasonable when
    sigma is very small.
    """

    metric_name = "point_outlier_rate"
    metric_output_type = MetricOutputType.single_value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, estimate, reference):
        """Calculates the catastrophic outlier rate

        Parameters
        ----------
        estimate : Numpy 1d array
            Point estimate values
        reference : Numpy 1d array
            True values

        Returns
        -------
        float
            Fraction of catastrophic outliers for full sample
        """

        ez = (estimate - reference) / (1.0 + reference)
        num = len(ez)
        sig_iqr = PointSigmaIQR().evaluate(estimate, reference)
        three_sig = 3.0 * sig_iqr
        cut_criterion = np.maximum(0.06, three_sig)
        mask = np.fabs(ez) > cut_criterion
        outlier = np.sum(mask)
        return float(outlier) / float(num)

    def compute_from_digest(self, digest):
        # this replaces the call to PointSigmaIQR().evaluate()
        x75, x25 = digest.inverse_cdf([0.75,0.25])
        iqr = x75 - x25
        sigma_iqr = iqr / 1.349

        three_sig = 3.0 * sigma_iqr
        cut_criterion = np.maximum(0.06, three_sig)

        # here we use the number of points in the centroids as an approximation
        # of ez.
        centroids = digest.get_centroids()
        mask = np.fabs(centroids[:,0]) > cut_criterion
        outlier = np.sum(centroids[mask,1])

        # Since we use equal weights for all the values in the digest
        # digest.weight is the total number of values, and is stored as a float.
        return float(outlier) / digest.weight


class PointSigmaMAD(PointToPointMetricDigester):
    """Function to calculate median absolute deviation and sigma
    based on MAD (just scaled up by 1.4826) for the full and
    magnitude trimmed samples of ez values
    """

    metric_name = "point_stats_sigma_mad"
    metric_output_type = MetricOutputType.single_value

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._num_bins = 1_000_000
        if "num_bins" in kwargs:
            self._num_bins = kwargs["num_bins"]

    def evaluate(self, estimate, reference):
        """Function to calculate SigmaMAD (the median absolute deviation scaled
        up by constant factor, ``SCALE_FACTOR``.

        Parameters
        ----------
        estimate : Numpy 1d array
            Point estimate values
        reference : Numpy 1d array
            True values

        Returns
        -------
        float
            sigma_MAD for full sample
        """

        SCALE_FACTOR = 1.4826
        ez = (estimate - reference) / (1.0 + reference)
        mad = np.median(np.fabs(ez - np.median(ez)))
        return mad * SCALE_FACTOR

    def compute_from_digest(self, digest):
        SCALE_FACTOR = 1.4826

        # calculation of `np.median(np.fabs(ez - np.median(ez)))` as suggested by Eric Charles
        this_median = digest.inverse_cdf(0.50)
        this_min = digest.inverse_cdf(0)
        this_max = digest.inverse_cdf(1)
        bins = np.linspace(this_min, this_max, self._num_bins)
        bin_cents = (bins[0:-1] + bins[1:]) / 2.0
        this_pdf = digest.cdf(bins[1:]) - digest.cdf(bins[0:-1]) # len(this_pdf) = lots_of_bins - 1
        bin_dist = np.fabs(bin_cents - this_median) # get the distance to the center for each bin in the hist

        sorted_bins_dist_idx = np.argsort(bin_dist) # sort the bins by dist to median
        sorted_bins_dist = bin_dist[sorted_bins_dist_idx] # get the sorted distances
        cumulative_sorted = this_pdf[sorted_bins_dist_idx].cumsum() # the cumulate PDF within the nearest bins
        median_sorted_bin = np.searchsorted(cumulative_sorted, 0.5) # which bins are the nearest 50% of the PDF
        dist_to_median = sorted_bins_dist[median_sorted_bin] # return the corresponding distance to the median

        return dist_to_median * SCALE_FACTOR
