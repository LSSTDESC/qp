import numpy as np
from qp.metrics.base_metric_classes import (
    MetricOuputType,
    PointToPointMetric,
)


class PointStatsEz(PointToPointMetric):
    """Copied from PZDC1paper repo. Adapted to remove the cut based on
    magnitude."""

    metric_name = "point_stats_ez"

    #! This doesn't seem quiet correct, perhaps we need a `single_value_per_input_element` ???
    metric_output_type = MetricOuputType.one_value_per_distribution

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


class PointSigmaIQR(PointToPointMetric):
    """Calculate sigmaIQR"""

    metric_name = "point_stats_iqr"
    metric_output_type = MetricOuputType.single_value

    def __init__(self) -> None:
        super().__init__()

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


class PointBias(PointToPointMetric):
    """calculates the bias of the point stats ez samples.
    In keeping with the Science Book, this is just the median of the ez values.
    """

    metric_name = "point_bias"
    metric_output_type = MetricOuputType.single_value

    def __init__(self) -> None:
        super().__init__()

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


class PointOutlierRate(PointToPointMetric):
    """Calculates the catastrophic outlier rate, defined in the
    Science Book as the number of galaxies with ez larger than
    max(0.06,3sigma).  This keeps the fraction reasonable when
    sigma is very small.
    """

    metric_name = "point_outlier_rate"
    metric_output_type = MetricOuputType.single_value

    def __init__(self) -> None:
        super().__init__()

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


class PointSigmaMAD(PointToPointMetric):
    """Function to calculate median absolute deviation and sigma
    based on MAD (just scaled up by 1.4826) for the full and
    magnitude trimmed samples of ez values
    """

    metric_name = "point_stats_sigma_mad"
    metric_output_type = MetricOuputType.single_value

    def __init__(self) -> None:
        super().__init__()

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
