# pylint: disable=unused-argument

import numpy as np

from qp.metrics.base_metric_classes import (
    MetricOutputType,
    DistToDistMetric,
    DistToPointMetric,
    SingleEnsembleMetric,
)
from qp.metrics.metrics import (
    calculate_brier,
    calculate_brier_for_accumulation,
    calculate_goodness_of_fit,
    calculate_kld,
    calculate_moment,
    calculate_outlier_rate,
    calculate_rmse,
    calculate_rbpe,
)
from qp.metrics.pit import PIT

from pytdigest import TDigest
from functools import reduce
from operator import add


class DistToPointMetricDigester(DistToPointMetric):

    def __init__(self, tdigest_compression: int = 1000, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tdigest_compression = tdigest_compression

    def initialize(self):
        pass

    def accumulate(self, estimate, reference):  #pragma: no cover
        raise NotImplementedError()

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
        raise NotImplementedError()


class MomentMetric(SingleEnsembleMetric):
    """Class wrapper around the `calculate_moment` function."""

    metric_name = "moment"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(
        self,
        moment_order: int = 1,
        limits: tuple = (0.0, 3.0),
        dx: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(limits, dx)
        self._moment_order = moment_order

    def evaluate(self, estimate) -> list:
        return calculate_moment(estimate, self._moment_order, self._limits, self._dx)


class KLDMetric(DistToDistMetric):
    """Class wrapper around the KLD metric"""

    metric_name = "kld"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01, **kwargs) -> None:
        super().__init__(limits, dx)

    def evaluate(self, estimate, reference) -> list:
        return calculate_kld(estimate, reference, self._limits, self._dx)


class RMSEMetric(DistToDistMetric):
    """Class wrapper around the Root Mean Square Error metric"""

    metric_name = "rmse"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01, **kwargs) -> None:
        super().__init__(limits, dx)

    def evaluate(self, estimate, reference) -> list:
        return calculate_rmse(estimate, reference, self._limits, self._dx)


class RBPEMetric(SingleEnsembleMetric):
    """Class wrapper around the Risk Based Point Estimate metric."""

    metric_name = "rbpe"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(self, limits: tuple = (np.inf, np.inf), **kwargs) -> None:
        super().__init__(limits)

    def evaluate(self, estimate) -> list:
        return calculate_rbpe(estimate, self._limits)


#! Should this be implemented as `DistToPointMetric` or `DistToDistMetric` ???
class BrierMetric(DistToPointMetricDigester):
    """Class wrapper around the calculate_brier function. (Which itself is a
    wrapper around the `Brier` metric evaluator class).
    """

    metric_name = "brier"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01, **kwargs) -> None:
        kwargs.update({"limits": limits, "dx": dx})
        super().__init__(**kwargs)

    def evaluate(self, estimate, reference) -> list:
        return calculate_brier(estimate, reference, self._limits, self._dx)

    def accumulate(self, estimate, reference):
        brier_sum_npdf_tuple = calculate_brier_for_accumulation(estimate, reference, self._limits, self._dx)
        return brier_sum_npdf_tuple

    def finalize(self, tuples):
        # tuples is a list of tuples. The first value in the tuple is the Brier sum
        # The second value is the number of PDFs
        summed_terms = np.sum(np.atleast_2d(tuples), axis=0)

        # calculate the mean from the summed terms
        return summed_terms[0] / summed_terms[1]

class OutlierMetric(SingleEnsembleMetric):
    """Class wrapper around the outlier calculation metric."""

    metric_name = "outlier"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(self, cdf_limits: tuple = (0.0001, 0.9999), **kwargs) -> None:
        super().__init__()
        self._cdf_limits = cdf_limits

    def evaluate(self, estimate) -> list:
        return calculate_outlier_rate(
            estimate, self._cdf_limits[0], self._cdf_limits[1]
        )


class ADMetric(DistToDistMetric):
    """Class wrapper for Anderson Darling metric."""

    metric_name = "ad"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(
        self, num_samples: int = 100, _random_state: float = None, **kwargs
    ) -> None:
        super().__init__()
        self._num_samples = num_samples
        self._random_state = _random_state

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

    def evaluate(self, estimate, reference) -> list:
        return calculate_goodness_of_fit(
            estimate,
            reference,
            fit_metric=self.metric_name,
            num_samples=self._num_samples,
            _random_state=self._random_state,
        )


class CvMMetric(DistToDistMetric):
    """Class wrapper for Cramer von Mises metric."""

    metric_name = "cvm"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(
        self, num_samples: int = 100, _random_state: float = None, **kwargs
    ) -> None:
        super().__init__()
        self._num_samples = num_samples
        self._random_state = _random_state

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

    def evaluate(self, estimate, reference) -> list:
        return calculate_goodness_of_fit(
            estimate,
            reference,
            fit_metric=self.metric_name,
            num_samples=self._num_samples,
            _random_state=self._random_state,
        )


class KSMetric(DistToDistMetric):
    """Class wrapper for Kolmogorov Smirnov metric."""

    metric_name = "ks"
    metric_output_type = MetricOutputType.one_value_per_distribution

    def __init__(
        self, num_samples: int = 100, _random_state: float = None, **kwargs
    ) -> None:
        super().__init__()
        self._num_samples = num_samples
        self._random_state = _random_state

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

    def evaluate(self, estimate, reference) -> list:
        return calculate_goodness_of_fit(
            estimate,
            reference,
            fit_metric=self.metric_name,
            num_samples=self._num_samples,
            _random_state=self._random_state,
        )


class PITMetric(DistToPointMetricDigester):
    """Class wrapper for the PIT Metric class."""

    metric_name = "pit"
    metric_output_type = MetricOutputType.single_distribution
    default_eval_grid = np.linspace(0, 1, 100)

    def __init__(self, eval_grid: list = default_eval_grid, **kwargs) -> None:
        super().__init__(**kwargs)
        self._eval_grid = eval_grid

    def evaluate(self, estimate, reference):
        pit_object = PIT(estimate, reference, self._eval_grid)
        return pit_object.pit

    def accumulate(self, estimate, reference):
        pit_samples = PIT(estimate, reference, self._eval_grid)._gather_pit_samples(estimate, reference)
        digest = TDigest.compute(pit_samples, compression=self._tdigest_compression)
        centroids = digest.get_centroids()
        return centroids

    def compute_from_digest(self, digest):
        # Since we use equal weights for all the values in the digest
        # digest.weight is the total number of values, it is stored as a float,
        # so we cast to int.
        eval_grid = self._eval_grid
        total_samples = int(digest.weight)
        n_pit = np.min([total_samples, len(eval_grid)])
        if n_pit < len(eval_grid):  # pragma: no cover
            #! TODO: Determine what the appropriate style of logging is going to be for metrics.
            print(
                "Number of pit samples is smaller than the evaluation grid size. "
                "Will create a new evaluation grid with size = number of pit samples"
            )
            eval_grid = np.linspace(0, 1, n_pit)

        data_quants = digest.inverse_cdf(eval_grid)
        return PIT._produce_output_ensemble(data_quants, eval_grid)


class CDELossMetric(DistToPointMetricDigester):
    """Conditional density loss"""

    metric_name = "cdeloss"
    metric_output_type = MetricOutputType.single_value
    default_eval_grid = np.linspace(0, 2.5, 301)

    def __init__(self, eval_grid: list = default_eval_grid, **kwargs) -> None:
        super().__init__()
        self._xvals = eval_grid

    def evaluate(self, estimate, reference):
        """Evaluate the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).
        """

        pdfs = estimate.pdf(self._xvals)
        npdf = estimate.npdf

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(pdfs**2, x=self._xvals))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(self._xvals - z)) for z in reference]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(pdfs[range(npdf), nns])
        cdeloss = term1 - 2 * term2
        return cdeloss

    def accumulate(self, estimate, reference):
        pdfs = estimate.pdf(self._xvals)
        npdf = estimate.npdf
        term1_sum = np.sum(np.trapz(pdfs**2, x=self._xvals))

        nns = [np.argmin(np.abs(self._xvals - z)) for z in reference]
        term2_sum = np.sum(pdfs[range(npdf), nns])

        return (term1_sum, term2_sum, npdf)

    def finalize(self, tuples):
        summed_terms = np.sum(np.atleast_2d(tuples), axis=0)
        term1 = summed_terms[0] / summed_terms[2]
        term2 = summed_terms[1] / summed_terms[2]
        return term1 - 2 * term2
