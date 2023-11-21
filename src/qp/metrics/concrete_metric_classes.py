# pylint: disable=unused-argument

import numpy as np

from qp.ensemble import Ensemble
from qp.metrics.base_metric_classes import (
    MetricOuputType,
    DistToDistMetric,
    DistToPointMetric,
    SingleEnsembleMetric,
)
from qp.metrics.metrics import (
    calculate_brier,
    calculate_goodness_of_fit,
    calculate_kld,
    calculate_moment,
    calculate_outlier_rate,
    calculate_rmse,
    calculate_rbpe,
)
from qp.metrics.pit import PIT


class MomentMetric(SingleEnsembleMetric):
    """Class wrapper around the `calculate_moment` function."""

    metric_name = "moment"

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
    metric_output_type = MetricOuputType.one_value_per_distribution

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01, **kwargs) -> None:
        super().__init__(limits, dx)

    def evaluate(self, estimate, reference) -> list:
        return calculate_kld(estimate, reference, self._limits, self._dx)


class RMSEMetric(DistToDistMetric):
    """Class wrapper around the Root Mean Square Error metric"""

    metric_name = "rmse"
    metric_output_type = MetricOuputType.one_value_per_distribution

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01, **kwargs) -> None:
        super().__init__(limits, dx)

    def evaluate(self, estimate, reference) -> list:
        return calculate_rmse(estimate, reference, self._limits, self._dx)


class RBPEMetric(SingleEnsembleMetric):
    """Class wrapper around the Risk Based Point Estimate metric."""

    metric_name = "rbpe"
    metric_output_type = MetricOuputType.one_value_per_distribution

    def __init__(self, limits: tuple = (np.inf, np.inf), **kwargs) -> None:
        super().__init__(limits)

    def evaluate(self, estimate) -> list:
        return calculate_rbpe(estimate, self._limits)


#! Should this be implemented as `DistToPointMetric` or `DistToDistMetric` ???
class BrierMetric(DistToPointMetric):
    """Class wrapper around the calculate_brier function. (Which itself is a
    wrapper around the `Brier` metric evaluator class).
    """

    metric_name = "brier"
    metric_output_type = MetricOuputType.one_value_per_distribution

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01, **kwargs) -> None:
        super().__init__(limits, dx)

    def evaluate(self, estimate, reference) -> list:
        return calculate_brier(estimate, reference, self._limits, self._dx)


class OutlierMetric(SingleEnsembleMetric):
    """Class wrapper around the outlier calculation metric."""

    metric_name = "outlier"
    metric_output_type = MetricOuputType.one_value_per_distribution

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
    metric_output_type = MetricOuputType.one_value_per_distribution

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
    metric_output_type = MetricOuputType.one_value_per_distribution

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
    metric_output_type = MetricOuputType.one_value_per_distribution

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


#! Confirm metric output type - perhaps a new type is appropriate ???
class PITMetric(DistToPointMetric):
    """Class wrapper for the PIT Metric class."""

    metric_name = "pit"
    metric_output_type = MetricOuputType.single_distribution
    default_eval_grid = np.linspace(0, 1, 100)

    def __init__(self, eval_grid: list = default_eval_grid) -> None:
        super().__init__()
        self._eval_grid = eval_grid

    def evaluate(self, estimate, reference) -> Ensemble:
        pit_object = PIT(estimate, reference, self._eval_grid)
        return pit_object.pit
