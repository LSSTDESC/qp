import enum

from abc import ABC


class MetricInputType(enum.Enum):
    """Defines the various combinations of input types that metric classes accept."""

    unknown = -1

    # A single qp.Ensemble
    single_ensemble = 0

    # A distribution, or collection of distributions for estimate(s) and a
    # single value, or collection of values for reference(s)
    dist_to_point = 1

    # A distribution, or collection of distributions for estimate(s) and a
    # distribution, or collection of distributions for references(s)
    dist_to_dist = 2

    # A single value, or collection of values for estimate(s) and a
    # single value, or collection of values for reference(s).
    point_to_point = 3

    # A single value, or collection of values for estimate(s) and a
    # distribution, or collection of distributions for reference(s).
    point_to_dist = 4

    def uses_distribution_for_estimate(self) -> bool:
        return self in [
            MetricInputType.single_ensemble,
            MetricInputType.dist_to_point,
            MetricInputType.dist_to_dist,
        ]

    def uses_distribution_for_reference(self) -> bool:
        return self in [
            MetricInputType.dist_to_dist,
            MetricInputType.point_to_dist,
        ]

    def uses_point_for_estimate(self) -> bool:
        return self in [
            MetricInputType.point_to_dist,
            MetricInputType.point_to_point,
        ]

    def uses_point_for_reference(self) -> bool:
        return self in [
            MetricInputType.dist_to_point,
            MetricInputType.point_to_point,
        ]


class MetricOutputType(enum.Enum):
    """Defines the various output types that metric classes can return."""

    unknown = -1

    # The metric produces a single value for all input
    single_value = 0

    # The metric produces a single distribution for all input
    single_distribution = 1

    # The metric produces a value for each input distribution
    one_value_per_distribution = 2


class BaseMetric(ABC):
    """This is the base class for all of the qp metrics. It establishes the most
    of the basic API for a consistent interaction with the metrics qp provides.
    """

    metric_name = None  # The name for this metric, overwritten in subclasses
    metric_input_type = (
        MetricInputType.unknown
    )  # The type of input data expected for this metric
    metric_output_type = (
        MetricOutputType.unknown
    )  # The form of the output data from this metric

    def __init__(self, limits: tuple = (0.0, 3.0), dx: float = 0.01) -> None:
        self._limits = limits
        self._dx = dx

    def initialize(self):
        pass

    def finalize(self):
        pass

    @classmethod
    def uses_distribution_for_estimate(cls) -> bool:
        return cls.metric_input_type.uses_distribution_for_estimate()

    @classmethod
    def uses_distribution_for_reference(cls) -> bool:
        return cls.metric_input_type.uses_distribution_for_reference()

    @classmethod
    def uses_point_for_estimate(cls) -> bool:
        return cls.metric_input_type.uses_point_for_estimate()

    @classmethod
    def uses_point_for_reference(cls) -> bool:
        return cls.metric_input_type.uses_point_for_reference()


class SingleEnsembleMetric(BaseMetric):
    """A base class for metrics that accept only a single ensemble as input."""

    metric_input_type = MetricInputType.single_ensemble
    metric_output_type = MetricOutputType.one_value_per_distribution

    def evaluate(self, estimate):
        raise NotImplementedError()


class DistToDistMetric(BaseMetric):
    """A base class for metrics that requires distributions as input for both the
    estimated and reference values.
    """

    metric_input_type = MetricInputType.dist_to_dist

    def evaluate(self, estimate, reference):
        raise NotImplementedError()


class DistToPointMetric(BaseMetric):
    """A base class for metrics that require a distribution as the estimated
    value and a point estimate as the reference value.
    """

    metric_input_type = MetricInputType.dist_to_point

    def evaluate(self, estimate, reference):
        raise NotImplementedError()

    def initialize(self):  #pragma: no cover
        pass

    def accumulate(self, estimate, reference):  #pragma: no cover
        raise NotImplementedError()

    def finalize(self):  #pragma: no cover
        raise NotImplementedError()

class PointToPointMetric(BaseMetric):
    """A base class for metrics that require a point estimate as input for both
    the estimated and reference values.
    """

    metric_input_type = MetricInputType.point_to_point

    def eval_from_iterator(self, estimate, reference):
        self.initialize()
        for estimate, reference in zip(estimate, reference):
            centroids = self.accumulate(estimate, reference)
        return self.finalize([centroids])

    def evaluate(self, estimate, reference):
        raise NotImplementedError()

    def initialize(self):  #pragma: no cover
        pass

    def accumulate(self, estimate, reference):  #pragma: no cover
        raise NotImplementedError()

    def finalize(self):  #pragma: no cover
        raise NotImplementedError()


class PointToDistMetric(BaseMetric):
    """A base class for metrics that require a point estimate as the estimated
    value and a distribution as the reference value.
    """

    metric_input_type = MetricInputType.point_to_dist

    def evaluate(self, estimate, reference):
        raise NotImplementedError()
