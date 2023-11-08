import enum

from abc import ABC

class MetricInputType(enum.Enum):
    """Defines the various combinations of input types that metric classes accept.
    """

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
            MetricInputType.point_to_dist,
        ]

    def uses_point_for_reference(self) -> bool:
        return self in [
            MetricInputType.dist_to_point,
            MetricInputType.point_to_point,
        ]


class MetricOuputType(enum.Enum):
    """Defines the various output types that metric classes can return.
    """

    unknown = -1

    # The metric produces a single value for all input
    single_value = 0

    # The metric produces a single distribution for all input
    single_distribution = 1

    # The metric produces a value for each input distribution
    one_value_per_distribution = 2


class BaseMetric(ABC):
    metric_name = None # The name for this metric, overwritten in subclasses
    metric_input_type = MetricInputType.unknown # The type of input data expected for this metric
    metric_output_type = MetricOuputType.unknown # The form of the output data from this metric

    def __init__(self, limit:tuple=(0.0, 3.0), dx:float=0.01) -> None:

        self._limit = limit
        self._dx = dx

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

    metric_input_type = MetricInputType.single_ensemble
    metric_output_type = MetricOuputType.one_value_per_distribution


class DistToDistMetric(BaseMetric):

    metric_input_type = MetricInputType.dist_to_dist


class DistToPointMetric(BaseMetric):

    metric_input_type = MetricInputType.dist_to_point

    def initialize(self, **kwargs):
        raise NotImplementedError()

    def evaluate(self, estimate, reference, **kwargs):
        raise NotImplementedError()

    def finalize(self, **kwargs):
        raise NotImplementedError()


class PointToPointMetric(BaseMetric):

    metric_input_type = MetricInputType.point_to_point


class PointToDistMetric(BaseMetric):

    metric_input_type = MetricInputType.point_to_dist
