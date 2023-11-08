from qp.metrics.base_metric_classes import MetricOuputType, DistToDistMetric
from qp.metrics.metrics import calculate_kld

class KLDMetric(DistToDistMetric):
    """Class wrapper around the KLD metric
    """

    metric_name = "kld"
    metric_output_type = MetricOuputType.one_value_per_distribution

    def __init__(self, limits:tuple=(0.0, 3.0), dx:float=0.01) -> None:
        super().__init__()
        self._limits = limits
        self._dx = dx

    def initialize(self) -> None:
        pass

    def evaluate(self, estimate, reference) -> None:
        return calculate_kld(estimate, reference, self._limits, self._dx)

    def finalize(self) -> None:
        pass
