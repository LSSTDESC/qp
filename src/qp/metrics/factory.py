from .base_metric_classes import BaseMetric


def get_all_subclasses(cls):
    """Utility function to recursively get all the subclasses of a class"""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


class MetricFactory:
    metric_dict = {}

    @classmethod
    def update_metrics(cls):
        """Update the dictionary of known metrics"""
        all_subclasses = get_all_subclasses(BaseMetric)
        cls.metric_dict = {
            subcls.metric_name.replace("Metric", ""): subcls
            for subcls in all_subclasses
            if subcls.metric_name
        }

    @classmethod
    def print_metrics(cls, force_update=False):
        """List all the known metrics"""
        if not cls.metric_dict or force_update:
            cls.update_metrics()
        print("List of available metrics")
        for key, val in cls.metric_dict.items():
            print(key, val, val.metric_input_type.name)

    @classmethod
    def list_metrics(cls, force_update=False):
        """Get the list of all the metric names"""
        if not cls.metric_dict or force_update:
            cls.update_metrics()
        return list(cls.metric_dict.keys())

    @classmethod
    def create_metric(cls, name, force_update=False, **kwargs):
        """Create a metric evaluator"""
        if not cls.metric_dict or force_update:
            cls.update_metrics()
        try:
            metric_class = cls.metric_dict[name]
        except KeyError as msg:
            raise KeyError(
                f"{name} is not in the set of known metrics {str(cls.list_metrics())}"
            ) from msg
        return metric_class(**kwargs)
