import logging
import numpy as np


class Brier:
    """Brier score based on https://en.wikipedia.org/wiki/Brier_score#Original_definition_by_Brier

    Parameters
    ----------
    prediction: NxM array, float
        Predicted probability for N distributions to have a true value in
        one of M bins. The sum of values along each row N should be 1.
    truth: NxM array, int
        True values for N distributions, where Mth bin for the
        true value will have value 1, all other bins will have a value of
        0.
    """

    def __init__(self, prediction, truth):
        """Constructor"""

        self._prediction = prediction
        self._truth = truth
        self._axis_for_summation = None  # axis to sum for metric calculation

    def evaluate(self):
        """Evaluate the Brier score.

        Returns
        -------
        float
            The result of calculating the Brier metric, a value in the interval [0,2]
        """

        self._manipulate_data()
        self._validate_data()
        return self._calculate_metric()

    def accumulate(self):
        self._manipulate_data()
        self._validate_data()
        return self._calculate_metric_for_accumulation()

    def _manipulate_data(self):
        """
        Placeholder for data manipulation as required. i.e. converting from
        qp.ensemble objects into np.array objects.
        """

        # Attempt to convert the input variables into np.arrays
        self._prediction = np.array(self._prediction)
        self._truth = np.array(self._truth)

    def _validate_data(self):
        """
        Strictly for data validation - no calculations or data structure
        changes.

        Raises
        ------
        TypeError if either prediction or truth input could not be converted
        into a numeric Numpy array

        ValueError if the prediction and truth arrays do not have the same
        numpy.shape.

        Warning
        -------
        Logs a warning message if the input predictions do not each sum to 1.
        """

        # Raise TypeError exceptions if the inputs were not translated to
        # numeric np.arrays
        if not np.issubdtype(self._prediction.dtype, np.number):
            raise TypeError(
                "Input prediction array could not be converted to a Numpy array"
            )
        if not np.issubdtype(self._truth.dtype, np.number):
            raise TypeError("Input truth array could not be converted to a Numpy array")

        # Raise ValueError if the arrays have different shapes
        if self._prediction.shape != self._truth.shape:
            raise ValueError(
                "Input prediction and truth arrays do not have the same shape"
            )

        # Log a warning if the N rows of the input prediction do not each sum to
        # 1. Note: For 1d arrays, a sum along axis = 1 will fail, so we set
        # self._axis_for_summation appropriately for that case
        self._axis_for_summation = 0 if self._prediction.ndim == 1 else 1
        if not np.allclose(
            np.sum(self._prediction, axis=self._axis_for_summation), 1.0
        ):
            logging.warning("Input predictions do not sum to 1.")

    def _calculate_metric(self):
        """
        Calculate the Brier metric for the input data.
        """
        return np.mean(
            np.sum((self._prediction - self._truth) ** 2, axis=self._axis_for_summation)
        )

    def _calculate_metric_for_accumulation(self):
        return np.sum(
            np.sum((self._prediction - self._truth) ** 2, axis=self._axis_for_summation)
        )
