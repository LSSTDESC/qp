from typing import List

import numpy as np
from scipy.interpolate import interp1d

from qp.quantile_pdf_constructors.abstract_pdf_constructor import AbstractQuantilePdfConstructor

class DualSplineAverage(AbstractQuantilePdfConstructor):
    """Implementation of the "area-under-the-curve" using the average of
    the bounding splines fit to the CDF derivative.

    By using the difference between quantiles to solve for the area under the PDF, we can
    create an approximation of the original PDF. However, because we use a piecewise linear
    approximation for the continuous PDF, our approximated p(z) values will always be different
    that the original distribution. In practice they typically oscillate above and below the
    original curve as each calculation attempts to correct for over or undershooting of the prior
    calculation.

    If we fit two splines, one to the odd and one to the even approximated points, then take the
    average, the resulting average of those splines tend to fit the original distribution well.

    This constructor implements that algorithmic approach.
    """

    def __init__(self, quantiles:List[float], locations: List[List[float]]) -> None:
        """Constructor to instantiate this class.

        Parameters
        ----------
        quantiles : List[float]
            List of n quantile values in the range (0,1).
        locations : List[List[float]]
            List of m Lists, each containing n values corresponding to the
            y-value of the PPF function at the same quantile index.
        """
        self._quantiles = quantiles
        self._locations = np.atleast_2d(locations)

        self._p_of_zs = None
        self.y1 = None
        self.y2 = None

    def prepare_constructor(self) -> None:
        """This method solves for the area under the PDF via a stepwise algorithm.
        Given that the difference between any two quantile values is equal to the
        area under the PDF between the corresponding pair of locations, _and_ given
        that we know the p(z) value at 1 of those locations, we can solve
        for the unknown p(z) value at the other location.

        We approximate the area under the curve as a trapezoid with the following area:
        (q_i+1 - q_i) = (loc_i+1 - loc_i) * p(z_i) + (1/2) * (loc_i+1 - loc_i) * p(z_i+1 = p(z_i))

        Solving for p(z_i+1), we have:
        p(z_i+1) = [2 * (q_i+1 - q_i) / (loc_i+1 - loc_i)] - p(z_i)

        The first term in this equation is calculated as `first_term`. After that we step along
        all distributions simultaneously for each location, using the previous p(z) value to
        calculate the next.
        """

        # Prepare an empty container for the output
        self._p_of_zs = np.zeros(self._locations.shape)

        # Prepare an all-zero list, used in the step-wise calculation to prevent negative values
        zeros_for_comparison = np.zeros(self._locations.shape[0])

        # Calculate the first term
        first_term = 2 * np.diff(self._quantiles) / np.diff(self._locations)

        # Perform the step-wise calculation for all distributions simultaneously
        for i in range(1, np.shape(self._p_of_zs)[-1]):
            self._p_of_zs[:,i] = np.maximum(zeros_for_comparison, first_term[:,i-1] - self._p_of_zs[:,i-1])

        # Set any negative values to 0.
        self._p_of_zs = np.maximum(np.zeros(self._locations.shape), self._p_of_zs)

    def construct_pdf(self, grid: List[float], row: List[int] = None) -> List[List[float]]:
        """This method utilizes intermediate calculations from `prepare_constructor`
        along with the provided grid (i.e. x) values to return corresponding y values
        to construct the PDF approximation.

        Parameters
        ----------
        grid : List[float]
            x values used to calculate corresponding y values
        row : List[int], optional
            A list of indexes of the original distribution to return, used as a filter.
            By default None will do no filtering.

        Returns
        -------
        List[List[float]]
            The lists of y values returned from self._interpolation_functions
        """
        if self._p_of_zs is None:
            self.prepare_constructor()

        # Support the use of `row` as a filter. If row is None, do nothing,
        # otherwise, return a subset of the rows.
        # Using `map` alone will return an iterator that will be completely consumed after the first
        # list comprehension. Thus we convert the map to a list so that it can be used multiple times.
        filtered_p_of_zs = self._p_of_zs
        filtered_locations = self._locations
        if row is not None:
            filtered_p_of_zs = list(map(self._p_of_zs.__getitem__, np.unique(row)))
            filtered_locations = list(map(self._locations.__getitem__, np.unique(row)))

        # Create a list of interpolated splines for the even and odd pairs of
        # (specific_locations, specific_p_of_zs)
        f1 = np.asarray(
            [
                interp1d(
                    np.squeeze(specific_locations[0::2]),
                    np.squeeze(specific_p_of_zs[0::2]),
                    bounds_error=False, fill_value=0.0, kind='cubic'
                )
                for specific_p_of_zs, specific_locations in zip(filtered_p_of_zs, filtered_locations)
            ]
        )

        f2 = np.asarray(
            [
                interp1d(
                    np.squeeze(specific_locations[1::2]),
                    np.squeeze(specific_p_of_zs[1::2]),
                    bounds_error=False, fill_value=0.0, kind='cubic'
                )
                for specific_p_of_zs, specific_locations in zip(filtered_p_of_zs, filtered_locations)
            ]
        )

        # Evaluate all the splines at the input grid values
        self.y1 = np.asarray([func_1(grid) for func_1 in f1])
        self.y2 = np.asarray([func_2(grid) for func_2 in f2])

        # Return the average of the spline values at each of the evaluated points.
        return (self.y1 + self.y2) / 2

    def debug(self):
        """This is a debugging utility that is meant to return intermediate calculation values
        to make it easier to visualize and debug the reconstruction algorithm.

        Returns
        -------
            _quantiles :
                Input during constructor instantiation
            _locations :
                Input during constructor instantiation
            _p_of_zs :
                Resulting p(z) values found after calculating the area of trapezoids based
                on the difference between adjacent quantile values
            y1 :
                One of two splines fit to alternating pairs of (_locations, _p_of_zs)
            y2 :
                One of two splines fit to alternating pairs of (_locations, _p_of_zs)
        """
        return self._quantiles, self._locations, self._p_of_zs, self.y1, self.y2
