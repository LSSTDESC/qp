from typing import List

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from qp.quantile_pdf_constructors.abstract_pdf_constructor import AbstractQuantilePdfConstructor

class CdfSplineDerivative(AbstractQuantilePdfConstructor):
    """Implements an interpolation algorithm based on a list of quantiles and locations.
    First we fit a spline to the (quantile,location) pairs. Then evaluate the derivative
    of the spline. This represents a reconstruction of the original PDF from which the
    quantiles and locations were selected.

    Calling `cdf_spline.interpolate(grid)` will evaluate the spline derivatives at the
    provided grid values.
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

        # A list of interpolation functions (spline derivatives fit to quant,loc pairs)
        self._interpolation_functions = None

    def prepare_constructor(self, spline_order:int = 3) -> None:
        """Calculate the fit spline derivative for each of the original distributions
        This function is the least performant - for reference, on a M1 Mac,
        it requires about 30 seconds to produce an output given
        shape(locations) = (1_000_000, 30).

        Note: we are aware that the edges of the resulting pdf are showing an
        elephant foot.

        Parameters
        ----------
        spline_order : int
            Defines the order of the spline fit, defaults to 4
        """
        number_of_locations = len(self._locations[:,0])

        # ! create an issue (or fix) if the spline fit fails, can fall back to a simpler interpolator ???
        self._interpolation_functions = [
            InterpolatedUnivariateSpline(
                self._locations[i,:], self._quantiles, k=spline_order, ext=1
            ).derivative()
            for i in range(0,number_of_locations)
        ]

    def construct_pdf(self, grid: List[float], row: List[int] = None) -> List[List[float]]:
        """Evaluate the fit spline derivative at each of the grid values

        Parameters
        ----------
        grid : List[float]
            The x values to pass to self._interpolation_functions
        row : List[int], optional
            Defines which interpolation_functions to return values for, by default None

        Returns
        -------
        List[List[float]]
            The lists of y values returned from self._interpolation_functions
        """

        # Generate the fitted spline derivatives if they don't already exist.
        if self._interpolation_functions is None:
            self.prepare_constructor()

        # Support the use of `row` as a filter. If row is None, do nothing,
        # otherwise, return a subset of the rows.
        selected_interpolation_functions = self._interpolation_functions
        if row is not None:
            selected_interpolation_functions = map(self._interpolation_functions.__getitem__, np.unique(row))

        # For each of the fit spline derivative, calculate y value given the grid (or x) values.
        # Note: This implementation uses list comprehension, there might be a faster way.
        # Using np.vectorize was attempted, however:
        # 1) np.vectorize doesn't work well for a list of functions (i.e. self._interpolation_functions)
        # 2) np.vectorize doesn't provide any any performance improvements over list comprehension
        return np.asarray([func(grid) for func in selected_interpolation_functions])

    def debug(self):
        """This is a debugging utility that is meant to return intermediate calculation values
        to make it easier to visualize and debug the reconstruction algorithm.

        Returns
        -------
            _quantiles :
                Input during constructor instantiation
            _locations :
                Input during constructor instantiation
            _interpolation_functions :
                The list of analytic derivatives of splines fit to the input data
        """
        return self._quantiles, self._locations, self._interpolation_functions
