from typing import List

class AbstractQuantilePdfConstructor():
    """Abstract class to define an interface for concrete PDF Constructor classes
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

    def prepare_constructor(self) -> None:
        """All the intermediate math for a constructor should happen here.
        This is public so that the user can trigger a recalculation of the of
        variables needed to construct the original PDF.
        This method should either return functions or set variables that will 
        receive x values and return y values.
        """

    def construct_pdf(self, grid: List[float], row: List[int] = None) -> List[List[float]]:
        """This is the method that the user would most often be interacting with by
        passing a grid (set of x values) and optionally a list of indexes for
        for filtering.

        This is also the method that is called by `quant_gen._pdf`.

        Parameters
        ----------
        grid : List[float]
            The input x values used to calculate y values.
        row : List[int], optional
            A list that specifies which elements to calculate values for.
            Passing None will return y values for all distributions, by default None

        Returns
        -------
        List[List[float]]
            A list of lists of floats. Each list of floats has length equal to
            the length of `grid`. There will be N lists returned where N is the number
            of lists in `locations`, or the length of `row`, if `row` is being
            used to filter the output.
        """
