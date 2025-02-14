"""This is a template that developers can copy for creating new parameterizations"""

import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional

# these imports will work when this file is placed in a folder in the parameterizations folder
from ..base import Pdf_rows_gen
from ...core.factory import add_class
from ...core.ensemble import Ensemble


class parameterization_gen(Pdf_rows_gen):
    """[Description goes here]


    [Describe arguments and their structure]

    Notes
    -----

    [any notes/caveats on how specific distributions are calculated]

    """

    # pylint: disable=protected-access

    name = "parameterization"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, arg1: np.array, arg2: np.array, *args, **kwargs):
        """
        Create a new distribution using the given data. [details]

        Parameters
        ----------
        arg1 : array_like
          [description]

        arg2 : array_like
          [description]
        """

        # this should be included at some point in the initialization to check if the input data is normalized
        check_input = kwargs.pop("check_input", True)
        if check_input:
            # normalize the distribution here
            pass
        else:
            # skip normalization
            pass

        # initialize the data
        self._arg1 = arg1
        self._arg2 = arg2

        # Get the shape of the data
        # and pass it to the base constructor to set up other attributes
        kwargs["shape"] = arg2.shape[:-1]
        super().__init__(*args, **kwargs)

        # define data and metadata
        # metadata is shared across all distributions
        # data is quantities defined for each distribution
        self._addmetadata("arg1", self._arg1)
        self._addobjdata("arg2", self._arg2)

    @property
    def arg1(self):  # property that allows access to the 'metadata' field
        """Return arg1"""
        return self._arg1

    @property
    def arg2(self):  # property that allows access to the 'data' field
        """Return arg2"""
        return self._arg2

    # TODO: add expected return format?
    def _pdf(self, x, row):
        # pylint: disable=arguments-differ

        # put here functionality to evalulate the pdf
        return function_to_evaluate_pdf.ravel()

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ

        # put here functionality to evaluate the cdf
        return function_to_evaluate_cdf.ravel()

    def _updated_ctor_param(self) -> Mapping:
        """
        Sets the arguments as additional constructor arguments. This function is needed
        by scipy in order to copy distributions, and makes a dictionary of all parameters
        necessary to construct the distribution.
        """
        dct = super()._updated_ctor_param()
        dct["arg1"] = self._arg1
        dct["arg2"] = self._arg2
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Adds this class' mappings to the conversion dictionary. Specifically, this should include at
        least a creation method and a function to extract the necessary values from the distribution
        to provide to the creation method.

        """
        # ---------------------------------------------------------------------
        # Add the creation function
        # ---------------------------------------------------------------------
        # This should always be cls.create, and with a key of `None` to make it
        # the default.
        cls._add_creation_method(cls.create, None)

        # You may add additional creation methods here, but they need to have a
        # specific key so they can be referred to when converting
        #
        # Uncomment the line below to add additional creation method:
        # cls._add_creation_method(creation_method_function, "method_key")

        # ---------------------------------------------------------------------
        # Add the extraction function(s)
        # ---------------------------------------------------------------------
        #
        # At least one extraction method is required
        # The key for this extraction method should be `None`
        # to make it the default. To add an extraction method,
        # uncomment the line of code below and change `func_to_convert_default`
        # to a useable function, either in parameterization_utils.py, or
        # in utils.conversion.py

        # Line to uncomment:
        # cls._add_extraction_method(func_to_convert_default, None)

        # You can optionally provide additional conversion methods
        # These need to have a specific key so the user can refer to
        # them when converting. Uncomment the line below and change the
        # function and key to appropriate values

        # Line to uncomment:
        # cls._add_extraction_method(other_func_to_convert, "method_key")

    @classmethod
    def get_allocation_kwds(cls, npdf: int, **kwargs) -> Mapping:
        """Return the kwds necessary to create an `empty` HDF5 file with ``npdf`` entries
        for iterative write. We only need to allocate the data columns, as
        the metadata will be written when we finalize the file.

        The number of data columns is calculated based on the length or
        shape of the metadata. For example, the number of columns is ``nbins-1``
        for a histogram.

        Parameters
        ----------
        npdf : int
            number of *total* distributions that will be written out
        kwargs :
            The keys needed to construct the shape of the data to be written.

        Returns
        -------
        Mapping
            A dictionary with a key for the objdata, a tuple with the shape of that data,
            and the data type of the data.

        Raises
        ------
        ValueError
            Raises an error if the required kwarg is not provided.
        """

        if "arg1" not in kwargs:
            raise ValueError("required argument 'arg1' not included in kwargs")
        narg1 = len(kwargs["arg1"].flatten())
        return dict(pdfs=((npdf, narg1), "f4"))

    @classmethod
    def create_ensemble(
        self, data: Mapping, ancil: Optional[Mapping] = None
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as [parameterization type].

        Input data format:
        data = {'arg1': array_like, 'arg2': array_like}, where args are the necessary inputs for the parameterization.


        Parameters
        ----------
        data : Mapping
            The dictionary of data for the distributions.
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have the same length as the number of distributions, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Example
        -------

        To create an Ensemble with two distributions and an 'ancil' table that provides ids for the distributions, you can use the following code:

        >>> import qp
        >>> import numpy as np
        >>> data = {test data}
        >>> ancil = {'ids': test ids }
        >>> ens = qp.[parameterization].create_ensemble(data,ancil)
        >>> ens.metadata()
        [output here]

        """

        return Ensemble(self, data, ancil)

    #
    # Optional methods
    #

    def _ppf():
        pass

    def _sf():
        pass

    def _isf():
        pass

    def _rvs():
        pass

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a histogram this shows the bin edges
        """
        # add any plotting functions you create for this to plotting.py
        pass


parameterization = (
    parameterization_gen.create
)  # alias the class to just the parameterization name
add_class(parameterization_gen)  # register the class with the factory
