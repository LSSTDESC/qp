"""This is a template that developers can copy for creating new parameterizations"""

from __future__ import annotations
import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike
import warnings

# these imports will work when this file is placed in a folder in the parameterizations folder
from ..base import Pdf_rows_gen
from ...core.factory import add_class
from ...core.ensemble import Ensemble


# ---------------------------------------------------------------------
# How to use this template
# ---------------------------------------------------------------------
#
# This template is meant to provide you with a starting point when
# creating a new parameterization. In this template, square brackets like these
# [] indicate text that should be replaced with the relevant text for this
# parameterization. For example, [parameterization] indicates that you should
# replace the entire square brackets and text within with the name of the
# parameterization you are creating.
#
# The uncommented code provided within this template should
# not need to be altered except as directed in comments above the code,
# usually just indicating to replace variable names as necessary. The
# template also provides some code lines that can be uncommented and altered
# as instructed.
#
# Start by copying this template to a new folder named after your new
# parameterization type in the parameterizations folder. Within this folder,
# you should have the following files at a minimum:
# - __init__.py
# - [paramterization].py file
# - [parameterization]_utils.py


# -> Replace `parameterization` in the class name with your chosen parameterization name
#    and fill out the relevant information in the class docstring below
class parameterization_gen(Pdf_rows_gen):
    """[One sentence description goes here]

    [ any further description needed here]

    By default, the input distribution is normalized. If the input data is
    already normalized, you can use the optional parameter ``norm = False``
    to skip the normalization process.

    Parameters
    ----------

    Attributes
    ----------
    [arg1] : [dtype]
        [description]
    [arg2] : [dtype]
        [description]
    [warn] : bool, optional
            If True, raises warnings if input is not valid input data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
    [[norm] : bool, optional
        If True, normalizes the input distribution. If False, assumes the
        given distribution is already normalized. By default True.]

    Methods
    -------
    create_ensemble(data,ancil)
        Create an ensemble with this parameterization.
    [add any additional optional methods here, i.e. plot_native]

    Notes
    -----

    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +----------+-----------+------------+
    | Function | Arguments | Method key |
    +----------+-----------+------------+
    |          |           |            |
    +----------+-----------+------------+

    Implementation notes:

    [any notes/caveats on how specific distributions are calculated]

    """

    # pylint: disable=protected-access

    name = "[parameterization]"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self, arg1: ArrayLike, arg2: ArrayLike, warn: bool = True, *args, **kwargs
    ):
        """
        Create a new distribution using the given data. [details]

        Parameters
        ----------
        [arg1] : ArrayLike
          [description]
        [arg2] : ArrayLike
          [description]
        warn : bool, optional
            If True, raises warnings if input is not valid input data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        [norm : bool, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.]
        """

        # ---------------------------------------------------------------------
        # The init function
        # ---------------------------------------------------------------------
        # -> Replace `arg1` and `arg2` in this function (and those throughout)
        #    with appropriate variable names for your coordinates and data arguments.
        #    Additional arguments can be added as necessary.
        # -> norm is an optional argument that you can include if the input data
        #    for this parameterization is a PDF (i.e. if it's normalizable.) If
        #    you include this parameter, add it to the init arguments and uncomment
        #    the optional normalize function below.

        # initialize the data
        self._arg1 = arg1
        self._arg2 = arg2

        # Set warn value and test for warnings as necessary
        self._warn = warn
        # -> Change relevant values below and add any additional warnings as necessary
        if self._warn:
            if not np.all(np.isfinite(self._arg1)):
                warnings.warn(
                    f"The given [arg1] contain non-finite values - {self._arg1}",
                    RuntimeWarning,
                )
            if not np.all(np.isfinite(self._arg2)):
                indices = np.where(np.isfinite(self._arg2) != True)
                warnings.warn(
                    f"There are non-finite values in [arg2] for the distributions: {indices[0]}",
                    RuntimeWarning,
                )

        # -> Uncomment code below to check if the input data is normalized
        # self._norm = norm
        # if self._norm:
        #     self._arg2 = self.normalize()

        # Get the shape of the data
        # and pass it to the base constructor to set up other attributes
        kwargs["shape"] = self._arg2.shape
        super().__init__(*args, **kwargs)

        # define data and metadata
        # metadata is shared across all distributions
        # data is quantities defined for each distribution
        self._addmetadata("arg1", self._arg1)
        self._addobjdata("arg2", self._arg2)

    # ---------------------------------------------------------------------
    # Functions that allow access to the data as attributes
    # ---------------------------------------------------------------------
    # Here we have properties for the two arguments, `arg1` and `arg2` that
    # we created. Add more functions if your parameterization has more
    # arguments.

    # property that allows access to the 'metadata' field
    # -> Replace `arg1` with the appropriate metadata variable
    @property
    def arg1(self):
        """Return arg1"""
        return self._arg1

    # property that allows access to the 'data' field
    # -> Replace `arg2` with the appropriate data variable
    @property
    def arg2(self):
        """Return arg2"""
        return self._arg2

    def x_samples(self):
        """Return a set of x values that can be used to plot all the [distributions]."""
        # This function is meant to return a set of x values that can be used to calculate
        # either the PDF or CDF, whichever is the characteristic distribution of
        # this parameterization. This means that the x values should allow you to plot
        # the data in the parameterization without missing anything due to too
        # large spacing between x values, for example.

        # -> Add in functionality to return an array of x values here
        return

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ

        # ---------------------------------------------------------------------
        # The pdf evaluation function
        # ---------------------------------------------------------------------
        # This typically takes the form of a function from the
        # [paramterization]_utils.py file, where the output
        # is typically a 2D array of shape (npdf, n)
        # .ravel() is necessary to ensure that the array is 1D
        # when passed into scipy's rv_continuous pdf function

        # -> Uncomment the line below and replace the function with
        #    the relevant function and arguments, then delete the additional
        #    return statement
        #
        # return function_to_evaluate_pdf.ravel()
        return

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ

        # ---------------------------------------------------------------------
        # The cdf evaluation function
        # ---------------------------------------------------------------------
        # This typically takes the form of a function from the
        # [parameterization]_utils.py file, where the output
        # is typically a 2D array of shape (npdf, n)
        # .ravel() is necessary to ensure that the array is 1D
        # when passed into scipy's rv_continuous cdf function

        # -> Uncomment the line below and replace the function with
        #    the relevant function and arguments, then delete the additional
        #    return statement
        #
        # return function_to_evaluate_cdf.ravel()
        return

    def _updated_ctor_param(self) -> Mapping:
        """
        Sets the arguments as additional constructor arguments. This function is needed
        by scipy in order to copy distributions, and makes a dictionary of all parameters
        necessary to construct the distribution.
        """
        dct = super()._updated_ctor_param()

        # -> replace `arg1` and `arg2` with your arguments and add any additional ones necessary
        dct["arg1"] = self._arg1
        dct["arg2"] = self._arg2
        dct["warn"] = self._warn
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Adds this class' mappings to the conversion dictionary. Specifically, this should include at
        least a creation method and a function to extract the necessary values from the distribution
        to provide to the creation method. Together these functions should allow conversion from any
        other parameterization to this one.

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
        # -> Uncomment the line below to add additional creation method:
        #    cls._add_creation_method(creation_method_function, "method_key")

        # ---------------------------------------------------------------------
        # Add the extraction function(s)
        # ---------------------------------------------------------------------
        #
        # At least one extraction method is required. The key for this
        # extraction method should be `None` to make it the default.

        # -> To add an extraction method, uncomment the line of code below
        #    and change `func_to_convert_default` to a useable function,
        #    either in parameterization_utils.py, or in utils.conversion.py
        #
        # cls._add_extraction_method(func_to_convert_default, None)

        # You can optionally provide additional conversion methods
        # These need to have a specific key so the user can refer to
        # them when converting.
        # -> Uncomment the line below and change the function and key
        #    to appropriate values
        #
        # cls._add_extraction_method(other_func_to_convert, "method_key")

    @classmethod
    def get_allocation_kwds(cls, npdf: int, **kwargs) -> Mapping:
        """Return the kwds necessary to create an `empty` HDF5 file with ``npdf`` entries
        for iterative write. We only need to allocate the data columns, as
        the metadata will be written when we finalize the file.

        The number of data columns is calculated based on the length or shape of the
        metadata, ``n``. For example, the number of columns is ``nbins-1``
        for a histogram.

        Parameters
        ----------
        npdf : int
            Total number of distributions that will be written out
        kwargs :
            The keys needed to construct the shape of the data to be written.

        Returns
        -------
        Mapping
            A dictionary with a key for the objdata, a tuple with the shape of that data,
            and the data type of the data as a string.
            i.e. ``{objdata_key = ( (npdf, n), "f4" )}``

        Raises
        ------
        ValueError
            Raises an error if the required kwarg is not provided.
        """

        # Checks that the required keyword is present in the function call
        # -> In the lines below fill out `arg1` with the appropriate
        #    metadata key
        if "arg1" not in kwargs:
            raise ValueError("required argument 'arg1' not included in kwargs")

        # Gets the shape of the coordinates/metadata. The line below is a
        # sample of how to do it, but you may need to modify as necessary.
        # -> Replace `arg1` and rename `narg1` below
        narg1 = len(kwargs["arg1"].flatten())
        return dict(pdfs=((npdf, narg1), "f4"))

    @classmethod
    def create_ensemble(
        self,
        arg1: ArrayLike,
        arg2: ArrayLike,
        warn: bool = True,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as [parameterization].


        Parameters
        ----------
        [arg1] : ArrayLike
          [description]
        [arg2] : ArrayLike
          [description]
        [warn] : bool, optional
            If True, raises warnings if input is not valid input data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have the
            same length as the number of distributions, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Examples
        --------

        To create an Ensemble with two distributions and an 'ancil' table that provides ids for the distributions, you can use the following code:

        >>> import qp
        >>> ancil = {'ids': [test ids] }
        >>> ens = qp.[parameterization].create_ensemble(arg1, arg2,ancil=ancil)
        >>> ens.metadata()
        [output here]

        """

        # pack up the data in to a dictionary to be passed to Ensemble
        # -> add norm below, in the docstring, and in the function call if your
        #    parameterization has a normalization function
        data = {"arg1": arg1, "arg2": arg2, "warn": warn}
        return Ensemble(self, data, ancil)

    # ---------------------------------------------------------------------
    # Optional normalization method
    # ---------------------------------------------------------------------
    # This optional method is meant to be used to normalize the distribution
    # data both when initializing the Ensemble and after it has been initialized.
    # -> If your parameterization input data can be normalized, uncomment the
    #    block of code below and add code to normalize the data (in this template
    #    arg2).
    # def normalize(self):

    #     # -> add some operations on object data to normalize here
    #     normed = self._arg2
    #     return normed

    # ---------------------------------------------------------------------
    # Optional methods
    # ---------------------------------------------------------------------
    # These are methods that will be calculated using scipy's underlying
    # functions, if no method is included for them here. You should only
    # add a method for one of these functions if it will speed up the
    # calculations or otherwise improve upon scipy's functionality.

    # -> Uncomment any method below if using

    # def _ppf():
    #     Keep in mind that by default ppf(0) and ppf(1) will not call this function
    #     and will instead return negative and positive infinity, respectively.
    #     pass

    # def _sf():
    #     pass

    # def _isf():
    #     pass

    # def _rvs():
    #     pass

    # ---------------------------------------------------------------------
    # Optional plotting method
    # ---------------------------------------------------------------------
    # Set a native plotting method that allows users to quickly plot PDFs
    # of your parameterization.
    # -> Uncomment the method below if used

    # @classmethod
    # def plot_native(cls, pdf, **kwargs):
    #     """Plot the PDF in a way that is particular to this type of distribution.
    #     [specific description of the plot for this parameterization]
    #
    #     Parameters
    #     ----------
    #     """
    #     # -> Add any plotting functions you create for this to the
    #     #    `plotting.py`` file in the `utils`` folder.

    #     # You should return the axes with the plot already created on them
    #     # as the return statement of the function.
    #     pass


# ---------------------------------------------------------------------
# Set alias and add class to factory
# ---------------------------------------------------------------------
# -> Replace `parameterization` in the lines below with the name of your parameterization
parameterization = parameterization_gen

# Register the class with the factory
# -> Replace `parameterization` with the name of your parameterization
add_class(parameterization_gen)
