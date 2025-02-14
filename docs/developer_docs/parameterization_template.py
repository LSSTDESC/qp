"""This is a template that developers can copy for creating new parameterizations"""

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

    def __init__(self, arg1, arg2, *args, **kwargs):
        """
        Create a new distribution using the given data

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

        # some code to initialize the data
        self._arg1 = arg1
        self._arg2 = arg2

        # Get the number of distributions, or shape of the data
        # and pass it to the base constructor to set up other attributes
        kwargs["shape"] = arg2.shape[:-1]
        super().__init__(*args, **kwargs)

        # define data and metadata
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

    def _pdf(self, x, row):
        """Function to evaluate the pdf"""
        # pylint: disable=arguments-differ
        return function_to_evaluate_pdf.ravel()

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        return function_to_evaluate_cdf.ravel()

    def _updated_ctor_param(self):
        """
        Set the arguments as additional constructor arguments.
        """
        dct = super()._updated_ctor_param()
        dct["arg1"] = self._arg1
        dct["arg2"] = self._arg2
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        # since the key given is None, this will be the default conversion function
        cls._add_extraction_method(func_to_convert_default, None)
        # You can optionally provide additional conversion methods
        # if you do, they need to have a specific key
        cls._add_extraction_method(other_func_to_convert, "samples")

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        """Return kwds necessary to create 'empty' hdf5 file with npdf entries
        for iterative writeout.  We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.
        """
        # TODO: I'm not actually sure this documentation is correct -- it looks like they're requiring the metadata but not the objdata
        if "arg1" not in kwargs:  # pragma: no cover
            raise ValueError("required argument 'arg1' not included in kwargs")
        narg1 = len(kwargs["arg1"].flatten())
        return dict(pdfs=((npdf, nbins - 1), "f4"))

    @classmethod
    def create_ensemble(data: Mapping, ancil: Optional[Mapping] = None) -> Ensemble:
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
        >>> ens = qp.[parameterization]_ensemble(data,ancil)
        >>> ens.metadata()
        [output here]

        """

        return Ensemble(parameterization, data, ancil)

    #
    # Optional methods
    #

    def _ppf():
        pass

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a histogram this shows the bin edges
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        vals = pdf.dist.pdfs[pdf.kwds["row"]]
        return plot_pdf_histogram_on_axes(axes, hist=(pdf.dist.bins, vals), **kw)

    @classmethod
    def make_test_data(cls):
        """Make data for unit tests"""
        cls.test_data = dict(
            hist=dict(
                gen_func=hist,
                ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),
                convert_data=dict(bins=XBINS),
                atol_diff=1e-1,
                atol_diff2=1e-1,
                test_xvals=TEST_XVALS,
            ),
            hist_samples=dict(
                gen_func=hist,
                ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),
                convert_data=dict(bins=XBINS, method="samples", size=NSAMPLES),
                atol_diff=1e-1,
                atol_diff2=1e-1,
                test_xvals=TEST_XVALS,
                do_samples=True,
            ),
        )


parameterization = parameterization_gen.create
add_class(parameterization_gen)
