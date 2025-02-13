"""This module implements a PDT distribution sub-class using interpolated quantiles
"""

import logging
import sys

import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional

from .quant_utils import extract_quantiles, pad_quantiles
from ...core.factory import add_class
from ...core.ensemble import Ensemble
from ..base import Pdf_rows_gen
from ...plotting import get_axes_and_xlims, plot_pdf_quantiles_on_axes
from . import (
    AbstractQuantilePdfConstructor,
    CdfSplineDerivative,
    DualSplineAverage,
    PiecewiseConstant,
    PiecewiseLinear,
)
from ...test_data import QLOCS, QUANTS, TEST_XVALS
from ...utils.array import reshape_to_pdf_size
from ...utils.interpolation import interpolate_multi_x_y, interpolate_x_multi_y

epsilon = sys.float_info.epsilon


DEFAULT_PDF_CONSTRUCTOR = "piecewise_linear"
PDF_CONSTRUCTORS = {
    "cdf_spline_derivative": CdfSplineDerivative,
    "dual_spline_average": DualSplineAverage,
    "piecewise_linear": PiecewiseLinear,
    "piecewise_constant": PiecewiseConstant,
}


def quant_ensemble(data: Mapping, ancil: Optional[Mapping] = None) -> Ensemble:
    """Creates an Ensemble of distributions parameterized as quantiles.

    Input data format:
    data = {`quants`: values, `locs`: values}
    The shape of quants should be n, where n is the number of quants. The shape of locs should be (npdfs, n), where npdfs is the number of distributions.
    If you would like to use a constructor function other than the default, `piecewise_linear`,
    you can include {`pdf_constructor_name`: value} in the data dictionary, where value is the string name of the constructor.

    The options are: `piecewise_linear`, `piecewise_constant`, `dual_spline_average` and 'cdf_spline_derivative`.


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

    To create an Ensemble with two distributions and associated ids, using the `dual_spline_average` constructor:

    >>> import qp
    >>> import numpy as np
    >>> data = {'quants': np.array([0.0001,0.25,0.5,0.75,0.9999]), 'locs': np.array([[0.0001,0.1,0.3,0.5,0.75],[0.01,0.05,0.15,0.3,0.5]]),'pdf_constructor_name':'dual_spline_average'}
    >>> ancil = {'ids':[11,18]}
    >>> ens = qp.quant_ensemble(data,ancil)
    >>> ens.metadata()
    {'pdf_name': array([b'quant'], dtype='|S5'),
     'pdf_version': array([0]),
     'quants': array([[0.000e+00, 1.000e-04, 2.500e-01, 5.000e-01, 7.500e-01, 9.999e-01,
             1.000e+00]]),
     'pdf_constructor_name': array(['dual_spline_average'], dtype='<U19'),
     'check_input': array([ True])}
    """

    return Ensemble(quant, data, ancil)


class quant_gen(Pdf_rows_gen):  # pylint: disable=too-many-instance-attributes
    """Quantile based distribution, where the PDF is defined piecewise from the quantiles

    Notes
    -----
    This implements a CDF by interpolating a set of quantile values

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the CDF
    """

    # pylint: disable=protected-access

    name = "quant"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, quants, locs, *args, **kwargs):
        """
        Create a new distribution using the given values

        Parameters
        ----------
        quants : array_like
           The quantiles used to build the CDF
        locs : array_like
           The locations at which those quantiles are reached
        """

        self._xmin = np.min(locs)
        self._xmax = np.max(locs)

        locs_2d = reshape_to_pdf_size(locs, -1)
        self._check_input = kwargs.pop("check_input", True)
        if self._check_input:
            quants, locs_2d = pad_quantiles(quants, locs_2d)

        self._quants = np.asarray(quants)
        self._nquants = self._quants.size
        if locs_2d.shape[-1] != self._nquants:  # pragma: no cover
            raise ValueError(
                "Number of locations (%i) != number of quantile values (%i)"
                % (self._nquants, locs_2d.shape[-1])
            )
        self._locs = locs_2d

        self._pdf_constructor_name = str(
            kwargs.pop("pdf_constructor_name", DEFAULT_PDF_CONSTRUCTOR)
        )
        self._pdf_constructor = None
        self._instantiate_pdf_constructor()

        kwargs["shape"] = locs.shape[:-1]
        super().__init__(*args, **kwargs)

        self._addmetadata("quants", self._quants)
        self._addmetadata("pdf_constructor_name", self._pdf_constructor_name)
        self._addmetadata("check_input", self._check_input)
        self._addobjdata("locs", self._locs)

    @property
    def quants(self):
        """Return quantiles used to build the CDF"""
        return self._quants

    @property
    def locs(self):
        """Return the locations at which those quantiles are reached"""
        return self._locs

    @property
    def pdf_constructor_name(self):
        """Returns the name of the current pdf constructor. Matches a key in
        the PDF_CONSTRUCTORS dictionary."""
        return self._pdf_constructor_name

    @pdf_constructor_name.setter
    def pdf_constructor_name(self, value: str):
        """Allows users to specify a different interpolator without having to recreate
        the ensemble.

        Parameters
        ----------
        value : str
            One of the supported interpolators. See PDF_CONSTRUCTORS
            dictionary for supported interpolators.

        Raises
        ------
        ValueError
            If the value provided isn't a key in PDF_CONSTRUCTORS, raise
            a value error.
        """
        if value not in PDF_CONSTRUCTORS:
            raise ValueError(
                f"Unknown interpolator provided: '{value}'. Allowed interpolators are {list(PDF_CONSTRUCTORS.keys())}"  # pylint: disable=line-too-long
            )

        if value is self._pdf_constructor_name:
            logging.warning("Already using interpolator: '%s'.", value)
            return

        self._pdf_constructor_name = value
        self._instantiate_pdf_constructor()
        self._addmetadata("pdf_constructor_name", self._pdf_constructor_name)

    @property
    def pdf_constructor(self) -> AbstractQuantilePdfConstructor:
        """Returns the current PDF constructor, and allows the user to interact
        with its methods.

        Returns
        -------
        AbstractQuantilePdfConstructor
            Abstract base class of the active concrete PDF constructor.
        """
        return self._pdf_constructor

    def _instantiate_pdf_constructor(self):
        self._pdf_constructor = PDF_CONSTRUCTORS[self._pdf_constructor_name](
            self._quants, self._locs
        )

    def _pdf(self, x, *args):
        # We're not requiring that the output be normalized!
        # `util.normalize_interp1d` addresses _one_ of the ways that a reconstruction
        # can be bad, but not all. It should be replaced with a more comprehensive
        # normalization function.
        # See qp issue #147
        row = args[0]
        return self._pdf_constructor.construct_pdf(x, row)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_multi_x_y(
            x,
            row,
            self._locs,
            self._quants,
            bounds_error=False,
            fill_value=(0.0, 1),
            kind="quadratic",
        ).ravel()

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_x_multi_y(
            x,
            row,
            self._quants,
            self._locs,
            bounds_error=False,
            fill_value=(self._xmin, self._xmax),
            kind="quadratic",
        ).ravel()

    def _updated_ctor_param(self):
        """
        Set the quants and locs as additional constructor arguments
        """
        dct = super()._updated_ctor_param()
        dct["quants"] = self._quants
        dct["locs"] = self._locs
        dct["pdf_constructor_name"] = self._pdf_constructor_name
        dct["check_input"] = self._check_input
        return dct

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        """Return kwds necessary to create 'empty' hdf5 file with npdf entries
        for iterative writeout.  We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.
        """
        try:
            quants = kwargs["quants"]
        except ValueError:  # pragma: no cover
            print("required argument 'quants' not included in kwargs")
        nquants = np.shape(quants)[-1]
        return dict(locs=((npdf, nquants), "f4"))

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a quantile this shows the quantiles points
        """
        axes, xlim, kw = get_axes_and_xlims(**kwargs)
        xvals = np.linspace(xlim[0], xlim[1], kw.pop("npts", 101))
        locs = np.squeeze(pdf.dist.locs[pdf.kwds["row"]])
        quants = np.squeeze(pdf.dist.quants)
        yvals = np.squeeze(pdf.pdf(xvals))
        return plot_pdf_quantiles_on_axes(
            axes, xvals, yvals, quantiles=(quants, locs), **kw
        )

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_quantiles, None)


quant = quant_gen.create

quant_gen.test_data = dict(
    quant=dict(
        gen_func=quant,
        ctor_data=dict(quants=QUANTS, locs=QLOCS),
        convert_data=dict(quants=QUANTS),
        test_xvals=TEST_XVALS,
    )
)

add_class(quant_gen)
