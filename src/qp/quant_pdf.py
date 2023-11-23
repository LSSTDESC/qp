"""This module implements a PDT distribution sub-class using interpolated quantiles
"""
import logging
import sys

import numpy as np
from scipy.stats import rv_continuous

from qp.conversion_funcs import extract_quantiles
from qp.factory import add_class
from qp.pdf_gen import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_quantiles_on_axes
from qp.quantile_pdf_constructors import (
    AbstractQuantilePdfConstructor,
    CdfSplineDerivative,
    DualSplineAverage,
    PiecewiseConstant,
    PiecewiseLinear,
)
from qp.test_data import QLOCS, QUANTS, TEST_XVALS
from qp.utils import interpolate_multi_x_y, interpolate_x_multi_y, reshape_to_pdf_size

epsilon = sys.float_info.epsilon


def pad_quantiles(quants, locs):
    """Pad the quantiles and locations used to build a quantile representation

    Parameters
    ---------
    quants : array_like
        The quantiles used to build the CDF
    locs : array_like
        The locations at which those quantiles are reached

    Returns
    -------
    quants : array_like
        The quantiles used to build the CDF
    locs : array_like
        The locations at which those quantiles are reached
    """
    n_out = n_vals = quants.size
    if quants[0] > sys.float_info.epsilon:
        offset_lo = 1
        pad_lo = True
        n_out += 1
    else:
        offset_lo = 0
        pad_lo = False
    if quants[-1] < 1.0:
        pad_hi = True
        n_out += 1
    else:
        pad_hi = False
    if n_out == n_vals:
        return quants, locs
    quants_out = np.zeros((n_out), quants.dtype)
    locs_out = np.zeros((locs.shape[0], n_out), quants.dtype)
    quants_out[offset_lo : n_vals + offset_lo] = quants
    locs_out[:, offset_lo : n_vals + offset_lo] = locs
    if pad_lo:
        locs_out[:, 0] = locs[:, 0] - quants[0] * (locs[:, 1] - locs[:, 0]) / (
            quants[1] - quants[0]
        )

    if pad_hi:
        quants_out[-1] = 1.0
        locs_out[:, -1] = locs[:, -1] - (1.0 - quants[-1]) * (
            locs[:, -2] - locs[:, -1]
        ) / (quants[-1] - quants[-2])

    return quants_out, locs_out


DEFAULT_PDF_CONSTRUCTOR = "piecewise_linear"
PDF_CONSTRUCTORS = {
    "cdf_spline_derivative": CdfSplineDerivative,
    "dual_spline_average": DualSplineAverage,
    "piecewise_linear": PiecewiseLinear,
    "piecewise_constant": PiecewiseConstant,
}


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
        """Plot the PDF in a way that is particular to this type of distibution

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
