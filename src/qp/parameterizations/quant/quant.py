"""This module implements a distribution parameterization sub-class using interpolated quantiles"""

from __future__ import annotations
import logging
import sys

import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike
import warnings

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


class quant_gen(Pdf_rows_gen):  # pylint: disable=too-many-instance-attributes
    """Quantile based distribution, where the PDF is defined from the quantiles.


    Parameters
    ----------
    quants : ArrayLike
        The quantiles of the CDF, of shape n
    locs : ArrayLike
        The locations at which those quantiles are reached, of shape (npdf, n)
    pdf_constructor_name : str, optional
        The constructor or interpolator to use to create the PDF, by default "piecewise_linear".
    ensure_extent : bool, optional
        If True, will ensure that the quants start at 0 and end at 1 by adding
        data points at both ends until this is true. locs are extrapolated linearly
        from input data. By default True.
    warn : bool, optional
        If True, raises warnings if input is not valid data (i.e. if
        data is not finite). If False, no warnings are raised. By default True.


    Notes
    -----

    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +---------------------+-----------+------------+
    | Function            | Arguments | Method key |
    +---------------------+-----------+------------+
    |`.extract_quantiles` | quants    | None       |
    +---------------------+-----------+------------+

    Implementation notes:

    This implements a CDF by interpolating a set of quantile values

    It takes a set of quants and locs values and uses `scipy.interpolate.interp1d`
    with a spline interpolation method of order 2 (kind=`quadratic`) to build the CDF.

    It has multiple PDF constructors to get the PDF from the quantiles. The default
    is the `piecewise_linear` method, which takes the numerical derivative of the
    CDF and interpolates between those points.

    `ppf(0)` returns negative infinity and `ppf(1)` returns positive infinity.

    """

    # pylint: disable=protected-access

    name = "quant"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        quants: ArrayLike,
        locs: ArrayLike,
        pdf_constructor_name: str = DEFAULT_PDF_CONSTRUCTOR,
        ensure_extent: bool = True,
        warn: bool = True,
        *args,
        **kwargs,
    ):
        """
        Create a new distribution using the given values

        Parameters
        ----------
        quants : ArrayLike
           The quantiles of the CDF, of shape n
        locs : ArrayLike
           The locations at which those quantiles are reached, of shape (npdf, n)
        pdf_constructor_name : str, optional
            The constructor to use to create the PDF, by default "piecewise_linear".
        ensure_extent : bool, optional
            If True, will ensure that the quants start at 0 and end at 1 by adding
            data points at both ends until this is true. locs are extrapolated linearly
            from input data. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid data (i.e. if
            data is not finite). If False, no warnings are raised. By default True.
        """

        self._xmin = np.min(locs)
        self._xmax = np.max(locs)

        locs_2d = reshape_to_pdf_size(np.asarray(locs), -1)

        # make sure input makes sense for a CDF
        self._validate_input(np.asarray(quants), locs_2d)

        # check locs are finite
        self._warn = warn
        if self._warn:
            if not np.all(np.isfinite(locs_2d)):
                indices = np.where(np.isfinite(locs_2d) != True)
                warnings.warn(
                    f"There are non-finite values in the locs for the distributions: {indices[0]}",
                    RuntimeWarning,
                )

        self._ensure_extent = ensure_extent
        if self._ensure_extent:
            quants, locs_2d = pad_quantiles(quants, locs_2d)

        self._quants = np.asarray(quants)
        self._nquants = self._quants.size
        if locs_2d.shape[-1] != self._nquants:  # pragma: no cover
            raise ValueError(
                "Number of locations (%i) != number of quantile values (%i)"
                % (self._nquants, locs_2d.shape[-1])
            )
        self._locs = locs_2d

        # set up PDF constructor
        if not isinstance(pdf_constructor_name, str):
            try:
                pdf_constructor_name = str(np.strings.decode(pdf_constructor_name))
            except AttributeError as a_err:
                pdf_constructor_name = str(pdf_constructor_name)

        if pdf_constructor_name not in PDF_CONSTRUCTORS:
            raise ValueError(
                f"Unknown interpolator provided: '{pdf_constructor_name}'. Allowed interpolators are {list(PDF_CONSTRUCTORS.keys())}"  # pylint: disable=line-too-long
            )
        self._pdf_constructor_name = pdf_constructor_name
        self._pdf_constructor = None
        self._instantiate_pdf_constructor()

        kwargs["shape"] = self._locs.shape  # locs.shape
        super().__init__(*args, **kwargs)

        self._addmetadata("quants", self._quants)
        self._addmetadata("pdf_constructor_name", self._pdf_constructor_name.encode())
        self._addmetadata("ensure_extent", self._ensure_extent)
        self._addobjdata("locs", self._locs)

    def _validate_input(self, quants, locs):
        """Ensures that given input matches criteria for a valid CDF."""

        if np.any(quants < 0) or np.any(quants > 1):
            raise ValueError(
                f"Invalid quants: One or more of the given quants is outside the allowed range (0,1): {quants}"
            )
        if not np.all(np.diff(quants) >= 0):
            raise ValueError(
                f"Invalid quants: \n There are decreasing values, quants must be given in order from 0 to 1: {quants}"
            )
        if not np.all(np.diff(locs) >= 0):
            indices = np.where(np.diff(locs) < 0)
            raise ValueError(
                f"Invalid locs: \n The given data does not produce a one-to-one CDF for the distributions at the following indices: {indices}"
            )

    @property
    def quants(self) -> np.ndarray[float]:
        """Return quantiles used to build the CDF"""
        return self._quants

    @property
    def locs(self) -> np.ndarray[float]:
        """Return the locations at which those quantiles are reached"""
        return self._locs

    @property
    def pdf_constructor_name(self) -> str:
        """Returns the name of the current pdf constructor. Matches a key in
        the `PDF_CONSTRUCTORS` dictionary."""
        return self._pdf_constructor_name

    @pdf_constructor_name.setter
    def pdf_constructor_name(self, value: str) -> None:
        """Allows users to specify a different interpolator without having to recreate
        the ensemble.

        Parameters
        ----------
        value : str
            One of the supported interpolators. See `PDF_CONSTRUCTORS`
            dictionary for supported interpolators.

        Raises
        ------
        ValueError
            If the value provided isn't a key in `PDF_CONSTRUCTORS`, raise
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

    def x_samples(self) -> np.ndarray[float]:
        """Return a set of x values that can be used to plot all the CDFs."""

        # get the range and median distance between points
        min_dx = np.median(np.diff(self._locs))
        min_val = np.min(self._locs)
        max_val = np.max(self._locs)

        # get the number of points (make sure it's less than some huge number)
        npts = (max_val - min_val) // min_dx
        npts = np.min([int(npts), 10000])
        return np.linspace(min_val, max_val, npts)

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
        dct["ensure_extent"] = self._ensure_extent
        dct["warn"] = self._warn
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf, **kwargs
    ) -> dict[str, tuple[tuple[int, int], str]]:
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
        dict[str, tuple[tuple[int, int], str]]
            A dictionary with a key for the objdata, a tuple with the shape of that data,
            and the data type of the data as a string.
            i.e. ``{objdata_key = ((npdf, n), "f4")}``

        Raises
        ------
        ValueError
            Raises an error if the required kwarg quants is not provided.
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

        For a quantile this shows the quantiles points.

        Parameters
        ----------
        axes : Axes
            The axes to plot on. Either this or xlim must be provided.
        xlim : tuple[float, float]
            The x-axis limits. Either this or axes must be provided.

        Other Parameters
        ----------------
        npts : int, optional
            The number of x values to create within the limits, by default 101
        kwargs :
            Any keyword arguments to pass to matplotlib's axes.hist() method.

        Returns
        -------
        axes : Axes
            The plot axes.
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
    def add_mappings(cls) -> None:
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_quantiles, None)

    @classmethod
    def create_ensemble(
        self,
        quants: ArrayLike,
        locs: ArrayLike,
        pdf_constructor_name: str = DEFAULT_PDF_CONSTRUCTOR,
        ensure_extent: bool = True,
        warn: bool = True,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as quantiles.


        The options for pdf_constructor_name are: `piecewise_linear`, `piecewise_constant`,
        `dual_spline_average` and 'cdf_spline_derivative`.


        Parameters
        ----------
        quants : ArrayLike
           The quantiles used to build the CDF, shape n
        locs : ArrayLike
           The locations at which those quantiles are reached, shape (npdfs, n),
           where npdfs is the number of distributions.
        pdf_constructor_name : str, optional
            The constructor to use to create the PDF, by default "piecewise_linear".
        ensure_extent : bool, optional
            If True, will ensure that the quants start at 0 and end at 1 by adding
            data points at both ends until this is true. locs are extrapolated linearly
            from input data. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid (i.e. if
            locs are not finite values). If False, no warnings are raised.
            By default True.
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have
            the same length as the number of distributions, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Examples
        --------

        To create an Ensemble with two distributions and associated ids, using the
        `dual_spline_average` constructor:

        >>> import qp
        >>> import numpy as np
        >>> quants = np.array([0.0001,0.25,0.5,0.75,0.9999])
        >>> locs = np.array([[0.0001,0.1,0.3,0.5,0.75],[0.01,0.05,0.15,0.3,0.5]])
        >>> pdf_constructor_name = 'dual_spline_average'
        >>> ancil = {'ids':[11,18]}
        >>> ens = qp.quant.create_ensemble(quants,locs,pdf_constructor_name,ancil=ancil)
        >>> ens.metadata
        {'pdf_name': array([b'quant'], dtype='|S5'),
        'pdf_version': array([0]),
        'quants': array([[0.000e+00, 1.000e-04, 2.500e-01, 5.000e-01, 7.500e-01, 9.999e-01,
                1.000e+00]]),
        'pdf_constructor_name': array(['dual_spline_average'], dtype='|S19'),
        'check_input': array([ True])}
        """
        data = {
            "quants": quants,
            "locs": locs,
            "pdf_constructor_name": pdf_constructor_name,
            "ensure_extent": ensure_extent,
            "warn": warn,
        }
        return Ensemble(self, data, ancil)


quant = quant_gen


add_class(quant_gen)
