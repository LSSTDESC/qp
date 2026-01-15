"""This module implements a PDT distribution sub-class using interpolated grids"""

from __future__ import annotations
import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike

from ...core.factory import add_class
from ...core.ensemble import Ensemble
from .packing_utils import PackingType, pack_array, unpack_array
from ..base import Pdf_rows_gen
from ...plotting import get_axes_and_xlims, plot_pdf_on_axes
from ...utils.array import reshape_to_pdf_size
from ...utils.interpolation import interpolate_multi_x_y, interpolate_x_multi_y


def extract_and_pack_vals_at_x(in_dist: "Ensemble", **kwargs):
    """Convert using a set of x and packed y values

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    xvals : np.ndarray
        Locations at which the pdf is evaluated

    packing_type : PackingType
        Enum specifying the type of packing to use

    Returns
    -------
    data : dict
        The extracted data
    """
    xvals = kwargs.pop("xvals", None)
    packing_type = kwargs.pop("packing_type")
    if xvals is None:  # pragma: no cover
        raise ValueError("To convert to extract_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    ypacked, ymax = pack_array(packing_type, yvals, **kwargs)
    return dict(
        xvals=xvals, ypacked=ypacked, ymax=ymax, packing_type=packing_type, **kwargs
    )


class packed_interp_gen(Pdf_rows_gen):  # pylint: disable=too-many-instance-attributes
    """Interpolator based distribution

    Notes
    -----
    This is a version of the interp_pdf that stores the data using a packed integer representation.

    See qp.packing_utils for options on packing

    See qp.interp_pdf for details on interpolation
    """

    # pylint: disable=protected-access

    name = "packed_interp"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        xvals,
        ypacked,
        ymax,
        *args,
        packing_type=PackingType.linear_from_rowmax,
        log_floor=-3.0,
        **kwargs,
    ):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : ArrayLike
          The x-values used to do the interpolation
        ypacked : ArrayLike
          The packed version of the y-values used to do the interpolation
        ymax : ArrayLike
          The maximum y-values for each pdf
        packing_type: PackingType
            By default `PackingType.linear_from_rowmax`
        log_floor: float
            By default -3
        """
        if np.size(xvals) != np.shape(ypacked)[-1]:  # pragma: no cover
            raise ValueError(
                "Shape of xbins in xvals (%s) != shape of xbins in yvals (%s)"
                % (np.size(xvals), np.shape(ypacked)[-1])
            )
        self._xvals = np.asarray(xvals)

        # Set support
        self._xmin = self._xvals[0]
        self._xmax = self._xvals[-1]
        # kwargs["shape"] = np.shape(ypacked)[:-1]

        self._yvals = None
        if isinstance(packing_type, PackingType):
            self._packing_type = packing_type.value
        else:
            self._packing_type = packing_type
        self._log_floor = log_floor
        self._ymax = reshape_to_pdf_size(ymax, -1)
        self._ypacked = reshape_to_pdf_size(ypacked, -1)
        kwargs["shape"] = np.shape(self._ypacked)

        check_input = kwargs.pop("check_input", True)
        if check_input:
            self._compute_ycumul()
            self._yvals = (self._yvals.T / self._ycumul[:, -1]).T
            self._ycumul = (self._ycumul.T / self._ycumul[:, -1]).T
        else:  # pragma: no cover
            self._ycumul = None

        super().__init__(*args, **kwargs)
        self._addmetadata("xvals", self._xvals)
        self._addmetadata("packing_type", self._packing_type)
        self._addmetadata("log_floor", self._log_floor)
        self._addobjdata("ypacked", self._ypacked)
        self._addobjdata("ymax", self._ymax)

    def _compute_ycumul(self):
        if self._yvals is None:
            self._unpack()
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.5 * self._yvals[:, 0] * (self._xvals[1] - self._xvals[0])
        self._ycumul[:, 1:] = np.cumsum(
            (self._xvals[1:] - self._xvals[:-1])
            * 0.5
            * np.add(self._yvals[:, 1:], self._yvals[:, :-1]),
            axis=1,
        )

    def _unpack(self):
        self._yvals = unpack_array(
            PackingType(self._packing_type),
            self._ypacked,
            row_max=self._ymax,
            log_floor=self._log_floor,
        )

    @property
    def xvals(self):
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def packing_type(self):
        """Returns the packing type"""
        return self._packing_type

    @property
    def log_floor(self):
        """Returns the packing type"""
        return self._log_floor

    @property
    def ypacked(self):
        """Returns the packed y-vals"""
        return self._ypacked

    @property
    def ymax(self):
        """Returns the max for each row"""
        return self._ymax

    @property
    def yvals(self):
        """Return the y-valus used to do the interpolation"""
        return self._yvals

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._yvals is None:  # pragma: no cover
            self._unpack()
        pdf = interpolate_x_multi_y(
            x, row, self._xvals, self._yvals, bounds_error=False, fill_value=0.0
        ).ravel()

        # reduce dimension to 0 if there's only one value
        if np.shape(pdf) == (1,) and len(pdf) == 1:
            return pdf[0]
        else:
            return pdf

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        return interpolate_x_multi_y(
            x, row, self._xvals, self._ycumul, bounds_error=False, fill_value=(0.0, 1.0)
        ).ravel()

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()

        return interpolate_multi_x_y(
            x,
            row,
            self._ycumul,
            self._xvals,
            bounds_error=False,
            fill_value=(self._xmin, self._xmax),
        ).ravel()

    def _munp(self, m, *args):
        """compute moments"""
        # pylint: disable=arguments-differ
        # Silence floating point warnings from integration.
        with np.errstate(all="ignore"):
            vals = self.custom_generic_moment(m)
        return vals

    def custom_generic_moment(self, m):
        """Compute the mth moment"""
        m = np.asarray(m)
        dx = self._xvals[1] - self._xvals[0]
        if self._yvals is None:  # pragma: no cover
            self._unpack()
        return np.sum(self._xvals**m * self._yvals, axis=1) * dx

    def _updated_ctor_param(self):
        """
        Set the bin edges and packing data as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct["xvals"] = self._xvals
        dct["ypacked"] = self._ypacked
        dct["ymax"] = self._ymax
        dct["log_floor"] = self._log_floor
        dct["packing_type"] = PackingType(self._packing_type)
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf, **kwargs
    ) -> dict[str, tuple[tuple[int, int], str]]:
        """Return the keywords necessary to create an 'empty' hdf5 file with npdf entries
        for iterative file writeout.  We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.

        Parameters
        ----------
        npdf: int
            number of *total* PDFs that will be written out
        kwargs: dict
            dictionary of kwargs needed to create the ensemble

        Returns
        -------
        dict[str, tuple[tuple[int, int], str]]
        """
        if "xvals" not in kwargs:  # pragma: no cover
            raise ValueError("required argument xvals not included in kwargs")
        ngrid = np.shape(kwargs["xvals"])[-1]
        return dict(
            ypacked=((npdf, ngrid), "u1"),
            ymax=((npdf, 1), "f4"),
        )

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a interpolated PDF this uses the interpolation points
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        return plot_pdf_on_axes(axes, pdf, pdf.dist.xvals, **kw)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_and_pack_vals_at_x, None)

    @classmethod
    def create_ensemble(
        self,
        xvals: ArrayLike,
        ypacked: ArrayLike,
        ymax: ArrayLike,
        packing_type=PackingType.linear_from_rowmax,
        log_floor=-3.0,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as interpolation that are stored as packed integers.


        Parameters
        ----------
        xvals : ArrayLike
          The x-values used to do the interpolation
        ypacked : ArrayLike
          The packed version of the y-values used to do the interpolation
        ymax : ArrayLike
          The maximum y-values for each pdf
        packing_type: PackingType
            By default `PackingType.linear_from_rowmax`
        log_floor: float
            By default -3

        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have the same length as the number of distributions, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.


        """

        data = {
            "xvals": xvals,
            "ypacked": ypacked,
            "ymax": ymax,
            "packing_type": packing_type,
            "log_floor": log_floor,
        }
        return Ensemble(self, data, ancil)


packed_interp = packed_interp_gen

add_class(packed_interp_gen)
