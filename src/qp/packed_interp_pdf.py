"""This module implements a PDT distribution sub-class using interpolated grids
"""

import numpy as np
from scipy.stats import rv_continuous

from qp.factory import add_class
from qp.packing_utils import PackingType, pack_array, unpack_array
from qp.pdf_gen import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.test_data import TEST_XVALS, XBINS, YARRAY
from qp.utils import interpolate_multi_x_y, interpolate_x_multi_y, reshape_to_pdf_size


def extract_and_pack_vals_at_x(in_dist, **kwargs):
    """Convert using a set of x and packed y values

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    packing_type : PackingType
        Enum specifying the type of packing to use

    Returns
    -------
    data : `dict`
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
        xvals : array_like
          The x-values used to do the interpolation
        ypacked : array_like
          The packed version of the y-values used to do the interpolation
        ymax : array_like
          The maximum y-values for each pdf
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
        kwargs["shape"] = np.shape(ypacked)[:-1]

        self._yvals = None
        if isinstance(packing_type, PackingType):
            self._packing_type = packing_type.value
        else:
            self._packing_type = packing_type
        self._log_floor = log_floor
        self._ymax = reshape_to_pdf_size(ymax, -1)
        self._ypacked = reshape_to_pdf_size(ypacked, -1)

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
        return interpolate_x_multi_y(
            x, row, self._xvals, self._yvals, bounds_error=False, fill_value=0.0
        ).ravel()

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
    def get_allocation_kwds(cls, npdf, **kwargs):
        """Return the keywords necessary to create an 'empty' hdf5 file with npdf entries
        for iterative file writeout.  We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.

        Parameters
        ----------
        npdf: int
            number of *total* PDFs that will be written out
        kwargs: dict
            dictionary of kwargs needed to create the ensemble
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
    def make_test_data(cls):
        """Make data for unit tests"""
        ypacked_lin, ymax_lin = pack_array(
            PackingType.linear_from_rowmax, YARRAY.copy()
        )
        ypacked_log, ymax_log = pack_array(
            PackingType.log_from_rowmax, YARRAY.copy(), log_floor=-3
        )

        cls.test_data = dict(
            lin_packed_interp=dict(
                gen_func=packed_interp,
                ctor_data=dict(
                    packing_type=PackingType.linear_from_rowmax,
                    xvals=XBINS,
                    ypacked=ypacked_lin,
                    ymax=ymax_lin,
                ),
                convert_data=dict(
                    xvals=XBINS, packing_type=PackingType.linear_from_rowmax
                ),
                test_xvals=TEST_XVALS,
            ),
            log_packed_interp=dict(
                gen_func=packed_interp,
                ctor_data=dict(
                    packing_type=PackingType.log_from_rowmax,
                    xvals=XBINS,
                    ypacked=ypacked_log,
                    ymax=ymax_log,
                    log_floor=-3.0,
                ),
                convert_data=dict(
                    xvals=XBINS,
                    packing_type=PackingType.log_from_rowmax,
                    log_floor=-3.0,
                ),
                test_xvals=TEST_XVALS,
            ),
        )


packed_interp = packed_interp_gen.create

add_class(packed_interp_gen)
