"""This module implements a PDT distribution sub-class using interpolated grids
"""

import numpy as np
from scipy.stats import rv_continuous

from qp.conversion_funcs import extract_vals_at_x, extract_xy_sparse, extract_xy_vals
from qp.factory import add_class
from qp.pdf_gen import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.test_data import TEST_XVALS, XARRAY, XBINS, YARRAY
from qp.utils import (
    interpolate_multi_x_multi_y,
    interpolate_multi_x_y,
    interpolate_x_multi_y,
    normalize_interp1d,
    reshape_to_pdf_size,
)


class interp_gen(Pdf_rows_gen):
    """Interpolator based distribution

    Notes
    -----
    This implements a PDF using a set of interpolated values.

    This version use the same xvals for all the the PDFs, which
    allows for much faster evaluation, and reduces the memory
    usage by a factor of 2.

    The relevant data members are:

    xvals:  (n) x values

    yvals:  (npdf, n) y values

    Inside the range xvals[0], xvals[-1] tt simply takes a set of x and y values
    and uses `scipy.interpolate.interp1d` to build the PDF.
    Outside the range xvals[0], xvals[-1] the pdf() will return 0.

    The cdf() is constructed by integrating analytically computing the cumulative
    sum at the xvals grid points and interpolating between them.
    This will give a slight discrepency with the true integral of the pdf(),
    bit is much, much faster to evaluate.
    Outside the range xvals[0], xvals[-1] the cdf() will return (0 or 1), respectively

    The ppf() is computed by inverting the cdf().
    ppf(0) will return xvals[0]
    ppf(1) will return xvals[-1]
    """

    # pylint: disable=protected-access

    name = "interp"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, xvals, yvals, *args, **kwargs):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation
        """
        if np.size(xvals) != np.shape(yvals)[-1]:  # pragma: no cover
            raise ValueError(
                "Shape of xbins in xvals (%s) != shape of xbins in yvals (%s)"
                % (np.size(xvals), np.shape(yvals)[-1])
            )
        self._xvals = np.asarray(xvals)

        # Set support
        self._xmin = self._xvals[0]
        self._xmax = self._xvals[-1]
        kwargs["shape"] = np.shape(yvals)[:-1]

        self._yvals = reshape_to_pdf_size(yvals, -1)

        check_input = kwargs.pop("check_input", True)
        if check_input:
            self._compute_ycumul()
            self._yvals = (self._yvals.T / self._ycumul[:, -1]).T
            self._ycumul = (self._ycumul.T / self._ycumul[:, -1]).T
        else:  # pragma: no cover
            self._ycumul = None

        super().__init__(*args, **kwargs)
        self._addmetadata("xvals", self._xvals)
        self._addobjdata("yvals", self._yvals)

    def _compute_ycumul(self):
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.5 * self._yvals[:, 0] * (self._xvals[1] - self._xvals[0])
        self._ycumul[:, 1:] = np.cumsum(
            (self._xvals[1:] - self._xvals[:-1])
            * 0.5
            * np.add(self._yvals[:, 1:], self._yvals[:, :-1]),
            axis=1,
        )

    @property
    def xvals(self):
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-valus used to do the interpolation"""
        return self._yvals

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
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
        return np.sum(self._xvals**m * self._yvals, axis=1) * dx

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct["xvals"] = self._xvals
        dct["yvals"] = self._yvals
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
        return dict(yvals=((npdf, ngrid), "f4"))

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
        cls._add_extraction_method(extract_vals_at_x, None)

    @classmethod
    def make_test_data(cls):
        """Make data for unit tests"""
        cls.test_data = dict(
            interp=dict(
                gen_func=interp,
                ctor_data=dict(xvals=XBINS, yvals=YARRAY),
                convert_data=dict(xvals=XBINS),
                test_xvals=TEST_XVALS,
            )
        )


interp = interp_gen.create


class interp_irregular_gen(Pdf_rows_gen):
    """Interpolator based distribution

    Notes
    -----
    This implements a PDF using a set of interpolated values.

    This version use the different xvals for each the the PDFs, which
    allows for more precision.

    The relevant data members are:

    xvals:  (npdf, n) x values

    yvals:  (npdf, n) y values

    Inside the range xvals[:,0], xvals[:,-1] tt simply takes a set of x and y values
    and uses `scipy.interpolate.interp1d` to build the PDF.
    Outside the range xvals[:,0], xvals[:,-1] the pdf() will return 0.

    The cdf() is constructed by integrating analytically computing the cumulative
    sum at the xvals grid points and interpolating between them.
    This will give a slight discrepency with the true integral of the pdf(),
    bit is much, much faster to evaluate.
    Outside the range xvals[:,0], xvals[:,-1] the cdf() will return (0 or 1), respectively

    The ppf() is computed by inverting the cdf().
    ppf(0) will return min(xvals)
    ppf(1) will return max(xvals)
    """

    # pylint: disable=protected-access

    name = "interp_irregular"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, xvals, yvals, *args, **kwargs):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation
        """
        if np.shape(xvals) != np.shape(yvals):  # pragma: no cover
            raise ValueError(
                "Shape of xvals (%s) != shape of yvals (%s)"
                % (np.shape(xvals), np.shape(yvals))
            )
        self._xvals = reshape_to_pdf_size(xvals, -1)

        self._xmin = np.min(self._xvals)
        self._xmax = np.max(self._xvals)
        kwargs["shape"] = np.shape(xvals)[:-1]

        check_input = kwargs.pop("check_input", True)
        self._yvals = reshape_to_pdf_size(yvals, -1)
        if check_input:
            self._yvals = normalize_interp1d(self._xvals, self._yvals)
        self._ycumul = None
        super().__init__(*args, **kwargs)
        self._addobjdata("xvals", self._xvals)
        self._addobjdata("yvals", self._yvals)

    def _compute_ycumul(self):
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.0
        self._ycumul[:, 1:] = np.cumsum(
            self._xvals[:, 1:] * self._yvals[:, 1:]
            - self._xvals[:, :-1] * self._yvals[:, 1:],
            axis=1,
        )

    @property
    def xvals(self):
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-valus used to do the interpolation"""
        return self._yvals

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_multi_x_multi_y(
            x, row, self._xvals, self._yvals, bounds_error=False, fill_value=0.0
        ).ravel()

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        return interpolate_multi_x_multi_y(
            x, row, self._xvals, self._ycumul, bounds_error=False, fill_value=(0.0, 1.0)
        ).ravel()

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        return interpolate_multi_x_multi_y(
            x,
            row,
            self._ycumul,
            self._xvals,
            bounds_error=False,
            fill_value=(self._xmin, self._xmax),
        ).ravel()

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct["xvals"] = self._xvals
        dct["yvals"] = self._yvals
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
        return dict(xvals=((npdf, ngrid), "f4"), yvals=((npdf, ngrid), "f4"))

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a interpolated PDF this uses the interpolation points
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        xvals_row = pdf.dist.xvals
        return plot_pdf_on_axes(axes, pdf, xvals_row, **kw)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_xy_vals, None)
        cls._add_extraction_method(extract_xy_sparse, None)

    @classmethod
    def make_test_data(cls):
        """Make data for unit tests"""
        cls.test_data = dict(
            interp_irregular=dict(
                gen_func=interp_irregular,
                ctor_data=dict(xvals=XARRAY, yvals=YARRAY),
                convert_data=dict(xvals=XBINS),
                test_xvals=TEST_XVALS,
            )
        )


interp_irregular = interp_irregular_gen.create
add_class(interp_gen)
add_class(interp_irregular_gen)
