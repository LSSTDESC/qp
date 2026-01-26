"""This module implements a distribution parameterization sub-class using interpolated grids"""

from __future__ import annotations
import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike
import warnings

from .interp_utils import (
    irreg_interp_extract_xy_vals,
    extract_vals_at_x,
    extract_xy_sparse,
)
from ...core.factory import add_class
from ...core.ensemble import Ensemble
from ..base import Pdf_rows_gen
from ...plotting import get_axes_and_xlims, plot_pdf_on_axes
from ...utils.array import reshape_to_pdf_size, reduce_dimensions
from ...utils.interpolation import (
    interpolate_multi_x_multi_y,
    interpolate_multi_x_y,
    interpolate_x_multi_y,
)


class interp_gen(Pdf_rows_gen):
    """Implements distributions parameterized as interpolated sets of values.

    All distributions share the same x values. Interpolation is performed using
    `scipy.interpolate.interp1d`, with the default interpolation method (linear).


    Parameters
    ----------
    xvals : ArrayLike
        The n x-values that are used by all the distributions
    yvals : ArrayLike
        The y-values that represent each distribution, with shape (npdf,n)
    norm : bool, optional
        If `True`, normalizes the input distribution. If `False`, assumes the
        given distribution is already normalized. By default `True`.
    warn : bool, optional
        If `True`, raises warnings if input is not valid PDF data (i.e. if
        data is negative). If `False`, no warnings are raised. By default `True`.


    Notes
    -----

    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +---------------------+-----------+------------+
    | Function            | Arguments | Method key |
    +---------------------+-----------+------------+
    | `.extract_vals_at_x`| xvals     | None       |
    +---------------------+-----------+------------+

    Implementation notes:

    This uses the same xvals for all the the PDFs, unlike `interp_irregular_gen` which
    has a different set of xvals for each distribution.
    `interp_gen` therefore allows for much faster evaluation than `interp_irregular_gen`,
    and reduces the memory usage by a factor of 2.

    Inside the range of given xvals it takes a set of x and y values
    and uses `scipy.interpolate.interp1d` to build the PDF.
    Outside the range of given xvals the `pdf()` will return 0.

    The `cdf()` is constructed by integrating analytically -- computing the cumulative
    sum at the given xvals and interpolating between them.
    This will give a slight discrepancy with the true integral of the `pdf()`,
    but is much, much faster to evaluate.
    Outside the range of given xvals the `cdf()` will return 0 or 1, respectively

    The `ppf()` is computed by inverting the `cdf()`. `ppf(0)` will return negative infinity
    and `ppf(1)` will return positive infinity.
    """

    # pylint: disable=protected-access

    name = "interp"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        xvals: ArrayLike,
        yvals: ArrayLike,
        norm: bool = True,
        warn: bool = True,
        *args,
        **kwargs,
    ):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : ArrayLike
            The n x-values that are used by all the distributions
        yvals : ArrayLike
            The y-values that represent each distribution, with shape (npdf,n)
        norm : bool, optional
            If `True`, normalizes the input distribution. If `False`, assumes the
            given distribution is already normalized. By default `True`.
        warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If `False`, no warnings are raised. By default `True`.
        """
        if np.size(xvals) != np.shape(yvals)[-1]:  # pragma: no cover
            raise ValueError(
                "Shape of xbins in xvals (%s) != shape of xbins in yvals (%s)"
                % (np.size(xvals), np.shape(yvals)[-1])
            )
        self._xvals = np.asarray(xvals)
        self._yvals = reshape_to_pdf_size(np.asarray(yvals), -1)
        kwargs["shape"] = np.shape(self._yvals)

        # make sure that the xvals are sorted
        if not np.all(np.diff(self._xvals) >= 0):
            raise ValueError(
                f"Invalid xvals: The given xvals are not sorted: {self._xvals}"
            )

        # raise warnings if input data is not finite or pdfs are not positive
        self._warn = warn
        if self._warn:
            if not np.all(np.isfinite(self._xvals)):
                warnings.warn(
                    "The given xvals contain non-finite values", RuntimeWarning
                )
            if not np.all(np.isfinite(self._yvals)):
                indices = np.where(np.isfinite(self._yvals) != True)
                warnings.warn(
                    f"There are non-finite values in the yvals for the following distributions: {indices}",
                    RuntimeWarning,
                )
            if np.any(self._yvals < 0):
                indices = np.where(self._yvals < 0)
                warnings.warn(
                    f"There are negative values in the yvals for the following distributions: {indices}",
                    RuntimeWarning,
                )

        # Set support
        self._xmin = self._xvals[0]
        self._xmax = self._xvals[-1]
        # kwargs["shape"] = np.shape(yvals)

        # normalize the distribution if norm is True
        self._norm = norm
        if self._norm:
            self._yvals = self.normalize()["yvals"]
        else:  # pragma: no cover
            self._ycumul = None

        super().__init__(*args, **kwargs)
        self._addmetadata("xvals", self._xvals)
        self._addobjdata("yvals", self._yvals)

    def _compute_ycumul(self) -> None:
        """Compute the integral under the distribution"""
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.5 * self._yvals[:, 0] * (self._xvals[1] - self._xvals[0])
        self._ycumul[:, 1:] = np.cumsum(
            (self._xvals[1:] - self._xvals[:-1])
            * 0.5
            * np.add(self._yvals[:, 1:], self._yvals[:, :-1]),
            axis=1,
        )

        # raise an error if the sum is 0 or negative
        if np.any(self._ycumul[:, -1] < 0):
            indices = np.where(self._ycumul[:, -1] < 0)
            raise ValueError(
                f"The distribution(s) cannot be properly normalized, the integral is < 0 for distributions at indices = {indices[0]} \n with yvals {self._yvals[indices[0]]}"
            )
        elif np.any(self._ycumul[:, -1] == 0):
            indices = np.where(self._ycumul[:, -1] == 0)
            warnings.warn(
                f"The distributions at indices = {indices[0]} have an integral of 0."
            )

    def normalize(self) -> Mapping[str, np.ndarray[float]]:
        """Normalizes the input distribution values.

        Returns
        -------
        Mapping [str, np.ndarray[float]]
            An (npdf, n) array of y values for the npdf distributions

        Raises
        ------
        ValueError
            Raised if the sum under the distribution <= 0.
        """

        self._compute_ycumul()

        new_yvals = (self._yvals.T / self._ycumul[:, -1]).T
        self._ycumul = (self._ycumul.T / self._ycumul[:, -1]).T
        return {"yvals": new_yvals}

    @property
    def xvals(self) -> np.ndarray[float]:
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self) -> np.ndarray[float]:
        """Return the y-values used to do the interpolation"""
        return self._yvals

    def x_samples(self) -> np.ndarray[float]:
        """Return a set of x values that can be used to plot all the PDFs."""
        return self._xvals

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
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
        return np.sum(self._xvals**m * self._yvals, axis=1) * dx

    def _updated_ctor_param(self):
        """
        Sets the arguments as additional constructor arguments. This function is needed
        by scipy in order to copy distributions, and makes a dictionary of all parameters
        necessary to construct the distribution.
        """
        dct = super()._updated_ctor_param()
        dct["xvals"] = self._xvals
        dct["yvals"] = self._yvals
        dct["norm"] = self._norm
        dct["warn"] = self._warn
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf: int, **kwargs
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
            i.e. ``{objdata_key = ( (npdf, n), "f4" )}``

        Raises
        ------
        ValueError
            Raises an error if xvals is not provided.
        """
        if "xvals" not in kwargs:  # pragma: no cover
            raise ValueError("required argument xvals not included in kwargs")
        ngrid = np.shape(kwargs["xvals"])[-1]
        return dict(yvals=((npdf, ngrid), "f4"))

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a interpolated PDF this uses the interpolation points.

        Parameters
        ----------
        axes : Axes
            The axes to plot on. Either this or xlim must be provided.
        xlim : tuple[float, float]
            The x-axis limits. Either this or axes must be provided.
        kwargs :
            Any keyword arguments to pass to `matplotlib.axes.Axes.hist`.

        Returns
        -------
        axes : Axes
            The plot axes.
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        return plot_pdf_on_axes(axes, pdf, pdf.dist.xvals, **kw)

    @classmethod
    def add_mappings(cls) -> None:
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_vals_at_x, None)

    @classmethod
    def create_ensemble(
        self,
        xvals: ArrayLike,
        yvals: ArrayLike,
        norm: bool = True,
        warn: bool = True,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as interpolations.


        Parameters
        ----------
        xvals : ArrayLike
          The x-values used to do the interpolation, shape is n
        yvals : ArrayLike
          The y-values used to do the interpolation, shape is (npdfs, n), where
          npdfs is the number of distributions
        norm : bool, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        ancil : Optional[Mapping]
            A dictionary of metadata for the distributions, where any arrays
            have the same length as the number of distributions

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Examples
        --------

        To create an ensemble with two distributions and their associated ids:

        >>> import qp
        >>> import numpy as np
        >>> xvals= np.array([0,0.5,1,1.5,2]),
        >>> yvals = np.array([[0.01, 0.2,0.3,0.2,0.01],[0.09,0.25,0.2,0.1,0.01]])
        >>> ancil = {'ids':[5,8]}
        >>> ens = qp.interp.create_ensemble(xvals, yvals,ancil=ancil)
        >>> ens.metadata
        {'pdf_name': array([b'interp'], dtype='|S6'),
        'pdf_version': array([0]),
        'xvals': array([[0. , 0.5, 1. , 1.5, 2. ]])}

        """
        data = {"xvals": xvals, "yvals": yvals, "norm": norm, "warn": warn}
        return Ensemble(self, data, ancil)


interp = interp_gen


class interp_irregular_gen(Pdf_rows_gen):
    """Implements distributions parameterized as interpolated sets of values.

    Each distribution has its own set of x values. Interpolation is performed using
    `scipy.interpolate.interp1d`, with the default interpolation method (linear).

    Parameters
    ----------
    xvals : ArrayLike
        The x-values that are used by each distribution, with shape (npdf,n)
    yvals : ArrayLike
        The y-values that represent each distribution, with shape (npdf,n)
    norm : bool, optional
        If True, normalizes the input distribution. If False, assumes the
        given distribution is already normalized. By default True.
    warn : bool, optional
        If True, raises warnings if input is not valid PDF data (i.e. if
        data is negative). If False, no warnings are raised. By default True.


    Notes
    -----

    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +-------------------------------------+-----------+------------+
    | Function                            | Arguments | Method key |
    +-------------------------------------+-----------+------------+
    | `.irreg_interp_extract_xy_vals`     | xvals     | None       |
    +-------------------------------------+-----------+------------+

    Implementation notes:

    Inside the range xvals[:,0], xvals[:,-1] it simply takes a set of x and y values
    and uses `scipy.interpolate.interp1d` to linearly interpolate the PDF.
    Outside the range xvals[:,0], xvals[:,-1] the `pdf()` will return 0.

    The cdf() is constructed by analytically computing the cumulative
    sum at the xvals grid points and linearly interpolating between them.
    This will give a slight discrepancy with the true integral of the `pdf()`,
    but is much, much faster to evaluate.
    Outside the range xvals[:,0], xvals[:,-1] the `cdf()` will return 0 or 1, respectively

    The `ppf()` is computed by inverting the `cdf()`. `ppf(0)` gives negative infinity, and
    `ppf(1)` gives positive infinity.

    """

    # pylint: disable=protected-access

    name = "interp_irregular"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        xvals: ArrayLike,
        yvals: ArrayLike,
        norm: bool = True,
        warn: bool = True,
        *args,
        **kwargs,
    ):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : ArrayLike
          The x-values for each distribution, with shape (npdf, n), where n is
          the number of x-values
        yvals : ArrayLike
          The y-values that represent each distribution, with shape (npdf,n)
        norm : bool, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        """
        if np.shape(xvals) != np.shape(yvals):  # pragma: no cover
            raise ValueError(
                "Shape of xvals (%s) != shape of yvals (%s)"
                % (np.shape(xvals), np.shape(yvals))
            )
        self._xvals = reshape_to_pdf_size(np.asarray(xvals), -1)
        self._yvals = reshape_to_pdf_size(np.asarray(yvals), -1)

        # raise warnings if input data is not finite or pdfs are not positive
        self._warn = warn
        if self._warn:
            if not np.all(np.isfinite(self._xvals)):
                indices = np.where(np.isfinite(xvals) != True)
                warnings.warn(
                    f"The given xvals contain non-finite values for the following distributions: {indices}",
                    RuntimeWarning,
                )
            if not np.all(np.isfinite(self._yvals)):
                indices = np.where(np.isfinite(self._yvals) != True)
                warnings.warn(
                    f"There are non-finite values in the yvals for the following distributions: {indices}",
                    RuntimeWarning,
                )
            if np.any(self._yvals < 0):
                indices = np.where(self._yvals < 0)
                warnings.warn(
                    f"There are negative values in the yvals for the following distributions: {indices}",
                    RuntimeWarning,
                )

        # make sure that the xvals are sorted
        if not np.all(np.diff(self._xvals) >= 0):
            raise ValueError(
                f"Invalid xvals: The given xvals are not sorted: {self._xvals}"
            )

        self._xmin = np.min(self._xvals)
        self._xmax = np.max(self._xvals)
        # kwargs["shape"] = np.shape(xvals)[:-1]
        kwargs["shape"] = np.shape(self._xvals)

        self._norm = norm

        if self._norm:
            self._yvals = self.normalize()["yvals"]
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

        # make sure that integrals are > 0
        if np.any(self._ycumul[:, -1] < 0):
            indices = np.where(self._ycumul[:, -1] < 0)
            raise ValueError(
                f"The integral is < 0 for distributions at indices = {indices[0]}, so the distribution(s) cannot be properly normalized."
            )
        elif np.any(self._ycumul[:, -1] == 0):
            indices = np.where(self._ycumul[:, -1] == 0)
            warnings.warn(
                f"The distributions at indices = {indices[0]} have an integral of 0."
            )

    def normalize(self) -> Mapping[str, np.ndarray[float]]:
        """
        Normalize a set of 1D interpolators

        Returns
        -------
        ynorm : Mapping[str, np.ndarray[float]]
            Normalized y-vals
        """
        # def row_integral(irow):
        #    return quad(interp1d(xvals[irow], yvals[irow], **kwargs), limits[0], limits[1])[0]

        # vv = np.vectorize(row_integral)
        # integrals = vv(np.arange(xvals.shape[0]))
        integrals = np.sum(
            self._xvals[:, 1:] * self._yvals[:, 1:]
            - self._xvals[:, :-1] * self._yvals[:, 1:],
            axis=1,
        )

        # make sure that integrals are >= 0
        if np.any(integrals < 0):
            indices = np.where(integrals < 0)
            raise ValueError(
                f"The integral is < 0 for distributions at indices = {indices[0]}, so the distribution(s) cannot be properly normalized."
            )
        elif np.any(integrals == 0):
            indices = np.where(integrals == 0)
            warnings.warn(
                f"The distributions at indices = {indices[0]} have an integral of 0."
            )

        return {"yvals": (self._yvals.T / integrals).T}

    @property
    def xvals(self) -> np.ndarray[float]:
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self) -> np.ndarray[float]:
        """Return the y-valus used to do the interpolation"""
        return self._yvals

    def x_samples(self) -> np.ndarray[float]:
        """Return a set of x values that can be used to plot all the PDFs."""
        dx = np.min(np.diff(self._yvals))
        return np.arange(np.min(self._xvals), np.max(self._yvals), dx)

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        pdf = interpolate_multi_x_multi_y(
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
        dct["norm"] = self._norm
        dct["warn"] = self._warn
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf: int, **kwargs
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
            Raises an error if xvals is not provided.
        """
        if "xvals" not in kwargs:  # pragma: no cover
            raise ValueError("required argument xvals not included in kwargs")
        ngrid = np.shape(kwargs["xvals"])[-1]
        return dict(xvals=((npdf, ngrid), "f4"), yvals=((npdf, ngrid), "f4"))

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a interpolated PDF this uses the interpolation points.

        Parameters
        ----------
        axes : Axes
            The axes to plot on. Either this or xlim must be provided.
        xlim : tuple[float, float]
            The x-axis limits. Either this or axes must be provided.
        kwargs :
            Any keyword arguments to pass to matplotlib's axes.hist() method.

        Returns
        -------
        axes : Axes
            The plot axes.
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        xvals_row = pdf.dist.xvals
        return plot_pdf_on_axes(axes, pdf, xvals_row, **kw)

    @classmethod
    def add_mappings(cls) -> None:
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(irreg_interp_extract_xy_vals, None)
        cls._add_extraction_method(extract_xy_sparse, "sparse")

    @classmethod
    def create_ensemble(
        self,
        xvals: ArrayLike,
        yvals: ArrayLike,
        norm: bool = True,
        warn: bool = True,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as interpolations.

        Parameters
        ----------
        xvals : ArrayLike
          The x-values for each distribution, with shape (npdf, n), where n is
          the number of x-values
        yvals : ArrayLike
          The y-values that represent each distribution, with shape (npdf,n)
        norm : bool, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        ancil : Optional[Mapping]
            A dictionary of metadata for the distributions, where any arrays have the same length as the number of distributions.

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Examples
        --------

        To create an Ensemble with two distributions and their associated ids:

        >>> import qp
        >>> import numpy as np
        >>> xvals = np.array([[0,0.5,1,1.5,2],[0.5,0.75,1,1.25,1.5]]),
        >>> yvals = np.array([[0.01, 0.2,0.3,0.2,0.01],[0.09,0.25,0.2,0.1,0.01]])}
        >>> ancil = {'ids':[5,8]}
        >>> ens = qp.interp_irregular.create_ensemble(xvals, yvals,ancil)
        >>> ens.metadata
        {'pdf_name': array([b'interp_irregular'], dtype='|S16'),
        'pdf_version': array([0])}

        """
        data = {"xvals": xvals, "yvals": yvals, "norm": norm, "warn": warn}
        return Ensemble(self, data, ancil)


interp_irregular = interp_irregular_gen
add_class(interp_gen)
add_class(interp_irregular_gen)
