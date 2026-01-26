"""This module implements a distribution parameterization sub-class using histograms"""

from __future__ import annotations
import numpy as np

from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike


import warnings

from .hist_utils import (
    evaluate_hist_x_multi_y,
    extract_hist_values,
    extract_hist_samples,
)
from ..base import Pdf_rows_gen
from ...plotting import get_axes_and_xlims, plot_pdf_histogram_on_axes
from ...utils.array import reshape_to_pdf_size, reduce_dimensions

from ...utils.interpolation import interpolate_multi_x_y, interpolate_x_multi_y

from ...core.factory import add_class
from ...core.ensemble import Ensemble


class hist_gen(Pdf_rows_gen):
    """Implements distributions parameterized as histograms.

    By default, the input distribution is normalized. If the input data is
    already normalized, you can use the optional parameter ``norm = False``
    to skip the normalization process.

    Parameters
    ----------
    bins : ArrayLike
        The array containing the (n+1) bin boundaries
    pdfs : ArrayLike
        The array containing the (npdf, n) bin values
    norm : bool, optional
        If True, normalizes the input distribution. If False, assumes the
        given distribution is already normalized. By default True.
    warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If False, no warnings are raised. By default True.


    Notes
    -----

    There must be a minimum of 2 bins.

    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +------------------------+-----------------------------------------------------+------------+
    | Function               | Arguments                                           | Method key |
    +------------------------+-----------------------------------------------------+------------+
    | `.extract_hist_values` | bins (array of bin edges)                           | None       |
    +------------------------+-----------------------------------------------------+------------+
    | `.extract_hist_samples`| bins (array of bin edges),                          | samples    |
    |                        | size (int, optional, number of samples to generate) |            |
    +------------------------+-----------------------------------------------------+------------+

    Implementation notes:

    Inside a given bin `pdf()` will return the `hist_gen.pdfs` value.
    Outside the range of the given bins `pdf()` will return 0.

    Inside a given bin `cdf()` will use a linear interpolation across the bin.
    Outside the range of the given bins `cdf()` will return (0 or 1), respectively.

    The percentage point function `ppf()` will return negative infinity at 0 and positive
    infinity at 1.

    """

    # pylint: disable=protected-access

    name = "hist"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        bins: ArrayLike,
        pdfs: ArrayLike,
        norm: bool = True,
        warn: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Create a new distribution using the given histogram.

        Parameters
        ----------
        bins : ArrayLike
          The array containing the (n+1) bin boundaries
        pdfs : ArrayLike
          The array containing the (npdf, n) bin values
        norm : bool, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        """
        self._hbins = np.squeeze(bins)  # make sure bins is 1D
        self._nbins = self._hbins.size - 1
        self._hpdfs = reshape_to_pdf_size(np.asarray(pdfs), -1)

        # make sure that the bins are sorted
        if not np.all(np.diff(self._hbins) >= 0):
            raise ValueError(
                f"Invalid bins: The given bins are not sorted: {self._hbins}"
            )

        # raise warnings if input data is not finite or pdfs are not positive
        self._warn = warn
        if self._warn:
            if not np.all(np.isfinite(self._hbins)):
                warnings.warn(
                    f"The given bins contain non-finite values - {self._hbins}",
                    RuntimeWarning,
                )
            if not np.all(np.isfinite(pdfs)):
                indices = np.where(np.isfinite(pdfs) != True)
                warnings.warn(
                    f"There are non-finite values in the pdfs for the distributions: {indices[0]}",
                    RuntimeWarning,
                )
            if np.any(self._hpdfs < 0):
                indices = np.where(self._hpdfs < 0)
                warnings.warn(
                    f"There are negative values in the pdfs for the distributions: {indices[0]}",
                    RuntimeWarning,
                )

        # check data shapes make sense
        if np.shape(pdfs)[-1] != self._nbins:  # pragma: no cover
            raise ValueError(
                "Number of bins (%i) != number of values (%i)"
                % (self._nbins, np.shape(pdfs)[-1])
            )

        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        self._xmin = self._hbins[0]
        self._xmax = self._hbins[-1]

        # normalize the input data if norm is True
        self._norm = norm
        if self._norm:
            self._hpdfs = self.normalize()["pdfs"]

        self._hcdfs = None
        # Set support
        kwargs["shape"] = self._hpdfs.shape  # pdfs.shape
        super().__init__(*args, **kwargs)
        self._addmetadata("bins", self._hbins)
        self._addobjdata("pdfs", self._hpdfs)

    def _compute_cdfs(self):
        copy_shape = np.array(self._hpdfs.shape)
        copy_shape[-1] += 1
        self._hcdfs = np.ndarray(copy_shape)
        self._hcdfs[:, 0] = 0.0
        self._hcdfs[:, 1:] = np.cumsum(self._hpdfs * self._hbin_widths, axis=1)

    def normalize(self) -> Mapping[str, np.ndarray[float]]:
        """Normalizes the input distribution values.

        Returns
        -------
        Mapping [str, np.ndarray[float]]
            An (npdf, n) array of pdf values in the n bins for the npdf distributions

        Raises
        ------
        ValueError
            Raised if the sum under the distribution <= 0.
        """
        pdfs_2d = self._hpdfs
        sums = np.sum(pdfs_2d * self._hbin_widths, axis=1)
        if np.any(sums < 0):
            indices = np.where(sums < 0)
            raise ValueError(
                f"The distribution(s) cannot be properly normalized, the sum of the pdfs is < 0 for distributions at index = {indices[0]} "
            )
        elif np.any(sums == 0):
            indices = np.where(sums == 0)
            warnings.warn(
                f"The distributions(s) with indices {indices[0]} have an integral of 0."
            )
        return {"pdfs": (pdfs_2d.T / sums).T}

    @property
    def bins(self) -> np.ndarray[float]:
        """Return the histogram bin edges"""
        return self._hbins

    @property
    def pdfs(self) -> np.ndarray[float]:
        """Return the histogram bin values"""
        return self._hpdfs

    def x_samples(self) -> np.ndarray[float]:
        """Return a set of x values that can be used to plot all the PDFs."""
        # TODO: possibly add a bin to the left and right?
        return (self._hbins[:-1] + self._hbins[1:]) / 2

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        pdf = evaluate_hist_x_multi_y(x, row, self._hbins, self._hpdfs).ravel()

        # reduce dimension to 0 if there's only one value
        if np.shape(pdf) == (1,) and len(pdf) == 1:
            return pdf[0]
        else:
            return pdf

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._hcdfs is None:  # pragma: no cover
            self._compute_cdfs()
        return interpolate_x_multi_y(
            x, row, self._hbins, self._hcdfs, bounds_error=False, fill_value=(0.0, 1.0)
        ).ravel()

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        if self._hcdfs is None:  # pragma: no cover
            self._compute_cdfs()
        return interpolate_multi_x_y(
            x,
            row,
            self._hcdfs,
            self._hbins,
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

    def custom_generic_moment(self, m: ArrayLike) -> np.ndarray[float]:
        """Compute the mth moment"""
        m = np.asarray(m)
        dx = self._hbins[1] - self._hbins[0]
        xv = 0.5 * (self._hbins[1:] + self._hbins[:-1])
        return np.sum(xv**m * self._hpdfs, axis=1) * dx

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct["bins"] = self._hbins
        dct["pdfs"] = self._hpdfs
        dct["norm"] = self._norm
        dct["warn"] = self._warn
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf: int, **kwargs: str
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
        dict [ str, tuple [ tuple [int , int], str]]
            A dictionary with a key for the objdata, a tuple with the shape of that data,
            and the data type of the data as a string.
            i.e. ``{objdata_key = ( (npdf, n), "f4" )}``

        Raises
        ------
        ValueError
            Raises an error if the bins is not provided."""

        if "bins" not in kwargs:  # pragma: no cover
            raise ValueError("required argument 'bins' not included in kwargs")
        nbins = len(kwargs["bins"].flatten())
        return dict(pdfs=((npdf, nbins - 1), "f4"))

    @classmethod
    def plot_native(cls, pdf: Ensemble, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a histogram this shows the bin edges.

        Parameters
        ----------
        axes : Axes
            The axes to plot on. Either this or xlim must be provided.
        xlim : tuple [float, float]
            The x-axis limits. Either this or axes must be provided.
        kwargs :
            Any keyword arguments to pass to matplotlib's `matplotlib.axes.Axes.hist` method.

        Returns
        -------
        axes : Axes
            The plot axes.
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        vals = pdf.dist.pdfs[pdf.kwds["row"]]
        return plot_pdf_histogram_on_axes(axes, hist=(pdf.dist.bins, vals), **kw)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_hist_values, None)
        cls._add_extraction_method(extract_hist_samples, "samples")

    @classmethod
    def create_ensemble(
        self,
        bins: ArrayLike,
        pdfs: ArrayLike,
        norm: bool = True,
        warn: bool = True,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as histograms.


        Parameters
        ----------
        bins : ArrayLike
          The array containing the (n+1) bin boundaries
        pdfs : ArrayLike
          The array containing the (npdf, n) bin values
        norm : bool, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.
        warn : bool, optional
            If True, raises warnings if input is not valid PDF data (i.e. if
            data is negative). If False, no warnings are raised. By default True.
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have
            length npdf, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Examples
        --------

        To create an Ensemble with two distributions and an 'ancil' table that
        provides ids for the distributions, you can use the following code:

        >>> import qp
        >>> import numpy as np
        >>> bins= [0,1,2,3,4,5]
        >>> pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ancil = {'ids': [105, 108]}
        >>> ens = qp.hist.create_ensemble(bins,pdfs,ancil=ancil)
        >>> ens.metadata
        {'pdf_name': array([b'hist'], dtype='|S4'),
        'pdf_version': array([0]),
        'bins': array([[0, 1, 2, 3, 4, 5]])}

        """

        data = {"bins": bins, "pdfs": pdfs, "norm": norm, "warn": warn}
        return Ensemble(self, data, ancil)


hist = hist_gen
add_class(hist_gen)
