"""This module implements a PDT distribution sub-class using histograms
"""

import numpy as np

from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike

from .hist_utils import (
    evaluate_hist_x_multi_y,
    extract_hist_values,
    extract_hist_samples,
)
from ..base import Pdf_rows_gen
from ...plotting import get_axes_and_xlims, plot_pdf_histogram_on_axes
from ...utils.array import reshape_to_pdf_size

from ...utils.interpolation import interpolate_multi_x_y, interpolate_x_multi_y

from ...test_data import XBINS, HIST_DATA, TEST_XVALS, NSAMPLES
from ...core.factory import add_class
from ...core.ensemble import Ensemble


class hist_gen(Pdf_rows_gen):
    """Implements distributions parameterized as histograms.

    By default, the input distribution is normalized. If the input data is
    already normalized, you can use the optional parameter ``check_input = False``
    to skip the normalization process.

    Parameters
    ----------
    bins : `arraylike`
        The array containing the (n+1) bin boundaries
    pdfs : `arraylike`
        The array containing the (npdf, n) bin values
    check_input : `bool`, optional
        If True, normalizes the input distribution. If False, assumes the
        given distribution is already normalized. By default True.


    Attributes
    ----------
    bins : `ndarray`
        The array containing the (n+1) bin boundaries
    pdfs : `ndarray`
        The array containing the (npdf, n) PDF values in the bins


    Methods
    -------
    create_ensemble(data,ancil)
        Create an Ensemble with this parameterization.
    plot_native(xlim,axes,**kwargs)
        Create a plot of a distribution with this parameterization.

    Notes
    -----

    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +------------------------+-----------------------------------------------------+------------+
    | Function               | Arguments                                           | Method key |
    +------------------------+-----------------------------------------------------+------------+
    | `extract_hist_values`  | bins (array of bin edges)                           | None       |
    +------------------------+-----------------------------------------------------+------------+
    | `extract_hist_samples` | bins (array of bin edges),                          | samples    |
    |                        | size (int, optional, number of samples to generate) |            |
    +------------------------+-----------------------------------------------------+------------+

    Implementation notes:

    Inside a given bin `pdf()` will return the `hist_gen.pdfs` value.
    Outside the range of the given bins `pdf()` will return 0.

    Inside a given bin `cdf()` will use a linear interpolation across the bin.
    Outside the range of the given bins `cdf()` will return (0 or 1), respectively.

    The percentage point function `ppf()` will return bins[0] at 0, and
    will return bins[-1] at 1.
    """

    # pylint: disable=protected-access

    name = "hist"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, bins: ArrayLike, pdfs: ArrayLike, *args, **kwargs):
        """
        Create a new distribution using the given histogram.

        Parameters
        ----------
        bins : `array_like`
          The array containing the (n+1) bin boundaries
        pdfs : `array_like`
          The array containing the (npdf, n) bin values
        check_input : `bool`, optional
            If True, normalizes the input distribution. If False, assumes the
            given distribution is already normalized. By default True.
        """
        self._hbins = np.asarray(bins)
        self._nbins = self._hbins.size - 1
        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        self._xmin = self._hbins[0]
        self._xmax = self._hbins[-1]
        if np.shape(pdfs)[-1] != self._nbins:  # pragma: no cover
            raise ValueError(
                "Number of bins (%i) != number of values (%i)"
                % (self._nbins, np.shape(pdfs)[-1])
            )

        check_input = kwargs.pop("check_input", True)
        if check_input:
            pdfs_2d = reshape_to_pdf_size(pdfs, -1)
            sums = np.sum(pdfs_2d * self._hbin_widths, axis=1)
            self._hpdfs = (pdfs_2d.T / sums).T
        else:  # pragma: no cover
            self._hpdfs = reshape_to_pdf_size(pdfs, -1)
        self._hcdfs = None
        # Set support
        kwargs["shape"] = pdfs.shape[:-1]
        super().__init__(*args, **kwargs)
        self._addmetadata("bins", self._hbins)
        self._addobjdata("pdfs", self._hpdfs)

    def _compute_cdfs(self):
        copy_shape = np.array(self._hpdfs.shape)
        copy_shape[-1] += 1
        self._hcdfs = np.ndarray(copy_shape)
        self._hcdfs[:, 0] = 0.0
        self._hcdfs[:, 1:] = np.cumsum(self._hpdfs * self._hbin_widths, axis=1)

    @property
    def bins(self):
        """Return the histogram bin edges"""
        return self._hbins

    @property
    def pdfs(self):
        """Return the histogram bin values"""
        return self._hpdfs

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        return evaluate_hist_x_multi_y(x, row, self._hbins, self._hpdfs).ravel()

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

    def custom_generic_moment(self, m):
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
        return dct

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        if "bins" not in kwargs:  # pragma: no cover
            raise ValueError("required argument 'bins' not included in kwargs")
        nbins = len(kwargs["bins"].flatten())
        return dict(pdfs=((npdf, nbins - 1), "f4"))

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distribution

        For a histogram this shows the bin edges.

        Parameters
        ----------
        axes : `matplotlib.axes`
            The axes to plot on. Either this or xlim must be provided.
        xlim : (float, float)
            The x-axis limits. Either this or axes must be provided.
        kwargs :
            Any keyword arguments to pass to matplotlib's axes.hist() method.

        Returns
        -------
        axes : `matplotlib.axes`
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
        self, data: Mapping, ancil: Optional[Mapping] = None
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as histograms.

        Input data format:
        data = {'bins': array_like, 'pdfs': array_like}, where bins are the bin
        edges, and so should be of shape (n+1,), and data is the value in those
        bins, so should have length of n and shape (npdfs, n), where npdfs is the
        number of distributions. The value of 'pdfs' can have multiple rows, where
        each row is a distribution.

        You can also add `check_input` as a key to the dictionary. By default
        this is True, but if your input is already normalized, you can pass
        `check_input` as False.



        Parameters
        ----------
        data : Mapping
            The dictionary of data for the distributions.
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have
            the same length as the number of distributions, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Example
        -------

        To create an Ensemble with two distributions and an 'ancil' table that
        provides ids for the distributions, you can use the following code:

        >>> import qp
        >>> import numpy as np
        >>> data = {'bins': [0,1,2,3,4,5],'pdfs': np.array([[0,0.1,0.1,0.4,0.2],
        ...         [0.05,0.09,0.2,0.3,0.15]])}
        >>> ancil = {'ids': [105, 108]}
        >>> ens = qp.hist.create_ensemble(data,ancil)
        >>> ens.metadata()
        {'pdf_name': array([b'hist'], dtype='|S4'),
        'pdf_version': array([0]),
        'bins': array([[0, 1, 2, 3, 4, 5]])}

        """

        return Ensemble(self, data, ancil)

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


hist = hist_gen
add_class(hist_gen)
