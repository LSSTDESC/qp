"""This module implements a PDT distribution sub-class using histograms
"""


import numpy as np

from scipy.stats import rv_continuous

from qp.pdf_gen import Pdf_rows_gen
from qp.conversion_funcs import extract_hist_values, extract_hist_samples
from qp.plotting import get_axes_and_xlims, plot_pdf_histogram_on_axes
from qp.utils import evaluate_hist_x_multi_y, evaluate_unfactored_hist_x_multi_y,\
     interpolate_unfactored_multi_x_y, interpolate_unfactored_x_multi_y,\
     interpolate_multi_x_y, interpolate_x_multi_y, reshape_to_pdf_size
from qp.test_data import XBINS, HIST_DATA, TEST_XVALS, NSAMPLES
from qp.factory import add_class



class hist_gen(Pdf_rows_gen):
    """Histogram based distribution

    Notes
    -----
    This implements a PDF using a set of histogramed values.

    """
    # pylint: disable=protected-access

    name = 'hist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, bins, pdfs, *args, **kwargs):
        """
        Create a new distribution using the given histogram
        Parameters
        ----------
        bins : array_like
          The second containing the (n+1) bin boundaries
        """
        self._hbins = np.asarray(bins)
        self._nbins = self._hbins.size - 1
        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        if pdfs.shape[-1] != self._nbins: # pragma: no cover
            raise ValueError("Number of bins (%i) != number of values (%i)" % (self._nbins, pdfs.shape[-1]))

        check_input = kwargs.pop('check_input', True)
        if check_input:
            pdfs_2d = reshape_to_pdf_size(pdfs, -1)
            sums = np.sum(pdfs_2d*self._hbin_widths, axis=1)
            self._hpdfs = (pdfs_2d.T / sums).T
        else: #pragma: no cover
            self._hpdfs = reshape_to_pdf_size(pdfs, -1)
        self._hcdfs = None
        # Set support
        kwargs['a'] = self.a = self._hbins[0]
        kwargs['b'] = self.b = self._hbins[-1]
        kwargs['shape'] = pdfs.shape[:-1]
        super(hist_gen, self).__init__(*args, **kwargs)
        self._addmetadata('bins', self._hbins)
        self._addobjdata('pdfs', self._hpdfs)


    def _compute_cdfs(self):
        copy_shape = np.array(self._hpdfs.shape)
        copy_shape[-1] += 1
        self._hcdfs = np.ndarray(copy_shape)
        self._hcdfs[:,0] = 0.
        self._hcdfs[:,1:] = np.cumsum(self._hpdfs * self._hbin_widths, axis=1)

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
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:  #pragma: no cover
            # x values and row values factorize
            return evaluate_hist_x_multi_y(xr, rr, self._hbins, self._hpdfs)
        return evaluate_unfactored_hist_x_multi_y(xr, rr, self._hbins, self._hpdfs)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._hcdfs is None: #pragma: no cover
            self._compute_cdfs()
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._hbins, self._hcdfs[rr]).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._hbins, self._hcdfs)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        if self._hcdfs is None: #pragma: no cover
            self._compute_cdfs()
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:  #pragma: no cover
            return interpolate_multi_x_y(xr, self._hcdfs[rr], self._hbins).reshape(x.shape)
        return interpolate_unfactored_multi_x_y(xr, rr, self._hcdfs, self._hbins)

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(hist_gen, self)._updated_ctor_param()
        dct['bins'] = self._hbins
        dct['pdfs'] = self._hpdfs
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a histogram this shows the bin edges
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        vals = pdf.dist.pdfs[pdf.kwds['row']]
        return plot_pdf_histogram_on_axes(axes, hist=(pdf.dist.bins, vals), **kw)


    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_hist_values, None)
        cls._add_extraction_method(extract_hist_samples, "samples")


hist = hist_gen.create

hist_gen.test_data = dict(hist=dict(gen_func=hist, ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),\
                                             convert_data=dict(bins=XBINS), atol_diff=1e-1, atol_diff2=1e-1, test_xvals=TEST_XVALS),
                               hist_samples=dict(gen_func=hist, ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),\
                                                     convert_data=dict(bins=XBINS, method='samples',\
                                                                           size=NSAMPLES),\
                                                     atol_diff=1e-1, atol_diff2=1e-1,\
                                                     test_xvals=TEST_XVALS, do_samples=True))
add_class(hist_gen)
