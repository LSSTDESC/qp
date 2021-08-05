"""This module implements a PDT distribution sub-class using interpolated quantiles
"""

import sys
import numpy as np

from scipy.stats import rv_continuous
from scipy.interpolate import interp1d

from qp.pdf_gen import Pdf_rows_gen

from qp.conversion_funcs import extract_quantiles
from qp.plotting import get_axes_and_xlims, plot_pdf_quantiles_on_axes
from qp.utils import evaluate_hist_multi_x_multi_y,\
     interpolate_multi_x_y, interpolate_x_multi_y,\
     reshape_to_pdf_size
from qp.test_data import QUANTS, QLOCS, TEST_XVALS
from qp.factory import add_class


epsilon = sys.float_info.epsilon

def pad_quantiles(quants, locs):
    """Pad the quantiles and locations used to build a quantile representation

    Paramters
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
    if quants[-1] < 1.:
        pad_hi = True
        n_out += 1
    else:
        pad_hi = False
    if n_out == n_vals:
        return quants, locs
    quants_out = np.zeros((n_out), quants.dtype)
    locs_out = np.zeros((locs.shape[0], n_out), quants.dtype)
    quants_out[offset_lo:n_vals+offset_lo] = quants
    locs_out[:,offset_lo:n_vals+offset_lo] = locs
    if pad_lo:
        locs_out[:, 0] = locs[:, 0] - quants[0] * (locs[:, 1] - locs[:, 0]) / (quants[1] - quants[0])

    if pad_hi:
        quants_out[-1] = 1.
        locs_out[:, -1] = locs[:, -1] - (1. - quants[-1]) * (locs[:, -2] - locs[:, -1]) / (quants[-1] - quants[-2])

    return quants_out, locs_out


class quant_gen(Pdf_rows_gen):
    """Quantile based distribution, where the PDF is defined piecewise from the quantiles

    Notes
    -----
    This implements a CDF by interpolating a set of quantile values

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the CDF
    """
    # pylint: disable=protected-access

    name = 'quant'
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
        kwargs['shape'] = locs.shape[:-1]

        self._xmin = np.min(locs)
        self._xmax = np.max(locs)

        super(quant_gen, self).__init__(*args, **kwargs)

        locs_2d = reshape_to_pdf_size(locs, -1)
        check_input = kwargs.pop('check_input', True)
        if check_input:
            quants, locs_2d = pad_quantiles(quants, locs_2d)

        self._quants = np.asarray(quants)
        self._nquants = self._quants.size
        if locs_2d.shape[-1] != self._nquants:  # pragma: no cover
            raise ValueError("Number of locations (%i) != number of quantile values (%i)" % (self._nquants, locs_2d.shape[-1]))
        self._locs = locs_2d
        self._valatloc = None
        self._addmetadata('quants', self._quants)
        self._addobjdata('locs', self._locs)


    def _compute_valatloc(self):
        self._valatloc = (self._quants[1:] - self._quants[0:-1])/(self._locs[:,1:] - self._locs[:,0:-1])


    @property
    def quants(self):
        """Return quantiles used to build the CDF"""
        return self._quants

    @property
    def locs(self):
        """Return the locations at which those quantiles are reached"""
        return self._locs

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._valatloc is None:  # pragma: no cover
            self._compute_valatloc()
        return evaluate_hist_multi_x_multi_y(x, row, self._locs, self._valatloc)


    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_multi_x_y(x, row, self._locs, self._quants,
                                     bounds_error=False, fill_value=(0., 1))

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_x_multi_y(x, row, self._quants, self._locs,
                                     bounds_error=False, fill_value=(self._xmin, self._xmax))

    def _updated_ctor_param(self):
        """
        Set the bins as additional construstor argument
        """
        dct = super(quant_gen, self)._updated_ctor_param()
        dct['quants'] = self._quants
        dct['locs'] = self._locs
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a quantile this shows the quantiles points
        """
        axes, xlim, kw = get_axes_and_xlims(**kwargs)
        xvals = np.linspace(xlim[0], xlim[1], kw.pop('npts', 101))
        locs = np.squeeze(pdf.dist.locs[pdf.kwds['row']])
        quants = np.squeeze(pdf.dist.quants)
        yvals = np.squeeze(pdf.pdf(xvals))
        return plot_pdf_quantiles_on_axes(axes, xvals, yvals, quantiles=(quants, locs), **kw)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_quantiles, None)


quant = quant_gen.create

quant_gen.test_data = dict(quant=dict(gen_func=quant, ctor_data=dict(quants=QUANTS, locs=QLOCS),\
                                          convert_data=dict(quants=QUANTS), test_xvals=TEST_XVALS))

add_class(quant_gen)




class quant_piecewise_gen(Pdf_rows_gen):
    """Quantile based distribution, where the PDF is defined piecewise from the quantiles

    Notes
    -----
    This implements a CDF by interpolating a set of quantile values

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the CDF
    """
    # pylint: disable=protected-access

    name = 'quant_piecewise'
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
        kwargs['shape'] = locs.shape[:-1]
        self._xmin = np.min(locs)
        self._xmax = np.max(locs)

        super(quant_piecewise_gen, self).__init__(*args, **kwargs)

        check_input = kwargs.pop('check_input', True)
        locs_2d = reshape_to_pdf_size(locs, -1)
        if check_input:
            quants, locs_2d = pad_quantiles(quants, locs_2d)

        self._quants = np.asarray(quants)
        self._nquants = self._quants.size
        if locs_2d.shape[-1] != self._nquants:  # pragma: no cover
            raise ValueError("Number of locations (%i) != number of quantile values (%i)" % (self._nquants, locs_2d.shape[-1]))
        self._locs = locs_2d
        self._valatloc = None
        self._addmetadata('quants', self._quants)
        self._addobjdata('locs', self._locs)


    def _compute_valatloc(self):
        self._valatloc = (self._quants[1:] - self._quants[0:-1])/(self._locs[:,1:] - self._locs[:,0:-1])


    @property
    def quants(self):
        """Return quantiles used to build the CDF"""
        return self._quants

    @property
    def locs(self):
        """Return the locations at which those quantiles are reached"""
        return self._locs

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._valatloc is None:  # pragma: no cover
            self._compute_valatloc()
        return evaluate_hist_multi_x_multi_y(x, row, self._locs, self._valatloc)


    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_multi_x_y(x, row, self._locs, self._quants,
                                     bounds_error=False, fill_value=(0., 1))

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        return interpolate_x_multi_y(x, row, self._quants, self._locs,
                                     bounds_error=False, fill_value=(self._xmin, self._xmax))
    

    def _updated_ctor_param(self):
        """
        Set the bins as additional construstor argument
        """
        dct = super(quant_piecewise_gen, self)._updated_ctor_param()
        dct['quants'] = self._quants
        dct['locs'] = self._locs
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a quantile this shows the quantiles points
        """
        axes, xlim, kw = get_axes_and_xlims(**kwargs)
        xvals = np.linspace(xlim[0], xlim[1], kw.pop('npts', 101))
        locs = np.squeeze(pdf.dist.locs[pdf.kwds['row']])
        quants = np.squeeze(pdf.dist.quants)
        yvals = np.squeeze(pdf.pdf(xvals))
        return plot_pdf_quantiles_on_axes(axes, xvals, yvals, quantiles=(quants, locs), **kw)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_quantiles, None)

    @classmethod
    def make_test_data(cls):
        """ Make data for unit tests """
        cls.test_data = dict(quant_piecewise=dict(gen_func=quant_piecewise, ctor_data=dict(quants=QUANTS, locs=QLOCS),\
                                                  convert_data=dict(quants=QUANTS), test_xvals=TEST_XVALS))

quant_piecewise = quant_piecewise_gen.create

add_class(quant_piecewise_gen)
