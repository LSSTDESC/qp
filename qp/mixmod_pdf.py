"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""

import numpy as np

from scipy.stats import rv_continuous
from scipy import stats as sps


from qp.pdf_gen import Pdf_rows_gen
from qp.conversion_funcs import extract_mixmod_fit_samples
from qp.test_data import WEIGHT_MIXMOD, MEAN_MIXMOD, STD_MIXMOD, TEST_XVALS
from qp.factory import add_class
from qp.utils import reshape_to_pdf_size, interpolate_unfactored_multi_x_y,\
    interpolate_multi_x_y

class mixmod_gen(Pdf_rows_gen):
    """Mixture model based distribution

    Notes
    -----
    This implements a PDF using a Gaussian Mixture model
    """
    # pylint: disable=protected-access

    name = 'mixmod'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, means, stds, weights, *args, **kwargs):
        """
        Create a new distribution using the given histogram

        Parameters
        ----------
        means : array_like
            The means of the Gaussians
        stds:  array_like
            The standard deviations of the Gaussians
        weights : array_like
            The weights to attach to the Gaussians
        """
        self._means = reshape_to_pdf_size(means, -1)
        self._stds = reshape_to_pdf_size(stds, -1)
        self._weights = reshape_to_pdf_size(weights, -1)
        kwargs['shape'] = means.shape[:-1]
        self._ncomps = means.shape[-1]
        super(mixmod_gen, self).__init__(*args, **kwargs)
        self._addobjdata('weights', self._weights)
        self._addobjdata('stds', self._stds)
        self._addobjdata('means', self._means)

    @property
    def weights(self):
        """Return weights to attach to the Gaussians"""
        return self._weights

    @property
    def means(self):
        """Return means of the Gaussians"""
        return self._means

    @property
    def stds(self):
        """Return standard deviations of the Gaussians"""
        return self._stds

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        #factored, xr, rr, _ = self._sliceargs(x, row)
        #if factored:  #pragma: no cover
        #    return (np.expand_dims(self.weights[rr], -1) *\
        #                sps.norm(loc=np.expand_dims(self._means[rr], -1),\
        #                             scale=np.expand_dims(self._stds[rr], -1)).pdf(np.expand_dims(xr, 0))).sum(axis=1).reshape(x.shape)
        if np.ndim(x) > 1:
            x = np.expand_dims(x, -2)
        return (self.weights[row].swapaxes(-2,-1) * 
                sps.norm(loc=self._means[row].swapaxes(-2,-1),
                         scale=self._stds[row].swapaxes(-2,-1)).pdf(x)).sum(axis=1)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        #factored, xr, rr, _ = self._sliceargs(x, row)
        #if factored:  #pragma: no cover
        #    return (np.expand_dims(self.weights[rr], -1) *\
        #                sps.norm(loc=np.expand_dims(self._means[rr], -1),\
        #                            scale=np.expand_dims(self._stds[rr], -1)).cdf(np.expand_dims(xr, 0))).sum(axis=1).reshape(x.shape)
        if np.ndim(x) > 1:
            x = np.expand_dims(x, -2)
        return (self.weights[row].swapaxes(-2,-1) * 
                sps.norm(loc=self._means[row].swapaxes(-2,-1),
                         scale=self._stds[row].swapaxes(-2,-1)).cdf(x)).sum(axis=1)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        min_val = np.min(self._means - 6*self._stds)
        max_val = np.max(self._means + 6*self._stds)
        grid = np.linspace(min_val, max_val, 201)
        cdf_vals = self.cdf(grid, row)
        #factored, xr, rr, _ = self._sliceargs(x, row)
        #if factored:  #pragma: no cover
        #    return interpolate_multi_x_y(xr, cdf_vals, grid, bounds_error=False,
        #                                 fill_value=(0.,1.)).reshape(x.shape)
        return interpolate_unfactored_multi_x_y(x, row, cdf_vals, grid,
                                                bounds_error=False, fill_value=(min_val, max_val))



    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(mixmod_gen, self)._updated_ctor_param()
        dct['means'] = self._means
        dct['stds'] = self._stds
        dct['weights'] = self._weights
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_mixmod_fit_samples, None)

    @classmethod
    def make_test_data(cls):
        cls.test_data = dict(mixmod=dict(gen_func=mixmod,\
                                         ctor_data=dict(weights=WEIGHT_MIXMOD,\
                                                        means=MEAN_MIXMOD,\
                                                        stds=STD_MIXMOD),\
                                         convert_data=dict(), test_xvals=TEST_XVALS,
                                         atol_diff2=1.))


mixmod = mixmod_gen.create

add_class(mixmod_gen)
