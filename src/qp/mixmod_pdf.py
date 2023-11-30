"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""

import numpy as np
from scipy import stats as sps
from scipy.stats import rv_continuous

from qp.conversion_funcs import extract_mixmod_fit_samples
from qp.factory import add_class
from qp.pdf_gen import Pdf_rows_gen
from qp.test_data import MEAN_MIXMOD, STD_MIXMOD, TEST_XVALS, WEIGHT_MIXMOD
from qp.utils import get_eval_case, interpolate_multi_x_y, reshape_to_pdf_size


class mixmod_gen(Pdf_rows_gen):
    """Mixture model based distribution

    Notes
    -----
    This is a base class for implementing PDFs using a mixture model. 
    Classes implementing mixture models with specific basis functions
    need to define the pdf and cdf.

    The relevant data members are:

    means:  (npdf, ncomp) means of the basis functions
    stds:  (npdf, ncomp) standard deviations of the basis functions
    weights: (npdf, ncomp) weights for the basis functions

    The pdf() and cdf() are exact, and are computed as a weighted sum of
    the pdf() and cdf() of the component basis functions.

    The ppf() is computed by computing the cdf() values on a fixed
    grid and interpolating the inverse function.
    """

    # pylint: disable=protected-access

    _support_mask = rv_continuous._support_mask

    name = 'mixmod'
    version = 0

    def __init__(self, gen_func, weights, data, ancil=None, *args, **kwargs):
        """
        Create a new distribution using the given histogram

        Parameters
        ----------
        means : array_like
            The means of the basis functions
        stds:  array_like
            The standard deviations of the basis functions
        weights : array_like
            The weights to attach to the Gaussians. Weights should sum up to one.
            If not, the weights are interpreted as relative weights.
        """
        self._gen_func = gen_func
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self._gen_class = type(self._gen_obj)
        self._data = data

        self._scipy_version_warning()
        self._weights = reshape_to_pdf_size(weights, -1)
        for key in self._data.keys():
            self._data[key] = reshape_to_pdf_size(self._data[key],-1)
        kwargs['shape'] = weights.shape[:-1]
        self._ncomps = weights.shape[-1]
        super().__init__(*args, **kwargs)
        if np.any(self._weights<0):
            raise ValueError('All weights need to be larger than zero')
        self._weights = self._weights/self._weights.sum(axis=1)[:,None]
        self._addobjdata('weights', self._weights)

    def _scipy_version_warning(self):
        import scipy  # pylint: disable=import-outside-toplevel

        scipy_version = scipy.__version__
        vtuple = scipy_version.split(".")
        if int(vtuple[0]) > 1 or int(vtuple[1]) > 7:
            return
        raise DeprecationWarning(
            f"Mixmod_gen will not work correctly with scipy version < 1.8.0, you have {scipy_version}"
        )  # pragma: no cover

    @property
    def weights(self):
        """Return weights to attach to the basis functions"""
        return self._weights

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        if np.ndim(x) > 1:  # pragma: no cover
            x = np.expand_dims(x, -2)
        data_swap=dict()
        for key in self._data.keys():
            data_swap[key] = self._data[key][row].swapaxes(-2,-1)
        return (self.weights[row].swapaxes(-2,-1) *
                self._gen_func(**data_swap).pdf(x)).sum(axis=0)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if np.ndim(x) > 1:  # pragma: no cover
            x = np.expand_dims(x, -2)
        data_swap=dict()
        for key in self._data.keys():
            data_swap[key] = self._data[key][row].swapaxes(-2,-1)
        return (self.weights[row].swapaxes(-2,-1) *
                self._gen_func(**data_swap).cdf(x)).sum(axis=0)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        min_val = np.min(self._means - 6 * self._stds)
        max_val = np.max(self._means + 6 * self._stds)
        grid = np.linspace(min_val, max_val, 201)
        case_idx, _, rr = get_eval_case(x, row)
        if case_idx == 1:
            cdf_vals = self.cdf(grid, rr)
        elif case_idx == 3:
            cdf_vals = self.cdf(grid, np.expand_dims(rr, -1))
        else:  # pragma: no cover
            raise ValueError(
                f"Opps, we handle this kind of input to mixmod._ppf {case_idx}"
            )
        return interpolate_multi_x_y(
            x, row, cdf_vals, grid, bounds_error=False, fill_value=(min_val, max_val)
        ).ravel()

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        # for key in self._data.keys():
        #     dct[key] = self._data[key]
        dct['weights'] = self._weights
        dct['data'] = self._data
        dct['gen_func'] = self._gen_func
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
        if "means" not in kwargs:  # pragma: no cover
            raise ValueError("required argument means not included in kwargs")

        ncomp = np.shape(kwargs["means"])[-1]
        return dict(
            means=((npdf, ncomp), "f4"),
            stds=((npdf, ncomp), "f4"),
            weights=((npdf, ncomp), "f4"),
        )

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_mixmod_fit_samples, None)

    @classmethod
    def make_test_data(cls):
        """Make data for unit tests"""
        cls.test_data = dict(
            mixmod=dict(
                gen_func=mixmod,
                ctor_data=dict(
                    weights=WEIGHT_MIXMOD, means=MEAN_MIXMOD, stds=STD_MIXMOD
                ),
                convert_data={},
                test_xvals=TEST_XVALS,
                atol_diff2=1.0,
            )
        )

mixmod = mixmod_gen.create

add_class(mixmod_gen)

