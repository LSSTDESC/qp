"""This module implements continous distributions generators that inherit from the
`scipy.stats.rv_continuous` class

If you would like to add a sub-class, please read the instructions on subclassing
here:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html


Open questions:
1) At this time the normalization is not enforced for many of the PDF types.  It is assumed that
the user values give correct normalization.  We should think about this more.

2) At this time for most of the distributions, only the _pdf function is overridden.  This is all that
is required to inherit from `scipy.stats.rv_continuous`; however, providing implementations of some of
_logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf could speed the code up a lot in some cases.

"""
import sys

import numpy as np
from numpy import asarray

from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from qp.utils import reshape_to_pdf_size, reshape_to_pdf_shape
from qp.dict_utils import get_val_or_default, set_val_or_default, pretty_print
from qp.plotting import plot_dist_pdf


class Pdf_gen:
    """Interface class to extend `scipy.stats.rv_continuous` with
    information needed for `qp`

    Notes
    -----
    Metadata are elements that are the same for all the PDFs
    These include the name and version of the PDF generation class, and
    possible data such as the bin edges used for histogram representations

    Object data are elements that differ for each PDFs
    """

    _reader_map = {}
    _creation_map = {}
    _extraction_map = {}

    def __init__(self, *args, **kwargs):
        """C'tor"""
        # pylint: disable=unused-argument
        self._metadata = {}
        self._objdata = {}
        self._addclassmetadata(type(self))

    def _addclassmetadata(self, cls):
        self._metadata["pdf_name"] = np.array([cls.name.encode()])
        self._metadata["pdf_version"] = np.array([cls.version])

    def _addmetadata(self, key, val):
        self._metadata[key] = np.expand_dims(val, 0)

    def _addobjdata(self, key, val):
        self._objdata[key] = val

    def _clearobjdata(self):
        self._objdata = {}

    @property
    def metadata(self):
        """Return the metadata for this set of PDFs"""
        return self._metadata

    @property
    def objdata(self):
        """Return the object data for this set of PDFs"""
        return self._objdata

    @classmethod
    def creation_method(cls, method=None):
        """Return the method used to create a PDF of this type"""
        return get_val_or_default(cls._creation_map, method)

    @classmethod
    def extraction_method(cls, method=None):
        """Return the method used to extract data to create a PDF of this type"""
        return get_val_or_default(cls._extraction_map, method)

    @classmethod
    def reader_method(cls, version=None):
        """Return the method used to convert data read from a file PDF of this type"""
        return get_val_or_default(cls._reader_map, version)

    @classmethod
    def add_method_dicts(cls):
        """Add empty method dicts"""
        cls._reader_map = {}
        cls._creation_map = {}
        cls._extraction_map = {}

    @classmethod
    def _add_creation_method(cls, the_func, method):
        """Add a method used to create a PDF of this type"""
        set_val_or_default(cls._creation_map, method, the_func)

    @classmethod
    def _add_extraction_method(cls, the_func, method):
        """Add a method used to extract data to create a PDF of this type"""
        set_val_or_default(cls._extraction_map, method, the_func)

    @classmethod
    def _add_reader_method(cls, the_func, version):  # pragma: no cover
        """Add a method used to convert data read from a file PDF of this type"""
        set_val_or_default(cls._reader_map, version, the_func)

    @classmethod
    def print_method_maps(cls, stream=sys.stdout):
        """Print the maps showing the methods"""
        pretty_print(cls._creation_map, ["Create  "], stream=stream)
        pretty_print(cls._extraction_map, ["Extract "], stream=stream)
        pretty_print(cls._reader_map, ["Reader  "], stream=stream)

    @classmethod
    def create_gen(cls, **kwds):
        """Create and return a `scipy.stats.rv_continuous` object using the
        keyword arguemntets provided"""
        kwds_copy = kwds.copy()
        name = kwds_copy.pop("name", "dist")
        return (cls(name=name), kwds_copy)

    @classmethod
    def create(cls, **kwds):
        """Create and return a `scipy.stats.rv_frozen` object using the
        keyword arguemntets provided"""
        # pylint: disable=not-callable
        obj, kwds_freeze = cls.create_gen(**kwds)
        return obj(**kwds_freeze)

    @classmethod
    def plot(cls, pdf, **kwargs):
        """Plot the pdf as a curve"""
        return plot_dist_pdf(pdf, **kwargs)

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        This defaults to plotting it as a curve, but this can be overwritten
        """
        return plot_dist_pdf(pdf, **kwargs)

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        """Return kwds necessary to create 'empty' hdf5 file with npdf entries
        for iterative writeout
        """
        raise NotImplementedError()  # pragma: no cover


class rv_frozen_func(rv_continuous_frozen):
    """Trivial extention of `scipy.stats.rv_frozen`
    that includes the number of PDFs it represents
    """

    def __init__(self, dist, *args, **kwds):
        """C'tor

        Parameters
        ----------
        dist : `scipy.stats.rv_continuous`
            The underlying distribution
        npdf : `int`
            The number of PDFs this object represents
        """
        super().__init__(dist, *args, **kwds)
        array_list = [np.array(val) for val in self.kwds.values()]
        bc = np.broadcast(array_list)
        ss = bc.shape
        if len(ss) < 2:
            self._shape = 1
        else:
            self._shape = ss[1:-1]
        self._npdf = np.prod(self._shape).astype(int)
        self._ndim = np.size(self._shape)

    @property
    def ndim(self):
        """Return the number of dimensions of PDFs in this ensemble"""
        return self._ndim

    @property
    def shape(self):
        """Return the shape of the set of PDFs this object represents"""
        return self._shape

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf

    def histogramize(self, bins):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        bins: ndarray, float, optional
            Array of N+1 endpoints of N bins

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        cdf_vals = reshape_to_pdf_size(self.cdf(bins), -1)
        bin_vals = cdf_vals[:, 1:] - cdf_vals[:, 0:-1]
        return (bins, reshape_to_pdf_shape(bin_vals, self._shape, bins.size - 1))


class rv_frozen_rows(rv_continuous_frozen):
    """Trivial extention of `scipy.stats.rv_frozen`
    that to use when we want to have a collection
    of distribution of objects such as histograms or splines,
    where each object represents a single distribtuion
    """

    def __init__(self, dist, shape, *args, **kwds):
        """C'tor"""
        self._shape = shape
        self._npdf = np.prod(shape).astype(int)
        self._ndim = np.size(shape)
        if self._npdf is not None:
            kwds.setdefault(
                "row", np.expand_dims(np.arange(self._npdf).reshape(self._shape), -1)
            )
        super().__init__(dist, *args, **kwds)

    @property
    def ndim(self):
        """Return the number of dimensions of PDFs in this ensemble"""
        return self._ndim

    @property
    def shape(self):
        """Return the shape of the set of PDFs this object represents"""
        return self._shape

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf

    def histogramize(self, bins):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        bins: ndarray, float, optional
            Array of N+1 endpoints of N bins

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        cdf_vals = reshape_to_pdf_size(self.cdf(bins), -1)
        bin_vals = cdf_vals[:, 1:] - cdf_vals[:, 0:-1]
        return (bins, reshape_to_pdf_shape(bin_vals, self._shape, bins.size - 1))


class Pdf_rows_gen(rv_continuous, Pdf_gen):
    """Class extend `scipy.stats.rv_continuous` with
    information needed for `qp` when we want to have a collection
    of distribution of objects such as histograms or splines,
    where each object represents a single distribtuion

    """

    def __init__(self, *args, **kwargs):
        """C'tor"""
        self._shape = kwargs.pop("shape", (1))
        self._npdf = np.prod(self._shape).astype(int)
        super().__init__(*args, **kwargs)

    @property
    def shape(self):
        """Return the shape of the set of PDFs this object represents"""
        return self._shape

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf

    @staticmethod
    def _sliceargs(x, row, *args):  # pragma: no cover
        if np.size(x) == 1 or np.size(row) == 1:
            return False, x, row, args
        xx = np.unique(x)
        rr = np.unique(row)
        if np.size(xx) == np.size(x):
            xx = x
        if np.size(rr) == np.size(row):
            rr = row
        if np.size(xx) * np.size(rr) != np.size(x):
            return False, x, row, args
        outargs = [arg[0 : np.size(xx)] for arg in args]
        return True, xx, rr, outargs

    def _rvs(self, *args, size=None, random_state=None):
        # Use basic inverse cdf algorithm for RV generation as default.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        if size is None:  # pragma: no cover
            return Y
        return Y.reshape(size)

    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        if args:
            cond = np.logical_and(
                cond,
                np.logical_and(asarray(args[0]) >= 0, asarray(args[0]) < self._npdf),
            )
        return np.atleast_1d(cond)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.
        """
        return rv_frozen_rows(self, self._shape, *args, **kwds)

    @classmethod
    def create_gen(cls, **kwds):
        """Create and return a `scipy.stats.rv_continuous` object using the
        keyword arguemntets provided"""
        return (cls(**kwds), {})

    def _scipy_version_warning(self):
        import scipy  # pylint: disable=import-outside-toplevel

        scipy_version = scipy.__version__
        vtuple = scipy_version.split(".")
        if int(vtuple[0]) > 1 or int(vtuple[1]) > 7:
            return
        raise DeprecationWarning(
            f"Ensemble.moments will not work correctly with scipy version < 1.8.0, you have {scipy_version}"
        )  # pragma: no cover

    def moment(self, n, *args, **kwds):
        """Returns the moments request moments for all the PDFs.

        This used to call a hacked version `Pdf_gen._moment_fix` which can handle cases of multiple PDFs.
        Now it prints a deprication warning for scipy < 1.8

        Parameters
        ----------
        n : int
            Order of the moment

        Returns
        -------
        moments : array_like
            The requested moments
        """
        self._scipy_version_warning()
        return rv_continuous.moment(self, n, *args, **kwds)


class Pdf_gen_wrap(Pdf_gen):
    """Mixin class to extend `scipy.stats.rv_continuous` with
    information needed for `qp` for analytic distributions.

    """

    def __init__(self, *args, **kwargs):
        """C'tor"""
        # pylint: disable=no-member,protected-access
        super().__init__(*args, **kwargs)
        self._other_init(*args, **kwargs)

    def _my_freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.
        """
        return rv_frozen_func(self, *args, **kwds)

    @classmethod
    def get_allocation_kwds(cls, npdf, **kwargs):
        return {key: ((npdf, 1), val.dtype) for key, val in kwargs.items()}

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
