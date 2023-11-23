"""Implemenation of an ensemble of distributions"""

import os

import numpy as np
from tables_io import io

from qp.dict_utils import (
    check_array_shapes,
    compare_dicts,
    concatenate_dicts,
    slice_dict,
)
from qp.metrics import quick_moment

# import psutil
# import timeit


class Ensemble:
    """An object comprised of many qp.PDF objects to efficiently perform operations on all of them"""

    def __init__(self, gen_func, data, ancil=None):
        """Class constructor

        Parameters
        ----------
        gen_func : `function`
            Function that creates generic distribution object
        data : `dict`
            Dictionary with data used to construct the ensemble

        """
        # start_time = timeit.default_timer()
        self._gen_func = gen_func
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self._gen_class = type(self._gen_obj)

        self._ancil = None
        self.set_ancil(ancil)

        self._gridded = None
        self._samples = None

    def __getitem__(self, key):
        """Build a `qp.Ensemble` object for a sub-set of the PDFs in this ensemble

        Parameter
        ---------
        key : `int` or `slice`
            Used to slice the data to pick out one PDF from this ensemble

        Returns
        -------
        pdf : `scipy.rv_frozen`
            The distribution for the requeseted element or slide
        """
        red_data = {}
        md = self.metadata()
        md.pop("pdf_name")
        md.pop("pdf_version")
        for k, v in md.items():
            red_data[k] = np.squeeze(v)
        dd = slice_dict(self.objdata(), key)
        for k, v in dd.items():
            if len(np.shape(v)) < 2:
                red_data[k] = np.expand_dims(v, 0)
            else:
                red_data[k] = v
        if self._ancil is not None:
            ancil = slice_dict(self._ancil, key)
        else:
            ancil = None
        return Ensemble(self._gen_obj.create, data=red_data, ancil=ancil)

    @property
    def gen_func(self):
        """Return the function used to create the distribution object for this ensemble"""
        return self._gen_func

    @property
    def gen_class(self):
        """Return the class used to generate distributions for this ensemble"""
        return self._gen_class

    @property
    def dist(self):
        """Return the `scipy.stats.rv_continuous` object that generates distributions for this ensemble"""
        return self._gen_obj

    @property
    def kwds(self):
        """Return the kwds associated to the frozen object"""
        return self._frozen.kwds

    @property
    def gen_obj(self):
        """Return the `scipy.stats.rv_continuous` object that generates distributions for this ensemble"""
        return self._gen_obj

    @property
    def frozen(self):
        """Return the `scipy.stats.rv_frozen` object that encapsultes the distributions for this ensemble"""
        return self._frozen

    @property
    def ndim(self):
        """Return the number of dimensions of PDFs in this ensemble"""
        return self._frozen.ndim

    @property
    def shape(self):
        """Return the number of PDFs in this ensemble"""
        return self._frozen.shape

    @property
    def npdf(self):
        """Return the number of PDFs in this ensemble"""
        return self._frozen.npdf

    @property
    def ancil(self):
        """Return the ancillary data dictionary"""
        return self._ancil

    def convert_to(self, to_class, **kwargs):
        """Convert a distribution or ensemble

        Parameters
        ----------
        to_class :  `class`
            Class to convert to
        **kwargs :
            keyword arguments are passed to the output class constructor

        Other Parameters
        ----------------
        method : `str`
            Optional argument to specify a non-default conversion algorithm

        Returns
        -------
        ens : `qp.Ensemble`
            Ensemble of pdfs yype class_to using the data from this object
        """
        kwds = kwargs.copy()
        method = kwds.pop("method", None)
        ctor_func = to_class.creation_method(method)
        class_name = to_class.name
        if ctor_func is None:  # pragma: no cover
            raise KeyError(
                "Class named %s does not have a creation_method named %s"
                % (class_name, method)
            )
        extract_func = to_class.extraction_method(method)
        if extract_func is None:  # pragma: no cover
            raise KeyError(
                "Class named %s does not have a extraction_method named %s"
                % (class_name, method)
            )
        data = extract_func(self, **kwds)
        return Ensemble(ctor_func, data=data)

    def update(self, data, ancil=None):
        """Update the frozen object

        Parameters
        ----------
        data : `dict`
            Dictionary with data used to construct the ensemble
        """
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self.set_ancil(ancil)
        self._gridded = None
        self._samples = None

    def update_objdata(self, data, ancil=None):
        """Update the object data in the distribution

        Parameters
        ----------
        data : `dict`
            Dictionary with data used to construct the ensemble
        """
        new_data = {}
        for k, v in self.metadata().items():
            if k in ["pdf_name", "pdf_version"]:
                continue
            new_data[k] = np.squeeze(v)
        new_data.update(self.objdata())
        new_data.update(data)
        self.update(new_data, ancil)

    def metadata(self):
        """Return the metadata for this ensemble

        Returns
        -------
        metadata : `dict`
            The metadata

        Notes
        -----
        Metadata are elements that are the same for all the PDFs in the ensemble
        These include the name and version of the PDF generation class
        """

        dd = {}
        dd.update(self._gen_obj.metadata)
        return dd

    def objdata(self):
        """Return the object data for this ensemble

        Returns
        -------
        objdata : `dict`
            The object data

        Notes
        -----
        Object data are elements that differ for each PDFs in the ensemble
        """

        dd = {}
        dd.update(self._frozen.kwds)
        dd.pop("row", None)
        dd.update(self._gen_obj.objdata)
        return dd

    def set_ancil(self, ancil):
        """Set the ancillary data dict

        Parameters
        ----------
        ancil : `dict`
            The ancillary data dictionary

        Notes
        -----
        Raises IndexError if the length of the arrays in ancil does not match
        the number of PDFs in the Ensemble
        """
        check_array_shapes(ancil, self.npdf)
        self._ancil = ancil

    def add_to_ancil(self, to_add):  # pragma: no cover
        """Add additionaly columns to the ancillary data dict

        Parameters
        ----------
        to_add : `dict`
            The columns to add to the ancillary data dict

        Notes
        -----
        Raises IndexError if the length of the arrays in to_add does not match
        the number of PDFs in the Ensemble

        This calls dict.update() so it will overwrite existing columns
        """
        check_array_shapes(to_add, self.npdf)
        self._ancil.update(to_add)

    def append(self, other_ens):
        """Append another other_ens to this one

        Parameters
        ----------
        other_ens : `qp.Ensemble`
            The other Ensemble
        """
        if not compare_dicts(
            [self.metadata(), other_ens.metadata()]
        ):  # pragma: no cover
            raise KeyError("Metadata does not match, can not append")
        full_objdata = concatenate_dicts([self.objdata(), other_ens.objdata()])
        if self._ancil is not None and other_ens.ancil is not None:  # pragma: no cover
            full_ancil = concatenate_dicts([self.ancil, other_ens.ancil])
        else:
            full_ancil = None
        self.update_objdata(full_objdata, full_ancil)

    def build_tables(self):
        """Return dicts of numpy arrays for the meta data and object data
        for this ensemble

        Returns
        -------
        meta : `dict`
            Table with the meta data
        data : `dict`
            Table with the object data
        """
        dd = dict(meta=self.metadata(), data=self.objdata())
        if self.ancil is not None:
            dd["ancil"] = self.ancil
        return dd

    def mode(self, grid):
        """return the mode of each ensemble PDF, evaluated on grid

        Parameters
        ----------
        new_grid: array-like
            Grid on which to evaluate PDF

        Returns
        -------
        mode: array-like
            The modes of the PDFs evaluated on new_grid

        Notes
        -----
        Adding expand_dims to return an (N, 1) array to be
        consistent with mean, median, and other point estimates
        """
        new_grid, griddata = self.gridded(grid)
        return np.expand_dims(new_grid[np.argmax(griddata, axis=1)], -1)

    def gridded(self, grid):
        """Build, cache are return the PDF values at grid points

        Parameters
        ----------
        grid : array-like
            The grid points

        Returns
        -------
        gridded : (grid, pdf_values)

        Notes
        -----
        This first comparse grid to the cached value, if they match it returns
        the cached value
        """
        if self._gridded is None or not np.array_equal(self._gridded[0], grid):
            self._gridded = (grid, self.pdf(grid))
        return self._gridded

    def write_to(self, filename):
        """Save this ensemble to a file

        Parameters
        ----------
        filename : `str`

        Notes
        -----
        This will actually write two files, one for the metadata and one for the object data

        This uses `tables_io` to write the data, so any filesuffix that works for
        `tables_io` will work here.
        """
        basename, ext = os.path.splitext(filename)
        tables = self.build_tables()
        io.write(tables, basename, ext[1:])

    def pdf(self, x):
        """
        Evaluates the probablity density function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.pdf(x)

    def logpdf(self, x):
        """
        Evaluates the log of the probablity density function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.logpdf(x)

    def cdf(self, x):
        """
        Evaluates the cumalative distribution function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.cdf(x)

    def logcdf(self, x):
        """
        Evaluates the log of the cumalative distribution function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.logcdf(x)

    def ppf(self, q):
        """
        Evaluates all the PPF of the distribution

        Parameters
        ----------
        q: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.ppf(q)

    def sf(self, q):
        """
        Evaluates the survival fraction of the distribution

        Parameters
        ----------
        x: float or ndarray, float
            (s) at which to evaluate the pdfs

        Returns
        -------
        """
        return self._frozen.sf(q)

    def logsf(self, q):
        """Evaluates the log of the survival function of the distribution

        Parameters
        ----------
        q: float or ndarray, float
            location(s) at which to evaluate the pdfs

        Returns
        -------
        float or ndarray
            Log of the survival function
        """
        return self._frozen.logsf(q)

    def isf(self, q):
        """
        Evaluates the inverse of the survival fraction of the distribution

        Parameters
        ----------
        x: float or ndarray, float
            (s) at which to evaluate the pdfs

        Returns
        -------
        """
        return self._frozen.isf(q)

    def rvs(self, size=None, random_state=None):
        """
        Generate samples from this ensmeble

        Parameters
        ----------
        size: int
            number of samples to return

        Returns
        -------
        """
        return self._frozen.rvs(
            size=(self._frozen.npdf, size), random_state=random_state
        )

    def stats(self, moments="mv"):
        """
        Retrun the stats for this ensemble

        Parameters
        ----------
        moments: `str`
            Which moments to include

        Returns
        -------
        """
        return self._frozen.stats(moments=moments)

    def median(self):
        """Return the medians for this ensemble"""
        return self._frozen.median()

    def mean(self):
        """Return the means for this ensemble"""
        return self._frozen.mean()

    def var(self):
        """Return the variences for this ensemble"""
        return self._frozen.var()

    def std(self):
        """Return the standard deviations for this ensemble"""
        return self._frozen.std()

    def moment(self, n):
        """Return the nth moments for this ensemble"""
        return self._frozen.moment(n)

    def entropy(self):
        """Return the entropy for this ensemble"""
        return self._frozen.entropy()

    # def pmf(self, k):
    #    """ Return the kth pmf for this ensemble """
    #    return self._frozen.pmf(k)

    # def logpmf(self, k):
    #    """ Return the log of the kth pmf for this ensemble """
    #    return self._frozen.logpmf(k)

    def interval(self, alpha):
        """Return the intervals corresponding to a confidnce level of alpha for this ensemble"""
        return self._frozen.interval(alpha)

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
        return self._frozen.histogramize(bins)

    def integrate(self, limits):
        """
        Computes the integral under the ensemble of PDFs between the given limits.

        Parameters
        ----------
        limits: numpy.ndarray, tuple, float
            limits of integration, may be different for all PDFs in the ensemble
        using: string
            parametrization over which to approximate the integral
        dx: float, optional
            granularity of integral

        Returns
        -------
        integral: numpy.ndarray, float
            value of the integral
        """
        return self.cdf(limits[1]) - self.cdf(limits[0])

    def mix_mod_fit(self, comps=5):  # pragma: no cover
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        comps: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation
        vb: boolean
            Report progress

        Returns
        -------
        self.mix_mod: list, qp.Composite objects
            list of qp.Composite objects approximating the PDFs

        Notes
        -----
        Currently only supports mixture of Gaussians
        """
        raise NotImplementedError("mix_mod_fit %i" % comps)

    def moment_partial(self, n, limits, dx=0.01):
        """Return the nth moments for this over a particular range"""
        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        P_eval = self.gridded(grid)[1]
        grid_to_n = grid**n
        return quick_moment(P_eval, grid_to_n, dx)

    def plot(self, key=0, **kwargs):
        """Plot the pdf as a curve

        Parameters
        ----------
        key : `int` or `slice`
            Which PDF or PDFs from this ensemble to plot
        """
        return self._gen_class.plot(self[key], **kwargs)

    def plot_native(self, key=0, **kwargs):
        """Plot the pdf as a curve

        Parameters
        ----------
        key : `int` or `slice`
            Which PDF or PDFs from this ensemble to plot
        """
        return self._gen_class.plot_native(self[key], **kwargs)

    def _get_allocation_kwds(self, npdf):
        tables = self.build_tables()
        keywords = {}
        for group, tab in tables.items():
            if group != "meta":
                keywords[group] = {}
                for key, array in tab.items():
                    shape = list(array.shape)
                    shape[0] = npdf
                    keywords[group][key] = (shape, array.dtype)
        return keywords

    def initializeHdf5Write(self, filename, npdf, comm=None):
        """set up the output write for an ensemble, but set size to npdf rather than
        the size of the ensemble, as the "initial chunk" will not contain the full data

        Parameters
        ----------
        filename : `str`
            Name of the file to create
        npdf : `int`
            Total number of pdfs that will contain the file,
            usually larger then the size of the current ensemble
        comm : `MPI communicator`
            Optional MPI communicator to allow parallel writing
        """
        kwds = self._get_allocation_kwds(npdf)
        group, fout = io.initializeHdf5Write(filename, comm=comm, **kwds)
        return group, fout

    def writeHdf5Chunk(self, fname, start, end):
        """write ensemble data chunk to file

        Parameters
        ----------
        fname : h5py `File object`
            file or group
        start : `int`
            starting index of h5py file
        end : `int`
            ending index in h5py file
        """
        odict = self.build_tables().copy()
        odict.pop("meta")
        io.writeDictToHdf5Chunk(fname, odict, start, end)

    def finalizeHdf5Write(self, filename):
        """write ensemble metadata to the output file

        Parameters
        ----------
        filename : h5py `File object`
            file or group
        """
        mdata = self.metadata()
        io.finalizeHdf5Write(filename, "meta", **mdata)

    # def stack(self, loc, using, vb=True):
    #     """
    #     Produces an average of the PDFs in the ensemble
    #
    #     Parameters
    #     ----------
    #     loc: ndarray, float or float
    #         location(s) at which to evaluate the PDFs
    #     using: string
    #         which parametrization to use for the approximation
    #     vb: boolean
    #         report on progress
    #
    #     Returns
    #     -------
    #     self.stacked: tuple, ndarray, float
    #         pair of arrays for locations where approximations were evaluated
    #         and the values of the stacked PDFs at those points
    #
    #     Notes
    #     -----
    #     Stacking refers to taking the sum of PDFs evaluated on a shared grid and
    #     normalizing it such that it integrates to unity.  This is equivalent to
    #     calculating an average probability (based on the PDFs in the ensemble) over the grid.
    #     This probably should be done in a script and not by qp!  The right way to do it would be to call
    #     qp.Ensemble.evaluate() and sum those outputs appropriately.
    #     TO DO: make this do something more efficient for mixmod, grid, histogram, samples
    #     TO DO: enable stacking on irregular grid
    #     """
    #     loc_range = max(loc) - min(loc)
    #     delta = loc_range / len(loc)
    #     evaluated = self.evaluate(loc, using=using, norm=True, vb=vb)
    #     stack = np.mean(evaluated[1], axis=0)
    #     stack /= np.sum(stack) * delta
    #     assert(np.isclose(np.sum(stack) * delta, 1.))
    #     self.stacked[using] = (evaluated[0], stack)
    #     return self.stacked


# Note: A copious quantity of commented code has been removed in this commit!
# For future reference, it can still be found here:
#  https://github.com/aimalz/qp/blob/d8d145af9514e29c76e079e869b8b4923f592f40/qp/ensemble.py
# Critical additions still remain.  Metrics of individual qp.PDF objects collected in aggregate
# over a qp.Ensemble are still desired.
