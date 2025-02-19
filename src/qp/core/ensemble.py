"""Implementation of an ensemble of distributions"""

import os

import numpy as np
import tables_io
from tables_io import hdf5
from typing import Mapping, Optional, Union
from numpy.typing import ArrayLike

from ..utils.dictionary import (
    check_array_shapes,
    compare_dicts,
    concatenate_dicts,
    slice_dict,
)
from ..metrics import quick_moment
from ..parameterizations.base import Pdf_gen

# import psutil
# import timeit


class Ensemble:
    """An object comprised of many qp distribution objects to efficiently perform operations on all of them."""

    def __init__(
        self,
        the_class: Pdf_gen,
        data: Mapping,
        ancil: Optional[Mapping] = None,
        method: Optional[str] = None,
    ):
        """Class constructor

        Parameters
        ----------
        the_class : a subclass of `Pdf_gen`
            The class to use to parameterize the distributions
        data : `dict`
            Dictionary with data used to construct the ensemble
        ancil : `dict`
            Dictionary with ancillary data, by default None
        method : `str`
            The key for the creation method to use, by default None

        """
        # start_time = timeit.default_timer()
        self._gen_func = the_class.creation_method(
            method
        )  # TODO: figure out if this is better than .create
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self._gen_class = type(self._gen_obj)

        self._ancil = None
        self.set_ancil(ancil)

        self._gridded = None
        self._samples = None

    def __getitem__(self, key: Union[int, slice]):
        """Build a `qp.Ensemble` object for a sub-set of the distributions in this ensemble

        Parameter
        ---------
        key : `int` or `slice`
            Used to slice the data to pick out one distribution from this ensemble

        Returns
        -------
        ens : Ensemble
            The ensemble for the requested distribution or slice of distributions
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
        return Ensemble(self._gen_obj, data=red_data, ancil=ancil)

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
        """Return the kwds associated to the frozen object for this ensemble"""
        return self._frozen.kwds

    @property
    def gen_obj(self):
        """Return the `scipy.stats.rv_continuous` object that generates distributions for this ensemble"""
        return self._gen_obj

    @property
    def frozen(self):
        """Return the `scipy.stats.rv_frozen` object that encapsulates the distributions for this ensemble"""
        return self._frozen

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of distributions in this ensemble."""
        return self._frozen.ndim

    @property
    def shape(self) -> tuple:
        """Return the shape of distributions in this ensemble."""
        return self._frozen.shape

    @property
    def npdf(self) -> int:
        """Return the number of distributions in this ensemble."""
        return self._frozen.npdf

    @property
    def ancil(self) -> Mapping:
        """Return the ancillary data dictionary for this ensemble."""
        return self._ancil

    def convert_to(self, to_class: Pdf_gen, **kwargs):
        """Convert this ensemble to the given parameterization class.

        Parameters
        ----------
        to_class :  `class`, must be based on `Pdf_gen`
            Parameterization class to convert to
        **kwargs :
            Keyword arguments that are passed to the output class constructor

        Other Parameters
        ----------------
        method : `str`
            Optional argument to specify a non-default conversion algorithm

        Returns
        -------
        ens : `qp.Ensemble`
            Ensemble of pdfs type class_to using the data from this object
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
        return Ensemble(to_class, data=data)

    def update(self, data: Mapping, ancil: Optional[Mapping] = None):
        """Update the frozen distribution object with the given data, and set
        the ancillary data table with ``ancil`` if given.

        Parameters
        ----------
        data : `Mapping`
            Dictionary with data used to construct the ensemble
        ancil : `Mapping`
            Optional dictionary that contains data for each of the distributions
            in the ensemble, by default None.
        """
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self.set_ancil(ancil)
        self._gridded = None
        self._samples = None

    def update_objdata(self, data: Mapping, ancil: Optional[Mapping] = None):
        """Updates the metadata and data in the frozen distribution, and sets
        the ancillary data table if given.

        Parameters
        ----------
        data : `dict`
            Dictionary with data used to construct the ensemble
        ancil : `Mapping`
            Optional dictionary that contains data for each of the distributions
            in the ensemble, by default None.
        """
        new_data = {}
        for k, v in self.metadata().items():
            if k in ["pdf_name", "pdf_version"]:
                continue
            new_data[k] = np.squeeze(v)
        new_data.update(self.objdata())
        new_data.update(data)
        self.update(new_data, ancil)

    def metadata(self) -> Mapping:
        """Return the metadata for this ensemble. Metadata are elements that are
        the same for all the distributions in the ensemble. These include the name
        and version of the distribution generation class

        Returns
        -------
        metadata : `dict`
            The dictionary of the metadata.

        """

        dd = {}
        dd.update(self._gen_obj.metadata)
        return dd

    def objdata(self) -> Mapping:
        """Return the data for this ensemble. These are the elements that differ
        for each distribution in the ensemble. For example, the data points that
        correspond to each of the coordinates given in the metadata.

        Returns
        -------
        objdata : `dict`
            The object data

        """

        dd = {}
        dd.update(self._frozen.kwds)
        dd.pop("row", None)
        dd.update(self._gen_obj.objdata)
        return dd

    def set_ancil(self, ancil: Mapping):
        """Set the ancillary data dictionary. The arrays in this dictionary must have
        one value for each of the distributions, which means that the length of these
        arrays must be the same as the number of distributions in the ensemble.

        Parameters
        ----------
        ancil : `dict`
            The ancillary data dictionary.

        Notes
        -----
        Raises IndexError if the length of the arrays in ancil does not match
        the number of PDFs in the Ensemble
        """
        check_array_shapes(ancil, self.npdf)
        self._ancil = ancil

    def add_to_ancil(self, to_add: Mapping):  # pragma: no cover
        """Add additional columns to the ancillary data dictionary.
        If any of these columns have the same name as already existing
        ancillary data columns, the new columns will overwrite the old ones.



        Parameters
        ----------
        to_add : `dict`
            The columns to add to the ancillary data dict

        Notes
        -----
        Raises IndexError if the length of the arrays in to_add does not match
        the number of distributions in the Ensembles
        """
        check_array_shapes(to_add, self.npdf)
        self._ancil.update(to_add)

    def append(self, other_ens):
        """Append another ensemble to this ensemble. The ensembles must be
        of the same parameterization, or this will not work. They must also
        have the same metadata, so for example if they are both histograms
        they must also have the same bins.

        Both ensembles must have an ancillary data dictionary in order for them
        to be appended to each other. If one ensemble has an ancillary data dictionary
        and the other does not, this will set the ancillary data dictionary to `None`.

        Parameters
        ----------
        other_ens : `qp.Ensemble`
            The ensemble to append to this one.

        Raises
        ------
        KeyError:
            Raised if the two ensembles do not have matching metadata.
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

    def build_tables(self) -> Mapping:
        """Returns a dictionary of dictionaries of numpy arrays for the meta data,
        object data, and the ancillary data (if it exists) for this ensemble.

        Returns
        -------
        data : `Mapping`
            The dictionary with the data. Has the keys: ``meta`` for metadata, ``data``
            for object data, and optionally ``ancil`` for ancillary data.

        """
        dd = dict(meta=self.metadata(), data=self.objdata())
        if self.ancil is not None:
            dd["ancil"] = self.ancil
        return dd

    def mode(self, grid: ArrayLike) -> ArrayLike:
        """Return the mode of each ensemble distribution, evaluated on the given grid.

        Parameters
        ----------
        new_grid: array-like
            Grid on which to evaluate distribution

        Returns
        -------
        mode: array-like
            The modes of the distributions evaluated on new_grid, with shape (npdf, 1)

        """
        new_grid, griddata = self.gridded(grid)
        return np.expand_dims(new_grid[np.argmax(griddata, axis=1)], -1)

    def gridded(self, grid: ArrayLike) -> ArrayLike:
        """Build, cache and return the PDF values at the given grid points.
        If the given grid matches the already cached grid, then this just
        returns the cached value.

        Parameters
        ----------
        grid : array-like
            The grid points to evaluate the PDF at.

        Returns
        -------
        gridded : (grid, pdf_values)


        """
        if self._gridded is None or not np.array_equal(self._gridded[0], grid):
            self._gridded = (grid, self.pdf(grid))
        return self._gridded

    def write_to(self, filename: str):
        """Write this ensemble to a file. The file type can be any of the
        those supported by tables_io. File type is indicated by the suffix
        of the file name given.

        If writing to parquet files, a file will be written for the metadata,
        the object data, and the ancillary data if it exists, where the identifying
        key is added to the filename.

        Parameters
        ----------
        filename : `str`

        """
        basename, ext = os.path.splitext(filename)
        tables = self.build_tables()
        tables_io.write(tables, basename, ext[1:])

    def pdf(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the probability density function (PDF) for the whole ensemble

        Parameters
        ----------
        x: `float` or `ndarray`
            Location(s) at which to evaluate the PDF for each distribution.

        Returns
        -------
        pdf :  `float` or `arraylike`
            The PDF value(s) at the given location(s).

        """
        return self._frozen.pdf(x)

    def logpdf(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the log of the probability density function (PDF) for the whole ensemble

        Parameters
        ----------
        x: `float` or `ndarray`
            Location(s) at which to do the evaluations

        Returns
        -------
        logpdf : `float` or `arraylike`
            The log of the PDF at the given location(s)
        """
        return self._frozen.logpdf(x)

    def cdf(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the cumulative distribution function (CDF) for the whole ensemble

        Parameters
        ----------
        x: `float` or `ndarray`
            Location(s) at which to do the evaluations

        Returns
        -------
        cdf : `float` or `arraylike`
            The CDF at the given location(s)
        """
        return self._frozen.cdf(x)

    def logcdf(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the log of the cumulative distribution function (CDF) for the whole ensemble

        Parameters
        ----------
        x: `float` or `ndarray`
            Location(s) at which to do the evaluations

        Returns
        -------
        cdf : `float` or `arraylike`
            The log of the CDF at the given location(s)
        """
        return self._frozen.logcdf(x)

    def ppf(self, q: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the percentage point function (PPF) for the whole ensemble.

        Parameters
        ----------
        q: `float` or `ndarray`
            Location(s) at which to do the evaluations

        Returns
        -------
        ppf : `float` or `arraylike`
            The PPF at the given location(s)
        """
        return self._frozen.ppf(q)

    def sf(self, q: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the survival fraction (SF) of the distribution for the whole ensemble.

        Parameters
        ----------
        q: `float` or `ndarray`
            Location(s) at which to evaluate the distributions

        Returns
        -------
        sf : `float` or `arraylike`
            The SF at the given location(s)
        """
        return self._frozen.sf(q)

    def logsf(self, q: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """Evaluates the log of the survival function (SF) of the distribution for the whole ensemble.

        Parameters
        ----------
        q: `float` or `ndarray`
            Location(s) at which to evaluate the distributions

        Returns
        -------
        sf : `float` or `arraylike`
            The log of the SF at the given location(s)
        """
        return self._frozen.logsf(q)

    def isf(self, q: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        """
        Evaluates the inverse of the survival fraction of the distribution for the whole ensemble.

        Parameters
        ----------
        q: `float` or `ndarray`
            Location(s) at which to evaluate the distributions

        Returns
        -------
        sf : `float` or `arraylike`
            The inverse of the survival fraction at the given location(s)
        """
        return self._frozen.isf(q)

    def rvs(
        self,
        size: Optional[int] = None,
        random_state: Union[None, int, np.random.Generator] = None,
    ) -> ArrayLike:
        """
        Generate samples from the distributions in this ensemble. The returned samples
        are of shape (npdf, size), where size is the number of samples per distribution.
        If no size is given, it defaults to 1.

        Parameters
        ----------
        size: `int`, optional
            Number of samples to return, by default None
        random_state : `int`, `np.random.Generator`, `None`, optional
            The random state to use. Can be provided with a random seed for consistency. By default None.

        Returns
        -------
        samples : `arraylike`
            The array of samples for each distribution in the ensemble.
        """
        return self._frozen.rvs(
            size=(self._frozen.npdf, size), random_state=random_state
        )

    def stats(self, moments: str = "mv") -> tuple[ArrayLike]:
        """
        Return some statistics for the distributions in this ensemble.

        The moments to be returned are determined by the string given to `moments`,
        where each letter represents a specific moment. The options are:
        "m" = mean, "v" = variance, "s" = (Fisher's) skew, "k" = (Fisher's) kurtosis.

        Parameters
        ----------
        moments: `str`
            Which moments to include, by default "mv"

        Returns
        -------
        stats : sequence
            A sequence of arrays of the moments requested, where the shape of the arrays is (npdf, 1)
        """
        return self._frozen.stats(moments=moments)

    def median(self) -> ArrayLike:
        """Return the medians of the distributions in this ensemble.

        Returns
        -------
        medians : `arraylike`
            The median for each distribution, the shape of the array is (npdf, 1)
        """
        return self._frozen.median()

    def mean(self) -> ArrayLike:
        """Return the means of the distributions in this ensemble.

        Returns
        -------
        means : `arraylike`
            The mean for each distribution, the shape of the array is (npdf, 1)
        """
        return self._frozen.mean()

    def var(self) -> ArrayLike:
        """Return the variances for the distributions in this ensemble.

        Returns
        -------
        variances : `arraylike`
            The variance for each distribution, the shape of the array is (npdf, 1)
        """
        return self._frozen.var()

    def std(self) -> ArrayLike:
        """Return the standard deviations the distributions in this ensemble.

        Returns
        -------
        stds : `arraylike`
            The standard deviations for each distribution, the shape of the array is (npdf, 1)
        """
        return self._frozen.std()

    def moment(self, n: int) -> ArrayLike:
        """Return the nth moments for the distributions in this ensemble.

        Parameters
        ----------
        n : `int`
            The order of the moment

        Returns
        -------
        moments : `arraylike`
            The nth moment for each distribution, the shape of the array is (npdf, 1)
        """
        return self._frozen.moment(n)

    def entropy(self) -> ArrayLike:
        """Return the differential entropy for the distributions in this ensemble.

        Returns
        -------
        entropy : `arraylike`
            The entropy for each distribution, the shape of the array is (npdf, 1)
        """
        return self._frozen.entropy()

    # def pmf(self, k):
    #    """ Return the kth pmf for this ensemble """
    #    return self._frozen.pmf(k)

    # def logpmf(self, k):
    #    """ Return the log of the kth pmf for this ensemble """
    #    return self._frozen.logpmf(k)

    def interval(self, alpha) -> tuple[ArrayLike]:
        """Return the intervals corresponding to a confidence level of alpha for the
         distributions in this ensemble.

        Parameters
        ----------
        alpha : `arraylike`
            The array of values to return intervals for. These should be the probability that an rv will be
            drawn from the returned range. Each value should be in the range [0,1].

        Returns
        -------
        interval :  `tuple[arraylike]`
            A tuple of the arrays containing the intervals for each distribution, where the
            shape of the arrays is (npdf, len(alpha))

        """
        return self._frozen.interval(alpha)

    def histogramize(self, bins: ArrayLike) -> tuple[ArrayLike]:
        """
        Computes integrated histogram bin values for all distributions in the ensemble.

        Parameters
        ----------
        bins: `ndarray`
            Array of N+1 endpoints of N bins

        Returns
        -------
        histogram: `tuple[ndarray]`
            The first array in the tuple is the bin edges that were input. The second
            array in the tuple is an (npdf, N) array of the values in the bins.
        """
        return self._frozen.histogramize(bins)

    def integrate(
        self, limits: tuple[Union[float, ArrayLike], Union[float, ArrayLike]]
    ) -> ArrayLike:
        """
        Computes the integral under the distributions in the ensemble between the given limits.

        Parameters
        ----------
        limits: tuple[Union[numpy.ndarray, float], Union[numpy.ndarray, float]]
            A tuple with the limits of integration, where the first object in the tuple is
            the lower limit, and the second object is the upper limit. The limit objects can
            be floats or arrays, where the number of limits is the length of those arrays, or
            `nlimits`.


        Returns
        -------
        integral: `numpy.ndarray`
            Value of the integral(s), with the shape (npdf, nlimits)
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

    def moment_partial(self, n: int, limits: tuple, dx: float = 0.01) -> ArrayLike:
        """Return the nth moments for the distributions in this ensemble
        over a particular range

        Parameters
        ----------
        n : int
            The order of the moment to return
        limits : tuple
            The range over which to calculate the moment, where the second number is the
            upper limit.
        dx : float, optional
            The distance between grid points when calculating, by default 0.01

        Returns
        -------
        ArrayLike
            Array of the moments for each of the distributions, with shape (npdf,)

        """
        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        P_eval = self.gridded(grid)[1]
        grid_to_n = grid**n
        return quick_moment(P_eval, grid_to_n, dx)

    def plot(self, key: Union[int, slice] = 0, **kwargs):
        """Plot the selected distribution as a curve.

        Parameters
        ----------
        key : `int` or `slice`
            The index or slice of the distribution or distributions from this ensemble
            to plot, by default 0.
        kwargs :
            The keyword arguments to pass to the parameterization's plotting method.

        Returns
        -------
        axes :
            The plot axes
        """
        return self._gen_class.plot(self[key], **kwargs)

    def plot_native(self, key: Union[int, slice] = 0, **kwargs):
        """Plot the selected distribution as a curve.

        Parameters
        ----------
        key : `int` or `slice`
            The index or slice of the distribution or distributions from this ensemble
            to plot, by default 0.
        kwargs :
            The keyword arguments to pass to the parameterization's plot_native method.

        Returns
        -------
        axes :
            The plot axes


        """
        return self._gen_class.plot_native(self[key], **kwargs)

    def _get_allocation_kwds(self, npdf: int) -> Mapping:
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

    def initializeHdf5Write(self, filename: str, npdf: int, comm=None):
        """set up the output write for an ensemble, but set size to npdf rather than
        the size of the ensemble, as the "initial chunk" will not contain the full data

        Parameters
        ----------
        filename : `str`
            Name of the file to create
        npdf : `int`
            Total number of distributions that the file will contain,
            usually larger then the size of the current ensemble
        comm : `MPI communicator`
            Optional MPI communicator to allow parallel writing

        Returns
        -------
        group : `dict` of `h5py.File` or `h5py.Group`
            A dictionary of the groups to write to.
        fout : `h5py.File`
            The output file object that has been created.
        """
        kwds = self._get_allocation_kwds(npdf)
        group, fout = hdf5.initialize_HDF5_write(filename, comm=comm, **kwds)
        return group, fout

    def writeHdf5Chunk(self, fname, start: int, end: int):
        """Write a chunk of the ensemble data to file. This will write
        the data for the distributions in the slice from [start:end] to the file.
        This includes the ancillary data table.

        Parameters
        ----------
        fname : h5py `File object` or `group`
            The file or group object to write to
        start : `int`
            Starting index of data to write in the h5py file
        end : `int`
            Ending index of data to write in the h5py file
        """
        odict = self.build_tables().copy()
        odict.pop("meta")
        hdf5.write_dict_to_HDF5_chunk(fname, odict, start, end)

    def finalizeHdf5Write(self, filename):
        """Write ensemble metadata to the output file and close the file.

        Parameters
        ----------
        filename : h5py `File object` or `group`
            The file or group object to complete writing and close.
        """
        mdata = self.metadata()
        hdf5.finalize_HDF5_write(filename, "meta", **mdata)

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
