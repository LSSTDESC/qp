"""Implementation of an Ensemble of distributions."""

from __future__ import annotations
import os
from typing import Mapping, Optional, Union

import h5py
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
    reduce_arrays_to_1d,
    make_len_equal,
    expand_dimensions,
)
from ..utils.array import encode_strings, reduce_dimensions
from ..metrics import quick_moment
from ..parameterizations.base import Pdf_gen

# import psutil
# import timeit


class Ensemble:
    """An object comprised of one or more distributions with the same parameterization.


    The Ensemble allows you to perform operations on the group of parameterizations as a whole.
    An Ensemble has three main data components, the last of which is optional:

    1. The metadata: this contains information about the parameterization, and
       the coordinates of the parameterization.
    2. The object data: this contains the data that is unique to each distribution,
       for example the values that correspond to the coordinates.
    3. The ancillary data (optional): this contains data points where there is one data point
       for each distribution in the ensemble. There can be many of these columns or
       arrays in the ancillary data table.


    Parameters
    ----------
    the_class : Pdf_gen subclass
        The class to use to parameterize the distributions
    data : Mapping
        Dictionary with data used to construct the ensemble. The keys required
        vary for different parameterizations.
    ancil : Optional[Mapping]
        Dictionary with ancillary data, by default None
    method : Optional[str]
        The key for the creation method to use, by default None

    Examples
    --------
    >>> import qp
    >>> import numpy as np
    >>> data = {'bins': [0,1,2,3,4,5],
    ...         'pdfs': np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
    >>> ancil = {'ids': [105, 108]}}
    >>> ens = qp.Ensemble(qp.hist,data,ancil)
    >>> ens.metadata
    {'pdf_name': array([b'hist'], dtype='|S4'),
    'pdf_version': array([0]),
    'bins': array([[0, 1, 2, 3, 4, 5]])}

    """

    def __init__(
        self,
        the_class: Pdf_gen,
        data: Mapping,
        ancil: Optional[Mapping] = None,
        method: Optional[str] = None,
    ):
        """Class constructor. The requirements are the class object that the ensemble is
        to be parameterized as, and the data dictionary.

        The data dictionary will need different keys depending on what parameterization
        you have chosen. If you are unsure which keys are required, try
        ``qp.[parameterization].create_ensemble?``, where [parameterization] is the
        class of ensemble you wish to create. This will output a docstring with which
        describes the necessary inputs (and this function can also be used to create an
        ensemble instead).

        An ancillary data dictionary can also be provided upon creation. This dictionary
        should contain arrays that are the same length as the number of distributions in the
        ensemble. Essentially, this should include arrays of data where each value in the array
        corresponds to a distribution.


        Parameters
        ----------
        the_class : Pdf_gen subclass
            The class to use to parameterize the distributions
        data : Mapping
            Dictionary with data used to construct the ensemble. The keys required
            vary for different parameterizations.
        ancil : Optional[Mapping]
            Dictionary with ancillary data, by default None
        method : Optional[str]
            The key for the creation method to use, by default None

        """
        # start_time = timeit.default_timer()
        self._gen_func = the_class.creation_method(method)
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self._gen_class = type(self._gen_obj)

        self._ancil = None
        self.set_ancil(ancil)

        self._gridded = None
        self._samples = None

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}(the_class={self._gen_class.name},shape={self.shape})"

    def __len__(self) -> int:
        return self.npdf

    def __getitem__(self, key: Union[int, slice]) -> Ensemble:
        """Build an Ensemble object for a sub-set of the distributions in this ensemble

        Parameters
        ----------
        key : Union [int , slice]
            Used to slice the data to pick out one distribution from this ensemble

        Returns
        -------
        ens : Ensemble
            The ensemble for the requested distribution or slice of distributions
        """
        red_data = {}
        md = self.metadata
        md.pop("pdf_name")
        md.pop("pdf_version")
        for k, v in md.items():
            red_data[k] = np.squeeze(v)

        if self.npdf > 1:
            dd = slice_dict(self.objdata, key)
        elif self.npdf == 1 and key == 0:
            dd = self.objdata
        else:
            raise IndexError(
                f"Cannot slice Ensemble object with {self.npdf} with given index/slice {key}."
            )

        for k, v in dd.items():
            if len(np.shape(v)) < 2:
                red_data[k] = np.expand_dims(v, 0)
            else:
                red_data[k] = v
        if self._ancil is not None and self.npdf > 1:
            ancil = slice_dict(self._ancil, key)
        elif self._ancil is not None and self.npdf == 1:
            ancil = self._ancil
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

    def x_samples(
        self, min: float = 0.0, max: float = 5.0, n: Optional[int] = 1000
    ) -> np.ndarray[float]:
        """Return an array of x values that can be used to plot all the distributions
        in the Ensemble.

        This is meant to plot the characteristic distribution for an Ensemble of
        discrete data. For example, for an ensemble of histograms that would be
        the PDF, and for an ensemble of quantiles that would be the CDF.

        Analytic parameterizations like `mixmod <qp.mixmod_gen>` or `scipy.stats.norm` will just return a
        `np.linspace(min,max,n) <numpy.linspace>`, and it's recommended you input the values as
        the defaults are the same for all analytic distributions.

        Parameters
        ----------
        min : float, optional
            The minimum x value to be used if the parameterization doesn't have an
            `x_samples` method or is analytic, by default 0.
        max : float, optional
            The maximum x value to be used if the parameterization doesn't have an
            `x_samples` method or is analytic, by default 5.
        n : Optional[int], optional
            The number of points to be used if the parameterization doesn't have an
            `x_samples` method or is analytic, by default 1000

        Returns
        -------
        xs : np.ndarray[float]
            The array of points to use.
        """
        try:
            return self._frozen.dist.x_samples()
        except:
            return np.linspace(min, max, n)

    def convert_to(self, to_class: Pdf_gen, **kwargs: str) -> Ensemble:
        """Convert this ensemble to the given parameterization class. To see
        the available conversion methods for the your chosen parameterization
        and their required arguments, check the docstrings for ``qp.to_class``.
        If the parameterization class doesn't have a conversion methods table,
        then it will not be possible to convert to that class.

        Parameters
        ----------
        to_class :  Pdf_gen subclass
            Parameterization class to convert to
        **kwargs :
            Keyword arguments that are passed to the output class constructor

        Other Parameters
        ----------------
        method : str
            Optional argument to specify a non-default conversion algorithm

        Returns
        -------
        ens : Ensemble
            Ensemble of distributions of type class_to using the data from this object

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]]))
        >>> ens_i = ens_h.convert_to(qp.interp, xvals=np.linspace(0,5,10))
        >>> ens_i.metadata
        {'pdf_name': array([b'interp'], dtype='|S6'),
        'pdf_version': array([0]),
        'xvals': array([0.        , 0.55555556, 1.11111111, 1.66666667, 2.22222222,
        2.77777778, 3.33333333, 3.88888889, 4.44444444, 5.        ]))}

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
        return Ensemble(to_class, data=data, method=method)

    def update(self, data: Mapping, ancil: Optional[Mapping] = None) -> None:
        """Update the frozen distribution object with the given data, and set
        the ancillary data table with ``ancil`` if given.

        Parameters
        ----------
        data : Mapping
            Dictionary with data used to construct the ensemble, including metadata.
        ancil : Optional[Mapping], optional
            Optional dictionary that contains data for each of the distributions
            in the ensemble, by default None.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
        >>> ens_h.update(data={'bins': np.array([1,2,3,4,5]), 'pdfs': np.array([0.1,0.1,0.4,0.2])})
        >>> ens_h.metadata
        {'pdf_name': array([b'hist'], dtype='|S4'),
        'pdf_version': array([0]),
        'bins': array([[1, 2, 3, 4, 5]])}

        """
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self.set_ancil(ancil)
        self._gridded = None
        self._samples = None

    def update_objdata(self, data: Mapping, ancil: Optional[Mapping] = None) -> None:
        """Updates the objdata in the frozen distribution, and sets
        the ancillary data table if given.

        Parameters
        ----------
        data : Mapping
            Dictionary with the object data that will be used to reconstruct the ensemble
        ancil : Optional[Mapping], optional
            Optional dictionary that contains data for each of the distributions
            in the ensemble, by default None.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
        >>> ens_h.objdata
        {'pdfs': array([0.   , 0.125, 0.125, 0.5  , 0.25 ])}
        >>> ens_h.update_objdata(data={'pdfs': np.array([0.05,0.09,0.2,0.3,0.15])})
        >>> ens_h.objdata
        {'pdfs': array([[0.06329114, 0.11392405, 0.25316456, 0.37974684, 0.18987342]])}

        """
        new_data = {}
        for k, v in self.metadata.items():
            if k in ["pdf_name", "pdf_version"]:
                continue
            new_data[k] = np.squeeze(v)
        new_data.update(self.objdata)
        new_data.update(data)
        self.update(new_data, ancil)

    @property
    def metadata(self) -> Mapping:
        """Return the metadata for this ensemble. Metadata are elements that are
        the same for all the distributions in the ensemble. These include the name
        and version of the distribution generation class

        Returns
        -------
        metadata : Mapping
            The dictionary of the metadata.

        """

        dd = {}
        dd.update(self._gen_obj.metadata)
        return dd

    @property
    def objdata(self) -> Mapping:
        """Return the data for this ensemble. These are the elements that differ
        for each distribution in the ensemble. For example, the data points that
        correspond to each of the coordinates given in the metadata.

        Returns
        -------
        objdata : Mapping
            The object data

        Notes
        -----

        If the distribution normalized the data (which many do by default), this
        will return the normalized data and not the original input data.

        """

        dd = {}
        dd.update(self._frozen.kwds)
        dd.pop("row", None)
        dd.update(self._gen_obj.objdata)

        # if there is only one distribution reshape data as necessary
        if self.npdf == 1:
            dd = reduce_arrays_to_1d(dd)

        return dd

    def set_ancil(self, ancil: Mapping) -> None:
        """Set the ancillary data dictionary. The arrays in this dictionary must have
        one row for each of the distributions, which means that the length of these
        arrays (or the first dimension) must be the same as the number of distributions
        in the ensemble.

        Parameters
        ----------
        ancil : Mapping
            The ancillary data dictionary.

        Raises
        ------
        IndexError
            If the length of the arrays in ancil does not match the number of
            distributions in the Ensemble.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]]))
        >>> ancil = {'ids': np.array([5,7])}
        >>> ens_h.set_ancil(ancil)
        >>> ens_h.ancil
        {'ids': array([5, 7])}

        """
        check_array_shapes(ancil, self.npdf)
        self._ancil = ancil

    def add_to_ancil(self, to_add: Mapping) -> None:  # pragma: no cover
        """Add additional columns to the ancillary data dictionary. The
        ancil dictionary must already exist. If it does not, use `set_ancil`.

        If any of these columns have the same name as already existing
        ancillary data columns, the new columns will overwrite the old ones.


        Parameters
        ----------
        to_add : Mapping
            The columns to add to the ancillary data dict


        Raises
        ------
        IndexError
            If the length of the arrays in to_add does not match the number of
            distributions in the Ensembles

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ancil = {'ids': np.array([5,7])}
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]]), ancil=ancil)
        >>> ens_h.add_to_ancil({'means':np.array([0.2,0.25])})
        >>> ens_h.ancil
        {'ids': array([5, 7]), 'means': array[0.2,0.25]}


        """
        check_array_shapes(to_add, self.npdf)
        self._ancil.update(to_add)

    def append(self, other_ens: Ensemble) -> None:
        """Append another ensemble to this ensemble. The ensembles must be
        of the same parameterization, or this will not work. They must also
        have the same metadata, so for example if they are both histograms
        they must also have the same bins.

        Both ensembles must have an ancillary data dictionary in order for them
        to be appended to each other. If one ensemble has an ancillary data
        dictionary and the other does not, this will set the ancillary data
        dictionary to `None`.

        Parameters
        ----------
        other_ens : Ensemble
            The ensemble to append to this one.

        Raises
        ------
        KeyError
            Raised if the two ensembles do not have matching metadata.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_1 = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
        >>> ens_2 = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0.5,0.15,0.25,0.45,0.1]))
        >>> ens_1.append(ens_2)
        >>> ens_1.npdf
        2

        """
        if not compare_dicts([self.metadata, other_ens.metadata]):  # pragma: no cover
            raise KeyError("Metadata does not match, can not append")
        full_objdata = concatenate_dicts([self.objdata, other_ens.objdata])
        if self._ancil is not None and other_ens.ancil is not None:  # pragma: no cover
            full_ancil = concatenate_dicts([self.ancil, other_ens.ancil])
        else:
            full_ancil = None
        self.update_objdata(full_objdata, full_ancil)

    def build_tables(self, encode: bool = False, ext: Optional[str] = None) -> Mapping:
        """Returns a dictionary of dictionaries of numpy arrays for the meta data,
        object data, and the ancillary data (if it exists) for this ensemble.

        Parameters
        ----------
        encode : bool
            If True and `ext` is 'hdf5', will encode any string columns in the `ancil` table,
            by default False.
        ext : str, optional
            If set to 'hdf5' when `encode` is True, will encode any string columns
            in the `ancil` table, by default None.


        Returns
        -------
        data : Mapping, `tables_io.TableDict-like`
            The dictionary with the data. Has the keys: ``meta`` for metadata, ``data``
            for object data, and optionally ``ancil`` for ancillary data.

        """
        meta = make_len_equal(self.metadata)
        dd = dict(meta=meta, data=self.objdata)

        # expand out the objdata to 2D arrays if there's only 1 distribution
        if self.npdf == 1:
            new_objdata = expand_dimensions(dd["data"], self.npdf, self.shape[1])
            dd.update(dict(data=new_objdata))

        if self.ancil is not None:
            # encode any string columns if the file will be hdf5
            if encode == True and ext == "hdf5":
                ancil_tmp = encode_strings(self.ancil)
                dd["ancil"] = ancil_tmp
            else:
                dd["ancil"] = self.ancil
        return dd

    def norm(self):
        """Normalizes the input distribution data if it represents a PDF
        and can be normalized.

        Raises
        ------
        AttributeError
            Raised if the parameterization doesn't have a normalization method.
        """

        # get normalized data values
        try:
            normed = self._gen_obj.normalize()
        except AttributeError as err:
            raise AttributeError(
                "This parameterization does not have a normalization function."
            ) from err
        except RuntimeError as err:
            raise err

        # update ensemble objdata with normalized values
        d_keys = list(self.objdata.keys())

        # add in any data from current ensemble which was unchanged
        for key in d_keys:
            if not key in normed.keys():
                normed[key] = self.objdata[key]

        self.update_objdata(data=normed, ancil=self.ancil)

    def mode(self, grid: ArrayLike) -> ArrayLike:
        """Return the mode of each ensemble distribution, evaluated on the given grid.

        Parameters
        ----------
        grid : ArrayLike
            Grid on which to evaluate distribution

        Returns
        -------
        mode : ArrayLike
            The modes of the distributions evaluated on grid, with shape (npdf, 1)

        """
        new_grid, griddata = self.gridded(grid)
        return np.expand_dims(new_grid[np.argmax(griddata, axis=1)], -1)

    def gridded(self, grid: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Build, cache and return the PDF values at the given grid points.
        If the given grid matches the already cached grid, then this just
        returns the cached value.

        Parameters
        ----------
        grid : ArrayLike
            The grid points to evaluate the PDF at.

        Returns
        -------
        gridded : tuple [ ArrayLike, ArrayLike ]
            (grid, pdf_values)


        """
        if self._gridded is None or not np.array_equal(self._gridded[0], grid):
            self._gridded = (grid, self.pdf(grid))
        return self._gridded

    def write_to(self, filename: str) -> None:
        """Write this ensemble to a file.

        The file type can be any of the those supported by tables_io. File type
        is indicated by the suffix of the file name given. Allowed formats are:
        'hdf5','h5','hf5','hd5','fits','fit','pq','parq','parquet'

        If writing to parquet files, a file will be written for the metadata,
        the object data, and the ancillary data if it exists, where the identifying
        key is added to the filename.

        Parameters
        ----------
        filename : str

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_1 = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
        >>> ens_1.write_to("hist-ensemble.hdf5")

        """
        basename, ext = os.path.splitext(filename)
        tables = self.build_tables(encode=True, ext=ext[1:])
        tables_io.write(tables, basename, ext[1:])

    def pdf(self, x: ArrayLike) -> ArrayLike:
        """
        Evaluates the probability density function (PDF) for each of the distributions in the ensemble

        Parameters
        ----------
        x : ArrayLike
            Location(s) at which to evaluate the PDF for each distribution.

        Returns
        -------
        pdf : ArrayLike
            The PDF value(s) at the given location(s).

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.pdf(np.linspace(3,6,6))
        array([[0.5       , 0.5       , 0.25      , 0.25      , 0.        ,
                0.        ],
               [0.37974684, 0.37974684, 0.18987342, 0.18987342, 0.        ,
                0.        ]])

        """

        pdf = self._frozen.pdf(x)

        # reduce dimensionality if possible
        if self.npdf == 1:
            pdf = reduce_dimensions(pdf, x)

        return pdf
        # return self._frozen.pdf(x)

    def logpdf(self, x: ArrayLike) -> ArrayLike:
        """
        Evaluates the log of the probability density function (PDF) for each of the distributions in the ensemble.

        Parameters
        ----------
        x : ArrayLike
            Location(s) at which to do the evaluations

        Returns
        -------
        logpdf : ArrayLike
            The log of the PDF at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.logpdf(np.linspace(3,6,6))
        array([[-0.69314718, -0.69314718, -1.38629436, -1.38629436,        -inf,
               -inf],
              [-0.96825047, -0.96825047, -1.66139765, -1.66139765,        -inf,
               -inf]])

        """
        logpdf = self._frozen.logpdf(x)

        # reduce dimensionality if possible
        if self.npdf == 1:
            logpdf = reduce_dimensions(logpdf, x)

        return logpdf

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """
        Evaluates the cumulative distribution function (CDF) for each of the distributions in the ensemble.

        Parameters
        ----------
        x : ArrayLike
            Location(s) at which to do the evaluations

        Returns
        -------
        cdf : ArrayLike
            The CDF at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.cdf(np.linspace(3,6,6))
        array([[0.25      , 0.55      , 0.8       , 0.95      , 1.        ,
                1.        ],
               [0.43037975, 0.65822785, 0.84810127, 0.96202532, 1.        ,
                1.        ]])

        """
        cdf = self._frozen.cdf(x)

        # reduce dimensionality if possible
        if self.npdf == 1:
            cdf = reduce_dimensions(cdf, x)

        return cdf

    def logcdf(self, x: ArrayLike) -> ArrayLike:
        """
        Evaluates the log of the cumulative distribution function (CDF) for each of the distributions in the ensemble.

        Parameters
        ----------
        x : ArrayLike
            Location(s) at which to do the evaluations

        Returns
        -------
        cdf : ArrayLike
            The log of the CDF at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.logcdf(np.linspace(3,6,6))
        array([[-1.38629436, -0.597837  , -0.22314355, -0.05129329,  0.        ,
                0.        ],
               [-0.84308733, -0.41820413, -0.16475523, -0.03871451,  0.        ,
                0.        ]])

        """
        logcdf = self._frozen.logcdf(x)

        # reduce dimensionality if possible
        if self.npdf == 1:
            logcdf = reduce_dimensions(logcdf, x)

        return logcdf

    def ppf(self, q: ArrayLike) -> ArrayLike:
        """
        Evaluates the percentage point function (PPF) for each of the distributions in the ensemble..

        Parameters
        ----------
        q : ArrayLike
            Location(s) at which to do the evaluations

        Returns
        -------
        ppf : ArrayLike
            The PPF at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.ppf(0.5)
        array([[3.5       ],
               [3.18333333]])

        """
        ppf = self._frozen.ppf(q)

        # reduce dimensionality if possible
        if self.npdf == 1:
            ppf = reduce_dimensions(ppf, q)
        return ppf

    def sf(self, q: ArrayLike) -> ArrayLike:
        """
        Evaluates the survival fraction (SF) for each of the distributions in the ensemble.

        Parameters
        ----------
        q : ArrayLike
            Location(s) at which to evaluate the distributions

        Returns
        -------
        sf : ArrayLike
            The SF at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.sf(0.5)
        array([[1.        ],
               [0.96835443]])

        """
        sf = self._frozen.sf(q)

        # reduce dimensionality if possible
        if self.npdf == 1:
            sf = reduce_dimensions(sf, q)

        return sf

    def logsf(self, q: ArrayLike) -> ArrayLike:
        """Evaluates the log of the survival function (SF) for each of the distributions in the ensemble.

        Parameters
        ----------
        q : ArrayLike
            Location(s) at which to evaluate the distributions

        Returns
        -------
        sf : ArrayLike
            The log of the SF at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.logsf(0.5)
        array([[ 0.        ],
               [-0.03215711]])

        """
        logsf = self._frozen.logsf(q)

        # reduce dimensionality if possible
        if self.npdf == 1:
            logsf = reduce_dimensions(logsf, q)

        return logsf

    def isf(self, q: ArrayLike) -> ArrayLike:
        """
        Evaluates the inverse of the survival fraction (SF) for each of the distributions in the ensemble.

        Parameters
        ----------
        q : ArrayLike
            Location(s) at which to evaluate the distributions

        Returns
        -------
        sf : ArrayLike
            The inverse of the survival fraction at the given location(s)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.isf(0.5)
        array([[3.5       ],
               [3.18333333]])

        """
        isf = self._frozen.isf(q)

        # reduce dimensionality if possible
        if self.npdf == 1:
            isf = reduce_dimensions(isf, q)

        return isf

    def rvs(
        self,
        size: int = 1,
        random_state: Union[None, int, np.random.Generator] = None,
    ) -> ArrayLike:
        """
        Generate samples from the distributions in this ensemble.

        The returned samples are of shape (npdf, size), where size is the number
        of samples per distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to return, by default 1.
        random_state : int, numpy.random.Generator, None, optional
            The random state to use. Can be provided with a random seed for consistency. By default None.

        Returns
        -------
        samples : ArrayLike
            The array of samples for each distribution in the ensemble, shape (npdf,size)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.rvs(size=2)
        array([[3.12956247, 3.72090937],
               [4.96783836, 3.24016123]])


        """
        return self._frozen.rvs(
            size=(self._frozen.npdf, size), random_state=random_state
        )

    def stats(self, moments: str = "mv") -> tuple[ArrayLike, ...]:
        """
        Return some statistics for each of the distributions in this ensemble.

        The moments to be returned are determined by the string given to `moments`,
        where each letter represents a specific moment. The options are:
        "m" = mean, "v" = variance, "s" = (Fisher's) skew, "k" = (Fisher's) kurtosis.

        Parameters
        ----------
        moments : str, optional
            Which moments to include, by default "mv"

        Returns
        -------
        stats : tuple[ArrayLike, ... ]
            A sequence of arrays of the moments requested, where the shape of the arrays is (npdf, 1)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.stats()
        (array([[3.375     ],
                [3.01898734]]),
         array([[0.859375  ],
                [1.23698125]]))

        """
        return self._frozen.stats(moments=moments)

    def median(self) -> ArrayLike:
        """Return the median for each of the distributions in this ensemble.

        Returns
        -------
        medians : ArrayLike
            The median for each distribution, returns a float if there is only one
            distribution, or the shape of the array is (npdf, 1)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.median()
        array([[3.5       ],
               [3.18333333]])

        """
        median = self._frozen.median()
        # reduce dimensionality if possible
        if self.npdf == 1:
            median = reduce_dimensions(median, 1)

        return median

    def mean(self) -> ArrayLike:
        """Return the mean for each of the distributions in this ensemble.

        Returns
        -------
        means : ArrayLike
            The mean for each distribution, returns a float if there is only one
            distribution, or the shape of the array is (npdf, 1)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.mean()
        array([[3.375     ],
               [3.01898734]])

        """
        mean = self._frozen.mean()

        # reduce dimensionality if possible
        if self.npdf == 1:
            mean = reduce_dimensions(mean, 1)

        return mean

    def var(self) -> ArrayLike:
        """Return the variance for each of the distributions in this ensemble.

        Returns
        -------
        variances : ArrayLike
            The variance for each distribution, returns a float if there is only one
            distribution, or the shape of the array is (npdf, 1)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.var()
        array([[0.859375  ],
               [1.23698125]])

        """
        var = self._frozen.var()

        # reduce dimensionality if possible
        if self.npdf == 1:
            var = reduce_dimensions(var, 1)

        return var

    def std(self) -> ArrayLike:
        """Return the standard deviation for each of the distributions in this ensemble.

        Returns
        -------
        stds : ArrayLike
            The standard deviations for each distribution, the shape of the array is (npdf, 1)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.std()
        array([[0.92702481],
               [1.11219659]])

        """
        std = self._frozen.std()
        # reduce dimensionality if possible
        if self.npdf == 1:
            std = reduce_dimensions(std, 1)

        return std

    def moment(self, n: int) -> ArrayLike:
        """Return the nth moment for each of the distributions in this ensemble.

        Parameters
        ----------
        n : int
            The order of the moment

        Returns
        -------
        moments : ArrayLike
            The nth moment for each distribution, the shape of the array is (npdf, 1)

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.moment(2)
        array([[12.25      ],
               [10.35126582]])

        """
        moment = self._frozen.moment(n)

        # reduce dimensionality if possible
        if self.npdf == 1:
            moment = reduce_dimensions(moment, 1)

        return moment

    def entropy(self) -> ArrayLike:
        """Return the differential entropy for each of the distributions in this ensemble.

        Returns
        -------
        entropy : ArrayLike
            The entropy for each distribution, the shape of the array is (npdf, 1)


        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.entropy()
        array([[1.21300757],
               [1.45307405]])

        """
        entropy = self._frozen.entropy()

        # reduce dimensionality if possible
        if self.npdf == 1:
            entropy = reduce_dimensions(entropy, 1)

        return entropy

    # def pmf(self, k):
    #    """ Return the kth pmf for this ensemble """
    #    return self._frozen.pmf(k)

    # def logpmf(self, k):
    #    """ Return the log of the kth pmf for this ensemble """
    #    return self._frozen.logpmf(k)

    def interval(self, alpha: ArrayLike) -> tuple[ArrayLike, ...]:
        """
        Return the intervals corresponding to a confidence level of `alpha` for each of the
        distributions in this ensemble.

        Parameters
        ----------
        alpha : ArrayLike
            The array of values to return intervals for. These should be the probability that a random variable will be
            drawn from the returned range. Each value should be in the range [0,1].

        Returns
        -------
        interval :  tuple[ArrayLike, ...]
            A tuple of the arrays containing the intervals for each distribution, where the
            shape of the arrays is (npdf, len(alpha))

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.interval(alpha=[0,0.5,0.9])
        (array([[1.4       , 3.        , 3.5       ],
                [0.79      , 2.2875    , 3.18333333]]),
         array([[3.5       , 4.        , 4.8       ],
                [3.18333333, 3.84166667, 4.73666667]]))

        """
        return self._frozen.interval(alpha)

    def histogramize(self, bins: ArrayLike) -> tuple[ArrayLike]:
        """
        Computes integrated histogram bin values for all distributions in the ensemble.

        Parameters
        ----------
        bins : ArrayLike
            Array of N+1 endpoints of N bins

        Returns
        -------
        histogram: tuple[ArrayLike, ArrayLike]
            The first array in the tuple is the bin edges that were input. The second
            array in the tuple is an (npdf, N) array of the values in the bins.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])
        >>> ens_h.histogramize(bins=np.array([1,2,3,4,5]))
        (array([1, 2, 3, 4, 5]),
         array([[0.125     , 0.125     , 0.5       , 0.25      ],
                [0.11392405, 0.25316456, 0.37974684, 0.18987342]]))
        """
        return self._frozen.histogramize(bins)

    def integrate(
        self, limits: tuple[Union[float, ArrayLike], Union[float, ArrayLike]]
    ) -> ArrayLike:
        """
        Computes the integral under the probability distribution functions (PDFs) of the distributions in the ensemble
        between the given limits.

        Parameters
        ----------
        limits : tuple[Union[float, ArrayLike], Union[float, ArrayLike]]
            A tuple with the limits of integration, where the first object in the tuple is
            the lower limit, and the second object is the upper limit. The limit objects can
            be floats or arrays, where the number of limits is the length of those arrays, or
            `nlimits`.


        Returns
        -------
        integral: ArrayLike
            Value of the integral(s), with the shape (npdf, nlimits)
        """
        return self.cdf(limits[1]) - self.cdf(limits[0])

    def mix_mod_fit(self, comps=5):  # pragma: no cover
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        comps : int, optional
            Number of components to consider
        using : str, optional
            Which existing approximation to use, defaults to first approximation
        vb : bool
            Report progress

        Returns
        -------
        self.mix_mod: list [ `qp.Composite` ]
            List of `qp.Composite` objects approximating the PDFs

        Notes
        -----
        Currently only supports mixture of Gaussians
        """
        raise NotImplementedError("mix_mod_fit %i" % comps)

    def moment_partial(self, n: int, limits: tuple, dx: float = 0.01) -> ArrayLike:
        """Return the nth moment over a particular range for each of the distributions in this ensemble.

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

    def plot(
        self,
        key: Union[int, slice] = 0,
        **kwargs: str,
    ):
        """Plot the selected distribution as a curve.

        Parameters
        ----------
        key : int or slice, optional
            The index or slice of the distribution or distributions from this ensemble
            to plot, by default 0.

        Other Parameters
        ----------------
        axes : Axes
            The axes to plot on. Either this or xlim must be provided.
        xlim : (float, float)
            The x-axis limits. Either this or axes must be provided.
        kwargs :
            Any keyword arguments to pass to matplotlib's axes.plot() method.

        Returns
        -------
        axes : Axes
            The plot axes
        """
        return self._gen_class.plot(self[key], **kwargs)

    def plot_native(self, key: Union[int, slice] = 0, **kwargs: str):
        """Plot the selected distribution in the default format for this parameterization. To find what arguments are
        required for specific parameterizations, you can check the docstrings
        of ``qp.[parameterization].plot_native``, where ``[parameterization]`` is the parameterization
        class for the current ensemble.

        Parameters
        ----------
        key : int or slice, optional
            The index or slice of the distribution or distributions from this ensemble
            to plot, by default 0.
        kwargs :
            The keyword arguments to pass to the parameterization's plot_native method.

        Returns
        -------
        axes : Axes
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

    def initializeHdf5Write(
        self, filename: str, npdf: int, comm=None
    ) -> tuple[dict[str, h5py.File | h5py.Group], h5py.File]:
        """Set up the output write for an ensemble, but set size to npdf rather than
        the size of the ensemble, as the "initial chunk" will not contain the full data

        Parameters
        ----------
        filename : str
            Name of the file to create
        npdf : int
            Total number of distributions that the file will contain,
            usually larger then the size of the current ensemble
        comm : MPI communicator
            Optional MPI communicator to allow parallel writing

        Returns
        -------
        group : dict[str, h5py.File | h5py.Group]
            A dictionary of the groups to write to.
        fout : h5py.File
            The output file object that has been created.
        """
        kwds = self._get_allocation_kwds(npdf)
        group, fout = hdf5.initialize_HDF5_write(filename, comm=comm, **kwds)
        return group, fout

    def writeHdf5Chunk(
        self, fname: "h5py.File" | "h5py.Group", start: int, end: int
    ) -> None:
        """Write a chunk of the ensemble data to file. This will write
        the data for the distributions in the slice from [start:end] to the file.
        This includes the ancillary data table.

        Parameters
        ----------
        fname : h5py.File | h5py.Group
            The file or group object to write to
        start : int
            Starting index of data to write in the h5py file
        end : int
            Ending index of data to write in the h5py file
        """
        odict = self.build_tables(encode=True, ext="hdf5").copy()
        odict.pop("meta")
        hdf5.write_dict_to_HDF5_chunk(fname, odict, start, end)

    def finalizeHdf5Write(self, filename: "h5py.File" | "h5py.Group") -> None:
        """Write ensemble metadata to the output file and close the file.

        Parameters
        ----------
        filename : h5py.File | h5py.Group
            The file or group object to complete writing and close.
        """
        mdata = make_len_equal(self.metadata)
        hdf5.finalize_HDF5_write(filename, "meta", **mdata)
