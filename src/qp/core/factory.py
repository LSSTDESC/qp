"""This module implements a factory that manages different types of Ensembles"""

from __future__ import annotations
import sys
import os

from collections import OrderedDict
from collections.abc import Iterator
from typing import Mapping, Union, Optional, Tuple

import numpy as np

from scipy import stats as sps

import tables_io
from tables_io import hdf5
from tables_io.types import NUMPY_DICT

from .ensemble import Ensemble

from ..utils.dictionary import compare_dicts, concatenate_dicts, reduce_arrays_to_1d
from ..utils.array import decode_strings

from ..parameterizations.base import Pdf_gen_wrap, Pdf_gen


class Factory(OrderedDict):
    """Factory that creates and manages Ensembles of distributions."""

    def __init__(self):
        """C'tor"""
        super().__init__()
        self._load_scipy_classes()

    @staticmethod
    def _build_data_dict(md_table: Mapping, data_table: Mapping) -> Mapping:
        """Convert the metadata and data tables to a single dictionary that can be used as input to
        build an Ensemble."""
        data_dict = {}

        for col, col_data in md_table.items():
            ndim = np.ndim(col_data)

            if ndim > 1:
                col_data = np.squeeze(col_data)
                if np.ndim(col_data) == 0:
                    col_data = col_data.item()
            elif ndim == 1:
                col_data = col_data[0]

            if isinstance(col_data, bytes):
                col_data = col_data.decode()

            data_dict[col] = col_data

        for col, col_data in data_table.items():
            if len(col_data.shape) < 2:  # pragma: no cover
                data_dict[col] = np.expand_dims(col_data, -1)
            else:
                data_dict[col] = col_data
        return data_dict

    def _make_scipy_wrapped_class(self, class_name: str, scipy_class):
        """Build a qp class from a scipy class"""
        # pylint: disable=protected-access
        override_dict = dict(
            name=class_name,
            version=0,
            freeze=Pdf_gen_wrap._my_freeze,
            _other_init=scipy_class.__init__,
            __doc__=scipy_class.__doc__,
        )
        the_class = type(class_name, (Pdf_gen_wrap, scipy_class), override_dict)

        def create_ensemble(data: Mapping, ancil: Optional[Mapping] = None) -> Ensemble:
            """Creates an Ensemble of distribution(s) in the given parameterization.

            Input data format:
            data = {'arg1': values, 'arg2': values ...} where 'arg1', 'arg2'... are the arguments for the parameterization.
            The length of the values should be the number of distributions being created in the Ensemble, with a minimum value of 1.


            Parameters
            ----------
            data : Mapping
                The dictionary of data for the distributions.
            ancil : Optional[Mapping], optional
                A dictionary of metadata for the distributions, where any arrays have the same length as the number of distributions, by default None

            Returns
            -------
            Ensemble
                An Ensemble object containing all of the given distributions.

            Examples
            --------

            To create an Ensemble with two Gaussian distributions and their associated ids:

            >>> import qp
            >>> data = {'loc': np.array([[0.45],[0.55]]) , 'scale': np.array([[0.2],[0.15]])}
            >>> ancil = {'ids': [20,25]}
            >>> ens = qp.stats.norm.create_ensemble(data,ancil)
            >>> ens.metadata
            {'pdf_name': array([b'norm'], dtype='|S4'), 'pdf_version': array([0])}

            """

            return Ensemble(the_class, data, ancil)

        setattr(
            the_class,
            "create_ensemble",
            create_ensemble,
        )

        self.add_class(the_class)

    def _load_scipy_classes(self):
        """Build qp classes from all the scipy.stats classes"""
        names = sps.__all__
        for name in names:
            attr = getattr(sps, name)
            if isinstance(attr, sps.rv_continuous):
                self._make_scipy_wrapped_class(name, type(attr))

    def add_class(self, the_class: Pdf_gen) -> None:
        """Add a parameterization class to the factory dictionary, so that it is
        included in the set of known parameterization classes. It includes an
        entry both for the actual class name, which ends in ``_gen``, and the
        parameterization name that is also aliased to the class.

        Parameters
        ----------
        the_class : Pdf_gen subclass
            The parameterization class we are adding, which must inherit from `Pdf_gen`.
        """
        # if not isinstance(the_class, Pdf_gen): #pragma: no cover
        #    raise TypeError("Can only add sub-classes of Pdf_Gen to factory")
        if not hasattr(the_class, "name"):  # pragma: no cover
            raise AttributeError(
                "Can not add class %s to factory because it doesn't have a name attribute"
                % the_class
            )
        if the_class.name in self:  # pragma: no cover
            raise KeyError(
                "Class named %s is already in factory, point to %s"
                % (the_class.name, self[the_class.name])
            )
        the_class.add_method_dicts()
        the_class.add_mappings()
        self[the_class.name] = the_class
        setattr(self, "%s_gen" % the_class.name, the_class)
        setattr(self, the_class.name, the_class)

    def create(
        self,
        class_name: Union[str, Pdf_gen],
        data: Mapping,
        method: Optional[str] = None,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Make an Ensemble of a particular type of distribution. The ``data``
        dictionary will need different keys depending on what parameterization
        you have chosen.

        If you are unsure which keys are required, try
        ``qp.[parameterization].create_ensemble?``, where [parameterization] is the
        class of ensemble you wish to create. This will output a docstring with the necessary
        inputs (and this function can also be used to create an Ensemble).

        Parameters
        ----------
        class_name : str or class
            The name of the parameterization to make a distribution from.
        data : Mapping
            Dictionary of values passed to the parameterization create function.
        method : str | None, optional
            Used to select which creation method to invoke if there are multiple.
        ancil : Mapping, optional
            Dictionary with ancillary data, by default `None`

        Returns
        -------
        ens : Ensemble
            The newly created Ensemble

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> data = {'bins': [0,1,2,3,4,5],
        ...         'pdfs': np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]])}
        >>> ens_h = qp.create('hist', data=data)
        >>> ens.metadata
        {'pdf_name': array([b'hist'], dtype='|S4'),
        'pdf_version': array([0]),
        'bins': array([[0, 1, 2, 3, 4, 5]])}


        """

        # handle if class creation function is given instead of string
        if not isinstance(class_name, str):
            class_name = class_name.name

        if class_name not in self:  # pragma: no cover
            raise KeyError("Class named %s is not in factory" % class_name)
        the_class = self[class_name]
        # ctor_func = the_class.creation_method(method)
        return Ensemble(the_class, data, method=method, ancil=ancil)

    def from_tables(
        self, tables: Mapping, decode: bool = False, ext: str = None
    ) -> Ensemble:
        """Build this Ensemble from a dictionary of tables, where the metadata has key `meta`,
        and the data has key `data`. If there is an ancillary data table, it should have the
        key `ancil`.

        The function will create the ensemble with the parameterization given in the `meta`
        table, and will use any other information in the `meta` table necessary to figure out
        how to construct the ensemble (i.e. construction method).

        Parameters
        ----------
        tables : Mapping
            The dictionary of tables to turn into an Ensemble.
        decode : bool
            If `True` and `ext` is 'hdf5', will decode any string type columns in `ancil`,
            by default False.
        ext : str, optional
            If 'hdf5' and `decode` is True, will decode any string type columns in `ancil`,
            by default None.

        Returns
        -------
        ens : Ensemble
            The ensemble constructed from the data in the tables.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> meta = {'pdf_name': np.array(['hist'.encode()]), 'pdf_version': np.array([0]),
        ... 'bins':np.array([0,1,2,3,4,5])}
        >>> data = {'pdfs': np.array([[0.  , 0.1 , 0.1 , 0.4 , 0.2 ],
        ... [0.05, 0.09, 0.2 , 0.3 , 0.15]])}
        >>> ens = qp.from_tables({'meta': meta, 'data': data})
        >>> ens.metadata
        {'pdf_name': array([b'hist'], dtype='|S4'),
        'pdf_version': array([0]),
        'bins': array([[0, 1, 2, 3, 4, 5]])}

        """
        md_table = tables["meta"]
        data_table = tables["data"]
        ancil_table = tables.get("ancil")
        if ancil_table is not None:
            if decode == True and ext == "hdf5":
                # decode string arrays
                ancil_table = decode_strings(ancil_table)

        data = self._build_data_dict(md_table, data_table)

        pdf_name = data.pop("pdf_name")
        pdf_version = data.pop("pdf_version")
        if pdf_name not in self:  # pragma: no cover
            raise KeyError("Class named %s is not in factory" % pdf_name)

        # make sure old files are compatible
        if pdf_name == "quant":
            check_input = data.pop("check_input", None)
            if not check_input == None:
                # replace with ensure extent
                data["ensure_extent"] = check_input
        # elif pdf_name == "hist" or pdf_name == "interp":
        #     check_input = data.pop("check_input", None)
        #     if not check_input == None:
        #         # replace with norm
        #         data["norm"] = check_input

        the_class = self[pdf_name]
        reader_convert = the_class.reader_method(pdf_version)
        # ctor_func = the_class.creation_method(None)
        if reader_convert is not None:  # pragma: no cover
            data = reader_convert(data)
        return Ensemble(the_class, data=data, ancil=ancil_table)

    def read_metadata(self, filename: str) -> Mapping:
        """Read an ensemble's metadata from a file, without loading the full data.
        The file must have multiple tables, one of which is called 'meta'.

        Parameters
        ----------
        filename : str
            The full path to the file.

        Returns
        -------
        meta : Mapping
            Returns the metadata table as a dictionary of numpy arrays.

        Examples
        --------

        >>> import qp
        >>> qp.read_metadata("hist-ensemble.hdf5")
        {'pdf_name': array([b'hist'], dtype='|S4'),
        'pdf_version': array([0]),
        'bins': array([[0, 1, 2, 3, 4, 5]])}

        """
        tables = tables_io.read(filename, NUMPY_DICT, keys=["meta"])
        return tables["meta"]

    def is_qp_file(self, filename: str) -> bool:
        """Test if a file is a `qp` file. Must have at least a table called 'meta' in the
        file, and that 'meta' table must have a property 'pdf_name'.

        Parameters
        ----------
        filename : str
            Path to file to test.

        Returns
        -------
        value : bool
            True if the file is a qp file

        Examples
        --------

        >>> import qp
        >>> qp.is_qp_file("test-qpfile.hdf5")
        True

        """
        try:
            # If this isn't a table-like file with a 'meta' table this will throw an exception
            tables = tables_io.read_native(filename, keys=["meta"])
            # If the 'meta' tables doesn't have 'pdf_name' or it is empty this will throw an exception or fail
            return len(tables["meta"]["pdf_name"]) > 0
        except Exception as msg:
            # Any exception means it isn't a qp file
            print(f"This is not a qp file because {msg}")
        return False

    def read(self, filename: str, fmt: Optional[str] = None) -> Ensemble:
        """Read this ensemble from a file. The file must be a `qp` file.

        The function will create the ensemble with the parameterization given in the metadata
        table, and will use any other information in the metadata table necessary to figure out
        how to construct the ensemble (i.e. construction method).

        Parameters
        ----------
        filename : str
            Path to the file.
        fmt : Optional[str], optional
            File format, if `None` it will be taken from the file extension.
            Allowed formats are: 'hdf5','h5','hf5','hd5','fits','fit','pq',
            'parq','parquet'

        Returns
        -------
        ens : Ensemble
            The ensemble constructed from the data in the file.

        Examples
        --------

        >>> import qp
        >>> ens = qp.read("test-qpfile.hdf5")

        """
        _, ext = os.path.splitext(filename)
        if ext in [".pq"]:
            keys = ["data", "meta", "ancil"]
            allow_missing_keys = True
        else:
            keys = None
            allow_missing_keys = False

        tables = tables_io.read(
            filename,
            NUMPY_DICT,
            fmt=fmt,
            keys=keys,
            allow_missing_keys=allow_missing_keys,
        )  # pylint: disable=no-member

        # set up file_fmt to have the file extension information
        if fmt is not None:
            file_fmt = fmt
        else:
            file_fmt = ext[1:]
        return self.from_tables(tables, decode=True, ext=file_fmt)

    def data_length(self, filename: str) -> int:
        """Get the size of data in a file. The file must be a `qp` file, which means
        it must contain an Ensemble with a metadata table.

        Parameters
        ----------
        filename : str
            The path to the file with the data.

        Returns
        -------
        nrows : int
            The length of the data, or the number of distributions in the data.

        Examples
        --------

        >>> import qp
        >>> qp.data_length("hist-ensemble.hdf5")
        2

        """
        f, _ = hdf5.read_HDF5_group(filename, "data")
        num_rows = hdf5.get_group_input_data_length(f)
        return num_rows

    def iterator(
        self,
        filename: str,
        chunk_size: int = 100_000,
        rank: int = 0,
        parallel_size: int = 1,
    ) -> Iterator[int, int, Ensemble]:
        """Iterates through a given Ensemble file and yields a chunk of the ensemble data at a time.
        This means that the returned Ensemble contains the distributions from the returned start
        index to the returned stop index. If there is an ancillary data table, the Ensemble will
        also contain any ancillary data for those distributions.

        Parameters
        ----------
        filename : str
            The path to the file to iterate through.
        chunk_size : int, optional
            The size of chunks to yield, by default 100_000
        rank : int, optional
            The process rank, if run in MPI, by default 0
        parallel_size : int, optional
            The number of processes, if run in MPI, by default 1

        Yields
        ------
        Iterator[int, int, Ensemble]
            the start index, ending index, and an Ensemble with distributions between those two indices

        Raises
        ------
        TypeError
            Raised if this function is run with files that are not ``hdf5`` files.
        KeyError
            Raised if the ``pdf_name`` in the file is not one of the available parameterizations.


        Examples
        --------

        To iterate through an HDF5 Ensemble file, we can use the following code:

        >>> data_file = "./test.hdf5"
        >>> for start, end, ens_chunk in qp.iterator(data_file, chunk_size=11):
        ...     print(f"Indices are: ({start}, {end})")
        ...     print(ens_chunk)
        Indices are: (0, 11)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (11, 22)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (22, 33)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (33, 44)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (44, 55)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (55, 66)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (66, 77)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (77, 88)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (88, 99)
        Ensemble(the_class=mixmod,shape=(11, 3))
        Indices are: (99, 100)
        Ensemble(the_class=mixmod,shape=(1, 3))


        """
        extension = os.path.splitext(filename)[1]
        if extension not in [".hdf5"]:  # pragma: no cover
            raise TypeError("Can only use qp.iterator on hdf5 files")

        metadata = hdf5.read_HDF5_to_dict(filename, "meta")
        pdf_name = metadata.pop("pdf_name")[0].decode()
        _pdf_version = metadata.pop("pdf_version")[0]
        if pdf_name not in self:  # pragma: no cover
            raise KeyError("Class named %s is not in factory" % pdf_name)
        the_class = self[pdf_name]
        # reader_convert = the_class.reader_method(pdf_version)
        # ctor_func = the_class.creation_method(None)

        f, infp = hdf5.read_HDF5_group(filename, "data")
        try:
            ancil_f, ancil_infp = hdf5.read_HDF5_group(filename, "ancil")
        except KeyError:  # pragma: no cover
            ancil_f, ancil_infp = (None, None)
        num_rows = hdf5.get_group_input_data_length(f)
        ranges = hdf5.data_ranges_by_rank(num_rows, chunk_size, parallel_size, rank)
        data = self._build_data_dict(metadata, {})
        ancil_data = OrderedDict()
        for start, end in ranges:
            for key, val in f.items():
                data[key] = hdf5.read_HDF5_dataset_to_array(val, start, end)
            if ancil_f is not None:
                for key, val in ancil_f.items():
                    ancil_data[key] = hdf5.read_HDF5_dataset_to_array(val, start, end)
            yield start, end, Ensemble(the_class, data=data, ancil=ancil_data)
        infp.close()
        if ancil_infp is not None:
            ancil_infp.close()

    def convert(self, in_dist: Ensemble, class_name: str, **kwds) -> Ensemble:
        """Convert an ensemble to a different parameterization. Keyword arguments are
        required to convert to a different parameterization, but the specific keyword
        arguments required will vary. To check the available conversion methods
        and their associated arguments refer to the docstrings for ``qp.class_name``
        of the parameterization you are converting to. If the class does not
        have a conversion methods table, then it will not be possible to convert
        to that parameterization.


        Parameters
        ----------
        in_dist : Ensemble
            The input Ensemble object to convert.
        class_name : str
            Name of the representation to convert to as a string
        kwds : Mapping
            The arguments required to convert to a function of the given type.

        Returns
        -------
        ens : Ensemble
            The ensemble we converted to

        Examples
        --------

        The following example demonstrates converting from a histogram parameterization
        to an interpolation parameterization. The arguments given will not be the same
        when converting between other parameterizations.

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0,0.1,0.1,0.4,0.2],[0.05,0.09,0.2,0.3,0.15]]))
        >>> ens_i = qp.convert(ens_h, "interp", xvals=np.linspace(0,5,10))
        >>> ens_i.metadata
        {'pdf_name': array([b'interp'], dtype='|S6'),
        'pdf_version': array([0]),
        'xvals': array([0.        , 0.55555556, 1.11111111, 1.66666667, 2.22222222,
        2.77777778, 3.33333333, 3.88888889, 4.44444444, 5.        ]))}

        """
        kwds_copy = kwds.copy()
        method = kwds_copy.pop("method", None)
        if class_name not in self:  # pragma: no cover
            raise KeyError("Class named %s is not in factory" % class_name)
        if class_name not in self:  # pragma: no cover
            raise KeyError("Class named %s is not in factory" % class_name)
        the_class = self[class_name]
        extract_func = the_class.extraction_method(method)
        if extract_func is None:  # pragma: no cover
            raise KeyError(
                "Class named %s does not have a extraction_method named %s"
                % (class_name, method)
            )
        data = extract_func(in_dist, **kwds_copy)
        return self.create(class_name, data, method)

    def pretty_print(self, stream=sys.stdout) -> None:
        """Print a level of the conversion dictionary in a human-readable format

        Parameters
        ----------
        stream : `stream`
            The stream to print to
        """
        for class_name, cl in self.items():
            stream.write("\n")
            stream.write("%s: %s\n" % (class_name, cl))
            cl.print_method_maps(stream)

    @staticmethod
    def concatenate(ensembles: list[Ensemble]) -> Ensemble:
        """Concatenate a list of Ensembles into one Ensemble. The
        Ensembles must be of the same parameterization and have the same metadata.

        Parameters
        ----------
        ensembles : list[Ensemble]
            The list of ensembles we are concatenating

        Returns
        -------
        ens : Ensemble
            The output

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_1 = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
        >>> ens_1.npdf
        1
        >>> ens_2 = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([[0.05,0.09,0.2,0.3,0.15]]))
        >>> ens_2.npdf
        1
        >>> ens_all = qp.concatenate([ens_1, ens_2])
        >>> ens_all.npdf
        2

        """
        if not ensembles:  # pragma: no cover
            return None
        metadata_list = []
        objdata_list = []
        ancil_list = []
        gen_class = None
        for ensemble in ensembles:
            metadata_list.append(ensemble.metadata)
            objdata_list.append(ensemble.objdata)
            if gen_class is None:
                gen_class = ensemble.gen_class
            if ancil_list is not None:
                if ensemble.ancil is None:
                    ancil_list = None
                else:  # pragma: no cover
                    ancil_list.append(ensemble.ancil)
        if not compare_dicts(metadata_list):  # pragma: no cover
            raise ValueError("Metadata does not match")
        metadata = metadata_list[0]
        data = concatenate_dicts(objdata_list)
        if ancil_list is not None:  # pragma: no cover
            ancil = concatenate_dicts(ancil_list, 1)
        else:
            ancil = None
        for k, v in metadata.items():
            if k in ["pdf_name", "pdf_version"]:
                continue
            data[k] = np.squeeze(v)
        return Ensemble(gen_class, data, ancil)

    @staticmethod
    def write_dict(filename: str, ensemble_dict: Mapping[str, Ensemble], **kwargs):
        """Writes out a dictionary of Ensembles to an HDF5 file. Each Ensemble
        in the dictionary will be written to a group, and within each Ensemble group there
        will be subgroups for the metadata, data, and (optional) ancillary data tables.

        Parameters
        ----------
        filename : str
            The file path to write to.
        ensemble_dict : Mapping[str, Ensemble]
            The dictionary of Ensembles to write.
        kwargs :
            Keyword arguments that are passed to the tables_io write_dicts_to_HDF5 function

        Raises
        ------
        ValueError
            Raised if the dictionary contains any values that are not Ensembles.

        Examples
        --------

        >>> import qp
        >>> import numpy as np
        >>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
        ... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
        >>> ens_i = qp.interp.create_ensemble(xvals= np.array([0,1,2,3,4]),
        ... yvals = np.array([[0.05,0.09,0.2,0.3,0.15]]))
        >>> qp.write_dict("qp-ensembles.hdf5",{"ens_h": ens_h, "ens_i": ens_i})

        """
        output_tables = {}
        for key, val in ensemble_dict.items():
            # check that val is a qp.Ensemble
            if not isinstance(val, Ensemble):
                raise ValueError(
                    "All values in ensemble_dict must be qp.Ensemble"
                )  # pragma: no cover

            output_tables[key] = val.build_tables(encode=True, ext="hdf5")
        hdf5.write_dicts_to_HDF5(output_tables, filename, **kwargs)

    @staticmethod
    def read_dict(filename: str) -> Mapping[str, Ensemble]:
        """Reads in one or more Ensembles from an HDF5 file to a dictionary of Ensembles.
        The file should contain one top-level group per ensemble. Each Ensemble group should
        have subgroups that are the metadata, data, and (optional) ancillary data tables.

        Parameters
        ----------
        filename : str
            The path to the ``HDF5`` file to read in.

        Returns
        -------
        Mapping[str, Ensemble]
            A dictionary with the Ensembles contained in the file.

        Examples
        --------

        >>> import qp
        >>> ens_dict = qp.read_dict("qp-ensembles.hdf5")

        """
        results = {}

        # retrieve all the top level groups. Assume each top level group
        # corresponds to an ensemble.
        top_level_groups = hdf5.read_HDF5_group_names(filename)

        # for each top level group, convert the subgroups (data, meta, ancil) into
        # a dictionary of dictionaries and pass the result to `from_tables`.
        for top_level_group in top_level_groups:
            tables = {}
            keys = hdf5.read_HDF5_group_names(filename, top_level_group)
            for key_name in keys:
                # retrieve the hdf5 group object
                group_object, _ = hdf5.read_HDF5_group(
                    filename, f"{top_level_group}/{key_name}"
                )

                # use the hdf5 group object to gather data into a dictionary
                tables[key_name] = hdf5.read_HDF5_group_to_dict(group_object)

            results[top_level_group] = from_tables(tables, decode=True, ext="hdf5")

        return results


_FACTORY = Factory()


def instance():
    """Return the factory instance"""
    return _FACTORY


stats = _FACTORY
add_class = _FACTORY.add_class
create = _FACTORY.create
read = _FACTORY.read
read_metadata = _FACTORY.read_metadata
iterator = _FACTORY.iterator
convert = _FACTORY.convert
concatenate = _FACTORY.concatenate
data_length = _FACTORY.data_length
from_tables = _FACTORY.from_tables
is_qp_file = _FACTORY.is_qp_file
write_dict = _FACTORY.write_dict
read_dict = _FACTORY.read_dict
