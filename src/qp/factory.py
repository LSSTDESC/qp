"""This module implements a factory that manages different types of PDFs"""

import sys
import os

from collections import OrderedDict

import numpy as np

from scipy import stats as sps

from tables_io import io
from tables_io.types import NUMPY_DICT

from qp.ensemble import Ensemble

from qp.dict_utils import compare_dicts, concatenate_dicts

from qp.pdf_gen import Pdf_gen_wrap


class Factory(OrderedDict):
    """Factory that creates and manages PDFs"""

    def __init__(self):
        """C'tor"""
        super().__init__()
        self._load_scipy_classes()

    @staticmethod
    def _build_data_dict(md_table, data_table):
        """Convert the tables to a dictionary that can be used to build an Ensemble"""
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

    def _make_scipy_wrapped_class(self, class_name, scipy_class):
        """Build a qp class from a scipy class"""
        # pylint: disable=protected-access
        override_dict = dict(
            name=class_name,
            version=0,
            freeze=Pdf_gen_wrap._my_freeze,
            _other_init=scipy_class.__init__,
        )
        the_class = type(class_name, (Pdf_gen_wrap, scipy_class), override_dict)
        self.add_class(the_class)

    def _load_scipy_classes(self):
        """Build qp classes from all the scipy classes"""
        names = sps.__all__
        for name in names:
            attr = getattr(sps, name)
            if isinstance(attr, sps.rv_continuous):
                self._make_scipy_wrapped_class(name, type(attr))

    def add_class(self, the_class):
        """Add a class to the factory

        Parameters
        ----------
        the_class : class
            The class we are adding, must inherit from Pdf_Gen
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
                "Class nameed %s is already in factory, point to %s"
                % (the_class.name, self[the_class.name])
            )
        the_class.add_method_dicts()
        the_class.add_mappings()
        self[the_class.name] = the_class
        setattr(self, "%s_gen" % the_class.name, the_class)
        setattr(self, the_class.name, the_class.create)

    def create(self, class_name, data, method=None):
        """Make an ensemble of a particular type of distribution

        Parameters
        ----------
        class_name : `str`
            The name of the class to make
        data : `dict`
            Values passed to class create function
        method : `str` [`None`]
            Used to select which creation method to invoke

        Returns
        -------
        ens : `qp.Ensemble`
            The newly created ensemble
        """
        if class_name not in self:  # pragma: no cover
            raise KeyError("Class nameed %s is not in factory" % class_name)
        the_class = self[class_name]
        ctor_func = the_class.creation_method(method)
        return Ensemble(ctor_func, data)

    def from_tables(self, tables):
        """Build this ensemble from a tables

        Parameters
        ----------
        tables: `dict`

        Notes
        -----
        This will use information in the meta data table to figure out how to construct the data
        need to build the ensemble.
        """
        md_table = tables["meta"]
        data_table = tables["data"]
        ancil_table = tables.get("ancil")

        data = self._build_data_dict(md_table, data_table)

        pdf_name = data.pop("pdf_name")
        pdf_version = data.pop("pdf_version")
        if pdf_name not in self:  # pragma: no cover
            raise KeyError("Class nameed %s is not in factory" % pdf_name)

        the_class = self[pdf_name]
        reader_convert = the_class.reader_method(pdf_version)
        ctor_func = the_class.creation_method(None)
        if reader_convert is not None:  # pragma: no cover
            data = reader_convert(data)
        return Ensemble(ctor_func, data=data, ancil=ancil_table)

    def read_metadata(self, filename):
        """Read an ensemble's metadata from a file, without loading the full data.

        Parameters
        ----------
        filename : `str`
        """
        tables = io.read(filename, NUMPY_DICT, keys=["meta"])
        return tables["meta"]

    def is_qp_file(self, filename):
        """Test if a file is a qp file

        Parameters
        ----------
        filename : `str`
            File to test

        Returns
        -------
        value : bool
            True if the file is a qp file
        """
        try:
            # If this isn't a table-like file with a 'meta' table this will throw an exception
            tables = io.readNative(filename, keys=["meta"])
            # If the 'meta' tables doesn't have 'pdf_name' or it is empty this will throw an exception or fail
            return len(tables["meta"]["pdf_name"]) > 0
        except Exception as msg:
            # Any exception means it isn't a qp file
            print(f"This is not a qp file because {msg}")
        return False

    def read(self, filename):
        """Read this ensemble from a file

        Parameters
        ----------
        filename : `str`

        Notes
        -----
        This will use information in the meta data to figure out how to construct the data
        need to build the ensemble.
        """
        _, ext = os.path.splitext(filename)
        if ext in [".pq"]:
            keys = ["data", "meta", "ancil"]
            allow_missing_keys = True
        else:
            keys = None
            allow_missing_keys = False

        tables = io.read(
            filename, NUMPY_DICT, keys=keys, allow_missing_keys=allow_missing_keys
        )  # pylint: disable=no-member

        return self.from_tables(tables)

    def data_length(self, filename):
        """Get the size of data

        Parameters
        ----------
        filename : `str`

        Returns
        -------
        nrows : `int`
        """
        f, _ = io.readHdf5Group(filename, "data")
        num_rows = io.getGroupInputDataLength(f)
        return num_rows

    def iterator(self, filename, chunk_size=100_000, rank=0, parallel_size=1):
        """Return an iterator for chunked read

        Parameters
        ----------
        filename : `str`

        chunk_size : `int`
        """
        extension = os.path.splitext(filename)[1]
        if extension not in [".hdf5"]:  # pragma: no cover
            raise TypeError("Can only use qp.iterator on hdf5 files")

        metadata = io.readHdf5ToDict(filename, "meta")
        pdf_name = metadata.pop("pdf_name")[0].decode()
        _pdf_version = metadata.pop("pdf_version")[0]
        if pdf_name not in self:  # pragma: no cover
            raise KeyError("Class nameed %s is not in factory" % pdf_name)
        the_class = self[pdf_name]
        # reader_convert = the_class.reader_method(pdf_version)
        ctor_func = the_class.creation_method(None)

        f, infp = io.readHdf5Group(filename, "data")
        try:
            ancil_f, ancil_infp = io.readHdf5Group(filename, "ancil")
        except KeyError:  # pragma: no cover
            ancil_f, ancil_infp = (None, None)
        num_rows = io.getGroupInputDataLength(f)
        ranges = io.data_ranges_by_rank(num_rows, chunk_size, parallel_size, rank)
        data = self._build_data_dict(metadata, {})
        ancil_data = OrderedDict()
        for start, end in ranges:
            for key, val in f.items():
                data[key] = io.readHdf5DatasetToArray(val, start, end)
            if ancil_f is not None:
                for key, val in ancil_f.items():
                    ancil_data[key] = io.readHdf5DatasetToArray(val, start, end)
            yield start, end, Ensemble(ctor_func, data=data, ancil=ancil_data)
        infp.close()
        if ancil_infp is not None:
            ancil_infp.close()

    def convert(self, in_dist, class_name, **kwds):
        """Read an ensemble to a different repersenation

        Parameters
        ----------
        in_dist : `qp.Ensemble`
            Input distributions
        class_name : `str`
            Representation to convert to

        Returns
        -------
        ens : `qp.Ensemble`
            The ensemble we converted to
        """
        kwds_copy = kwds.copy()
        method = kwds_copy.pop("method", None)
        if class_name not in self:  # pragma: no cover
            raise KeyError("Class nameed %s is not in factory" % class_name)
        if class_name not in self:  # pragma: no cover
            raise KeyError("Class nameed %s is not in factory" % class_name)
        the_class = self[class_name]
        extract_func = the_class.extraction_method(method)
        if extract_func is None:  # pragma: no cover
            raise KeyError(
                "Class named %s does not have a extraction_method named %s"
                % (class_name, method)
            )
        data = extract_func(in_dist, **kwds_copy)
        return self.create(class_name, data, method)

    def pretty_print(self, stream=sys.stdout):
        """Print a level of the converstion dictionary in a human-readable format

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
    def concatenate(ensembles):
        """Concatanate a list of ensembles

        Parameters
        ----------
        ensembles : `list`
            The ensembles we are concatanating

        Returns
        -------
        ens : `qp.Ensemble`
            The output
        """
        if not ensembles:  # pragma: no cover
            return None
        metadata_list = []
        objdata_list = []
        ancil_list = []
        gen_func = None
        for ensemble in ensembles:
            metadata_list.append(ensemble.metadata())
            objdata_list.append(ensemble.objdata())
            if gen_func is None:
                gen_func = ensemble.gen_func
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
            ancil = concatenate_dicts(ancil_list)
        else:
            ancil = None
        for k, v in metadata.items():
            if k in ["pdf_name", "pdf_version"]:
                continue
            data[k] = np.squeeze(v)
        return Ensemble(gen_func, data, ancil)

    @staticmethod
    def write_dict(filename, ensemble_dict, **kwargs):
        output_tables = {}
        for key, val in ensemble_dict.items():
            # check that val is a qp.Ensemble
            if not isinstance(val, Ensemble):
                raise ValueError("All values in ensemble_dict must be qp.Ensemble") # pragma: no cover

            output_tables[key] = val.build_tables()
        io.writeDictsToHdf5(output_tables, filename, **kwargs)

    @staticmethod
    def read_dict(filename):
        """Assume that filename is an HDF5 file, containing multiple qp.Ensembles
        that have been stored at nparrays."""
        results = {}

        # retrieve all the top level groups. Assume each top level group 
        # corresponds to an ensemble.
        top_level_groups = io.readHdf5GroupNames(filename)

        # for each top level group, convert the subgroups (data, meta, ancil) into
        # a dictionary of dictionaries and pass the result to `from_tables`.
        for top_level_group in top_level_groups:
            tables = {}
            keys = io.readHdf5GroupNames(filename, top_level_group)
            for key_name in keys:
                # retrieve the hdf5 group object
                group_object, _ = io.readHdf5Group(filename, f"{top_level_group}/{key_name}")

                # use the hdf5 group object to gather data into a dictionary
                tables[key_name] = io.readHdf5GroupToDict(group_object)

            results[top_level_group] = from_tables(tables)

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
