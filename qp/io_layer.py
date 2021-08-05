"""IO Functions for qp"""

from collections import OrderedDict

import numpy as np

try:
    from astropy.table import Table as apTable
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:  #pragma: no cover
    HAS_ASTROPY = False

try:
    import h5py
    HAS_HDF5 = True
except ImportError:  #pragma: no cover
    HAS_HDF5 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  #pragma: no cover
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PQ = True
except ImportError:  #pragma: no cover
    HAS_PQ = False


def forceToPandables(arr, check_nrow=None):
    """
    Forces a  `numpy.array` into a format that panda can handle

    Parameters
    ----------
    arr : `numpy.array`
        The input array

    check_nrow : `int` or `None`
        If not None, require that `arr.shape[0]` match this value

    Returns
    -------
    out : `numpy.array` or `list` of `numpy.array`
        Something that pandas can handle
    """
    ndim = np.ndim(arr)
    shape = np.shape(arr)
    nrow = shape[0]
    if check_nrow is not None and check_nrow != nrow:  # pragma: no cover
        raise ValueError("Number of rows does not match: %i != %i" % (nrow, check_nrow))
    if ndim == 1:
        return arr
    if ndim == 2:
        return list(arr)
    shape = np.shape(arr)  #pragma: no cover
    ncol = np.product(shape[1:])  #pragma: no cover
    return list(arr.reshape(nrow, ncol))  #pragma: no cover


def tableToDataframe(tab):
    """
    Convert an `astropy.table.Table` to a `pandas.DataFrame`

    Parameters
    ----------
    tab : `astropy.table.Table`
        The table

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    if not HAS_PANDAS:  #pragma: no cover
        raise ImportError("pandas is not available, can't make DataFrame")

    o_dict = OrderedDict()
    for colname in tab.columns:
        col = tab[colname]
        o_dict[colname] = forceToPandables(col.data)
    df = pd.DataFrame(o_dict)
    for k, v in tab.meta.items():
        df.attrs[k] = v  #pragma: no cover
    return df


def arraysToDataframe(array_dict, meta=None):  #pragma: no cover
    """
    Convert a `dict` of  `numpy.array` to a `pandas.DataFrame`

    Parameters
    ----------
    array_dict : `astropy.table.Table`
        The arrays

    meta : `dict` or `None`
        Optional dictionary of metadata

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is not available, can't make DataFrame")

    o_dict = OrderedDict()
    for k, v in array_dict:
        o_dict[k] = forceToPandables(v)
    df =  pd.DataFrame(o_dict)
    if meta is not None:
        for k, v in meta.items():
            df.attrs[k] = v
    return df


def dataframeToTable(df):
    """
    Convert a `pandas.DataFrame` to an `astropy.table.Table`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    if not HAS_ASTROPY:  #pragma: no cover
        raise ImportError("Astropy is not available, can't make astropy tables")

    o_dict = OrderedDict()
    for colname in df.columns:
        col = df[colname]
        if col.dtype.name == 'object':
            o_dict[colname] = np.vstack(col.to_numpy())
        else:
            o_dict[colname] = col.to_numpy()
    tab = apTable(o_dict)
    for k, v in df.attrs.items():
        tab.meta[k] = v  #pragma: no cover
    return tab


def tablesToDataframes(tables):
    """
    Convert several `astropy.table.Table` to `pandas.DataFrame`

    Parameters
    ----------
    tab : `dict` of `astropy.table.Table`
        The tables

    Returns
    -------
    df :  `OrderedDict` of `pandas.DataFrame`
        The dataframes
    """
    return OrderedDict([(k, tableToDataframe(v)) for k, v in tables.items()])


def dataframesToTables(dataframes):
    """
    Convert several `pandas.DataFrame` to `astropy.table.Table`

    Parameters
    ----------
    datafarmes :  `dict` of `pandas.DataFrame`
        The dataframes

    Returns
    -------
    tabs : `OrderedDict` of `astropy.table.Table`
        The tables
    """
    return OrderedDict([(k, dataframeToTable(v)) for k, v in dataframes.items()])


def writeTablesToFits(tables, filepath, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single FITS file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables

    filepath: `str`
        Path to output file

    kwargs are passed to `astropy.io.fits.writeto` call.
    """
    if not HAS_ASTROPY:  #pragma: no cover
        raise ImportError("Astropy is not available, can't save to FITS")
    out_list = [fits.PrimaryHDU()]
    for k, v in tables.items():
        hdu = fits.table_to_hdu(v)
        hdu.name = k
        out_list.append(hdu)
    hdu_list = fits.HDUList(out_list)
    hdu_list.writeto(filepath, **kwargs)


def readFitsToTables(filepath):
    """
    Reads `astropy.table.Table` objects from a FITS file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables
    """
    if not HAS_ASTROPY:  #pragma: no cover
        raise ImportError("Astropy is not available, can't read FITS")
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        tables[hdu.name.lower()] = apTable.read(filepath, hdu=hdu.name)
    return tables


def writeTablesToHdf5(tables, filepath, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    kwargs are passed to `astropy.table.Table` call.
    """
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't save to hdf5")

    for k, v in tables.items():
        v.write(filepath, path=k, append=True, **kwargs)


def readHdf5ToTables(filepath):
    """
    Reads `astropy.table.Table` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be 'paths', values will be tables
    """
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't read hdf5")
    fin = h5py.File(filepath)
    tables = OrderedDict()
    for k in fin.keys():
        tables[k] = apTable.read(filepath, path=k)
    return tables


def writeArraysToHdf5(arrays, filepath, **kwargs):
    """
    Writes a dictionary of `numpy.array` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `numpy.array`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    # pylint: disable=unused-argument
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't save to hdf5")
    raise NotImplementedError("writeArraysToHdf5")  #pragma: no cover


def readHdf5ToArrays(filepath):
    """
    Reads `numpy.array` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `numpy.array`
        Keys will be 'paths', values will be tables
    """
    # pylint: disable=unused-argument
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't read hdf5")
    raise NotImplementedError("writeArraysToHdf5")  #pragma: no cover


def writeDataframesToPq(dataFrames, filepath, **kwargs):
    """
    Writes a dictionary of `pandas.DataFrame` to a parquet files

    Parameters
    ----------
    tables : `dict` of `pandas.DataFrame`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    for k, v in dataFrames.items():
        _ = v.to_parquet("%s%s.pq" % (filepath, k), **kwargs)


def readPqToDataframes(basepath, keys=None, **kwargs):
    """
    Reads `pandas.DataFrame` objects from an parquet file.

    Parameters
    ----------
    basepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    Returns
    -------
    tables : `OrderedDict` of `pandas.DataFrame`
        Keys will be taken from keys
    """

    if keys is None:  #pragma: no cover
        keys = [""]
    dataframes = OrderedDict()
    for key in keys:
        try:
            pqtab = pq.read_table("%s%s.pq" % (basepath, key), **kwargs)
            dataframes[key] = pqtab.to_pandas()
        except Exception:
            pass
    return dataframes
