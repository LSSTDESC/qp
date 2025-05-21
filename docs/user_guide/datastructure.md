# Data Structure

## Basic structure

The basic data structure of any Ensemble consists of two required tables, the **metadata** table and the **objdata** table. There is also an optional **ancillary data** table. These 'tables' are in fact dictionaries of NumPy arrays. They are `Table-like` objects of the 'numpyDict' type as defined by `tables_io` (see [the documentation](https://tables-io.readthedocs.io/en/latest/index.html) for more), which essentially means that each of the NumPy arrays must have:

1. the same length, or first dimension of their shape
2. all items in the dictionary must be iterable

### Metadata table

The two keys or columns that are required in the **metadata** table are:

'pdf_name'
: the name of the parameterization class

'pdf_version'
: the version number for that parameterization.

Other parameters may be present in the metadata table depending on the parameterization. One type of parameter that is often present are the **coordinates** related to the parameterization. These are data that are shared across all of the distributions in an Ensemble. For example, the bin edges for a histogram, or quantiles in a quantile parameterization.

Here's an example of a **metadata** table for a histogram:

| key           | value              |
| ------------- | ------------------ |
| "pdf_name"    | `array(b["hist"])` |
| "pdf_version" | `array([0])`       |
| "bins"        | `array([0,1,2,3])` |

```{note}

Strings are byte-encoded (UTF-8) here to allow them to be easily written to HDF5 files. UTF-16 and UTF-32 strings are not supported.

```

### Data table (objdata)

The **data** table in an Ensemble is where the data specific to each distribution is stored. For example, in a histogram this is the bin values. The arrays are typically of shape ($n_{pdf}$, $n_{data}$), where:

- $n_{pdf}$: the number of distributions in the Ensemble
- $n_{data}$: the number of data points per distribution.

Typically $n_{data}$ corresponds to the number of coordinate values ($n_{coords}$), though in a histogram $n_{data} = n_{coords} - 1$ because there is one more bin edge than there are number of bins.

An example of the **data** table for a histogram:

| key    | value                              |
| ------ | ---------------------------------- |
| "pdfs" | `array([[4,5,6],[1,2,3],[7,8,9]])` |

### Ancillary data table

The **ancillary** table is optional, and can be used to store additional data about the distributions which does not affect the distribution itself. It is a dictionary of arrays of length (or first dimension) $n_{pdf}$, so that each array has one value or row that corresponds to one distribution.

An example **ancillary** table:

| key        | value                            |
| ---------- | -------------------------------- |
| "gal_name" | `array(["gal1","gal2", "gal3"])` |
| "ids"      | `array([5, 9, 15])`              |

## File structure

When a single Ensemble is stored in a file, it is essentially stored as a `TableDict-like` object (a dictionary of tables as defined in `tables_io`). The keys for the three tables are 'meta', 'data', and 'ancil'. It is written out using `tables_io`, so the supported file types are the same as for [`tables_io`](https://tables-io.readthedocs.io/en/latest/quickstart.html#supported-file-formats).

In an **HDF5** file, these are the 'groupnames', and each of these tables is stored in a 'group', where each column is a 'dataset'.

For a **FITS** file, each table is stored as a separate HDU. For parquet files, each table is stored as a different file, where the file name is 'filename`key`.pq'.

```{note}

To get a better sense of what's in an Ensemble file, check out the example notebook: <project:../nb/ensemble_file.ipynb>.

```

## Parameterization requirements

Each parameterization has its own required keys, and thus the specific set of keys and values will vary depending on the parameterization. For details on the data structure for a parameterization, take a look at its documentation page:

- [**Gaussian mixture model**](./parameterizations/mixmod.md#data-structure)
- [**Histogram**](./parameterizations/hist.md#data-structure)
- [**Interpolation**](./parameterizations/interp.md#data-structure)
- [**Irregular Interpolation**](./parameterizations/irregularinterp.md#data-structure)
- [**Quantiles**](./parameterizations/quant.md#data-structure)
