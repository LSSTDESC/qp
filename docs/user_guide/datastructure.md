# Data Structure

## Basic structure

- basic figure of the main three components and what they include

- metadata table is dictionary of arrays
  - pdf_name, pdf_version, and coordinates, where in the example below the coordinates key is 'bins'.

| key         | value              | dtype      |
| ----------- | ------------------ | ---------- |
| pdf_name    | `array(b["hist"])` | np.ndarray |
| pdf_version | `array([0])`       | np.ndarray |
| bins        | `array([1,2,3])`   | np.ndarray |

- data table is dictionary of arrays
  - typically data has one 2d array of data values, with a shape (num_pdfs (rows), len(coordinates) (columns))

| key  | value                             |
| ---- | --------------------------------- |
| var1 | array with shape (npdfs, ncoords) |

- ancil table is optional, when it exists it is a dictionary of arrays where the first dimension of those arrays = npdf

| key  | value                     |
| ---- | ------------------------- |
| col1 | array with shape (npdfs,) |

## Structure for each parameterization

- data schema is slightly different for each parameterization
- for all parameterizations that are not continuous, mostly just the 'coordinates' in the metadata table and the corresponding 'data values' in the data table have different names
- the format of the data itself within those places is roughly the same

### Histogram parameterization

- coordinates are 'bins'
- data is 'pdfs', number of columns is len(bins) - 1 instead of length of bins

### Quantile parameterization

- coordinates are 'quants'
- data is 'locs'
- metadata table includes pdf_constructor_name and check_input

### Mixed model parameterization

data table
weights, stds, means

### Interpolated parameterization

- coordinates are 'xvals'
- data is 'yvals'

## File structure

- how the data is stored in a file
  - i.e. keys for each table in an hdf5 file
  - accepted file formats
  - normal written file format
- include an example of how to create such a file outside of qp and have it read by qp (or link to it in cookbook)
