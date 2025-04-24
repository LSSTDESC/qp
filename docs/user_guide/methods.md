# Ensemble Methods

The following tables are lists of existing methods for use on Ensembles, with short summaries of their purpose. The methods link to their API documentation for more detailed information.

## Base Methods

These are some base methods used to perform operations on one or more Ensembles.

| Method                                                           | Description                                                                |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| {py:meth}`qp.create() <qp.factory.Factory.create>`               | Creates an Ensemble from a dictionary of data.                             |
| {py:meth}`qp.from_tables() <qp.factory.Factory.from_tables>`     | Creates an Ensemble from a dictionary of tables.                           |
| {py:meth}`qp.read_metadata() <qp.factory.Factory.read_metadata>` | Reads just an Ensemble's metadata from a file.                             |
| {py:meth}`qp.is_qp_file() <qp.factory.Factory.is_qp_file>`       | Tests if a file is a `qp` file containing a metadata table.                |
| {py:meth}`qp.read() <qp.factory.Factory.read>`                   | Reads an Ensemble from a file.                                             |
| {py:meth}`qp.data_length() <qp.factory.Factory.data_length>`     | Gets the number of distributions present in the file.                      |
| {py:meth}`qp.iterator() <qp.factory.Factory.iterator>`           | Iterates through an Ensemble in a given file.                              |
| {py:meth}`qp.convert() <qp.factory.Factory.convert>`             | Converts an Ensemble to a different parameterization.                      |
| {py:meth}`qp.concatenate() <qp.factory.Factory.concatenate>`     | Concatenates a list of Ensembles with the same metadata into one Ensemble. |
| {py:meth}`qp.write_dict() <qp.factory.Factory.write_dict>`       | Writes a dictionary of Ensembles to an HDF5 file.                          |
| {py:meth}`qp.read_dict() <qp.factory.Factory.read_dict>`         | Reads a dictionary of Ensembles from an HDF5 file.                         |
| {py:meth}`qp.add_class() <qp.factory.Factory.add_class>`         | Adds a new parameterization class to the dictionary of classes.            |

## Methods of the Ensemble class

### General use methods

These are more methods that are used to perform operations on Ensembles, from creating them to writing them to file. These are methods of an existing Ensemble, and so you would write `ens.set_ancil(ancil)`, for example.

| Method &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| {py:meth}`set_ancil() <qp.Ensemble.set_ancil>`                                                                  | Set the ancillary data dictionary.                                                                                            |
| {py:meth}`add_to_ancil() <qp.Ensemble.add_to_ancil>`                                                            | Add columns to the ancillary data dictionary.                                                                                 |
| {py:meth}`convert_to() <qp.Ensemble.convert_to>`                                                                | Convert Ensemble to another parameterization                                                                                  |
| {py:meth}`append() <qp.Ensemble.append>`                                                                        | Appends another Ensemble to the current one.                                                                                  |
| {py:meth}`update() <qp.Ensemble.update>`                                                                        | Re-create the Ensemble with new objdata and metadata, optionally ancillary data.                                              |
| {py:meth}`update_objdata() <qp.Ensemble.update_objdata>`                                                        | Re-create the Ensemble with the given objdata and optionally ancillary data.                                                  |
| {py:meth}`build_tables() <qp.Ensemble.build_tables>`                                                            | Returns a dictionary of the metadata, objdata and ancillary data dictionaries, with conversion necessary for writing to file. |
| {py:meth}`write_to() <qp.Ensemble.write_to>`                                                                    | Write the Ensemble to a `tables_io` compatible file.                                                                          |
| {py:meth}`initializeHdf5Write() <qp.Ensemble.initializeHdf5Write>`                                              | Sets up an HDF5 file that can be written to iteratively.                                                                      |
| {py:meth}`writeHdf5Chunk() <qp.Ensemble.writeHdf5Chunk>`                                                        | Write a chunk of the Ensemble objdata and ancillary data to the HDF5 file.                                                    |
| {py:meth}`finalizeHdf5Write() <qp.Ensemble.finalizeHdf5Write>`                                                  | Write Ensemble metadata to the output HDF5 file and close it.                                                                 |
| {py:meth}`x_samples() <qp.Ensemble.x_samples>`                                                                  | Returns an array of x values that can be used to plot all the distributions in the Ensemble.                                  |
| {py:meth}`plot() <qp.Ensemble.plot>`                                                                            | Plots the selected distribution as a curve and returns the figure axes.                                                       |
| {py:meth}`plot_native() <qp.Ensemble.plot_native>`                                                              | Plots the selected distribution as the default for that parameterization and returns the figure axes.                         |

### Statistics Methods

Many of these methods are inherited from [`scipy.stats.rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous). Some of these methods, such as `pdf()`, inherit from the SciPy method, but may have parameterization-specific functions to calculate the values. Others are inherited more directly.

| Method                                                   | Description                                                                                                                                                                                                                                           |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {py:meth}`norm() <qp.Ensemble.norm>`                     | Normalize the distributions in the Ensemble                                                                                                                                                                                                           |
| {py:meth}`pdf() <qp.Ensemble.pdf>`                       | Returns the value of the probability density function (PDF) for each distribution at the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.pdf.html#scipy.stats.rv_continuous.pdf)). |
| {py:meth}`gridded() <qp.Ensemble.gridded>`               | Return and cache the PDF values at the given grid points.                                                                                                                                                                                             |
| {py:meth}`logpdf() <qp.Ensemble.logpdf>`                 | Returns the log of the PDF for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.logpdf.html#scipy.stats.rv_continuous.logpdf)).                            |
| {py:meth}`cdf() <qp.Ensemble.cdf>`                       | Returns the cumulative distribution function (CDF) for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.cdf.html#scipy.stats.rv_continuous.cdf)).          |
| {py:meth}`logcdf() <qp.Ensemble.logcdf>`                 | Returns the log of the CDF for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.logcdf.html#scipy.stats.rv_continuous.logcdf)).                            |
| {py:meth}`ppf() <qp.Ensemble.ppf>`                       | Returns the percentage point function (PPF) for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html#scipy.stats.rv_continuous.ppf)).                 |
| {py:meth}`sf() <qp.Ensemble.sf>`                         | Returns the survival fraction (SF) for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.sf.html#scipy.stats.rv_continuous.sf)).                            |
| {py:meth}`logsf() <qp.Ensemble.logsf>`                   | Returns the log of the SF for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.logsf.html#scipy.stats.rv_continuous.logsf)).                               |
| {py:meth}`isf() <qp.Ensemble.isf>`                       | Returns the inverse of the SF for each distribution in the given location(s) (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.isf.html#scipy.stats.rv_continuous.isf)).                               |
| {py:meth}`rvs() <qp.Ensemble.rvs>`                       | Generate n samples from each distribution (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.rvs.html#scipy.stats.rv_continuous.rvs)).                                                                  |
| {py:meth}`stats() <qp.Ensemble.stats>`                   | Returns some statistical moments for each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.stats.html#scipy.stats.rv_continuous.stats)).                                         |
| {py:meth}`mode() <qp.Ensemble.mode>`                     | Return the mode of each of the distributions, evaluated on the given or cached grid points.                                                                                                                                                           |
| {py:meth}`median() <qp.Ensemble.median>`                 | Return the median of each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.median.html#scipy.stats.rv_continuous.median)).                                                       |
| {py:meth}`mean() <qp.Ensemble.mean>`                     | Return the means of each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.mean.html#scipy.stats.rv_continuous.mean)).                                                            |
| {py:meth}`var() <qp.Ensemble.var>`                       | Returns the variances for each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.var.html#scipy.stats.rv_continuous.var)).                                                        |
| {py:meth}`std() <qp.Ensemble.std>`                       | Returns the standard deviations of each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.std.html#scipy.stats.rv_continuous.std)).                                               |
| {py:meth}`moment() <qp.Ensemble.moment>`                 | Returns the nth moment for each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.moment.html#scipy.stats.rv_continuous.moment)).                                                 |
| {py:meth}`entropy() <qp.Ensemble.entropy>`               | Returns the differential entropy for each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.entropy.html#scipy.stats.rv_continuous.entropy)).                                     |
| {py:meth}`interval() <qp.Ensemble.interval>`             | Returns the intervals corresponding to a confidence level of alpha for each of the distributions (see also [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.interval.html#scipy.stats.rv_continuous.interval)). |
| {py:meth}`histogramize() <qp.Ensemble.histogramize>`     | Returns integrated histogram bin values for each of the distributions.                                                                                                                                                                                |
| {py:meth}`integrate() <qp.Ensemble.integrate>`           | Returns the integral under the PDFs between the given limits for each of the distributions.                                                                                                                                                           |
| {py:meth}`moment_partial() <qp.Ensemble.moment_partial>` | Returns the nth moment over a given range for each of the distributions.                                                                                                                                                                              |
