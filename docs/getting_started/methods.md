# Ensemble methods

## General use methods

| Method &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [`set_ancil` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.set_ancil)                                       | Set the ancillary data dictionary.                                                                                            |
| [`add_to_ancil` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.add_to_ancil)                                 | Add columns to the ancillary data dictionary.                                                                                 |
| [`convert_to` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.convert_to)                                     | Convert Ensemble to another parameterization                                                                                  |
| [`append` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.append)                                             | Appends another Ensemble to the current one.                                                                                  |
| [`update` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.update)                                             | Re-create the Ensemble with new objdata and metadata, optionally ancillary data.                                              |
| [`update_objdata` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.update_objdata)                             | Re-create the Ensemble with the given objdata and optionally ancillary data.                                                  |
| [`build_tables` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.build_tables)                                 | Returns a dictionary of the metadata, objdata and ancillary data dictionaries, with conversion necessary for writing to file. |
| [`write_to` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.write_to)                                         | Write the Ensemble to a `tables_io` compatible file.                                                                          |
| [`initializeHdf5Write` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.initializeHdf5Write)                   | Sets up an HDF5 file that can be written to iteratively.                                                                      |
| [`writeHdf5Chunk` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.writeHdf5Chunk)                             | Write a chunk of the Ensemble objdata and ancillary data to the HDF5 file.                                                    |
| [`finalizeHdf5Write` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.finalizeHdf5Write)                       | Write Ensemble metadata to the output HDF5 file and close it.                                                                 |
| [`x_samples` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.x_samples)                                       | Returns an array of x values that can be used to plot all the distributions in the Ensemble.                                  |
| [`plot` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.plot)                                                 | Plots the selected distribution as a curve and returns the figure axes.                                                       |
| [`plot_native` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.plot_native)                                   | Plots the selected distribution as the default for that parameterization and returns the figure axes.                         |

## Statistics Methods

| Method                                                                              | Description                                                                                                |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [`norm` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.norm)                     | Normalize the distributions in the Ensemble                                                                |
| [`pdf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.pdf)                       | Returns the value of the probability density function (PDF) for each distribution at the given location(s) |
| [`gridded` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.gridded)               | Return and cache the PDF values at the given grid points.                                                  |
| [`logpdf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.logpdf)                 | Returns the log of the PDF for each distribution in the given location(s).                                 |
| [`cdf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.cdf)                       | Returns the cumulative distribution function (CDF) for each distribution in the given location(s).         |
| [`logcdf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.logcdf)                 | Returns the log of the CDF for each distribution in the given location(s).                                 |
| [`ppf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.ppf)                       | Returns the percentage point function (PPF) for each distribution in the given location(s).                |
| [`sf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.sf)                         | Returns the survival fraction (SF) for each distribution in the given location(s).                         |
| [`logsf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.logsf)                   | Returns the log of the SF for each distribution in the given location(s).                                  |
| [`isf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.isf)                       | Returns the inverse of the SF for each distribution in the given location(s).                              |
| [`rvs` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.rvs)                       | Generate n samples from each distribution.                                                                 |
| [`logpdf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.logpdf)                 | Returns some statistical moments for each of the distributions.                                            |
| [`logpdf` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.logpdf)                 | Returns the log of the PDF for each distribution in the given location(s).                                 |
| [`mode` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.mode)                     | Return the mode of each of the distributions, evaluated on the given or cached grid points.                |
| [`median` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.median)                 | Return the median of each of the distributions.                                                            |
| [`mean` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.mean)                     | Return the means of each of the distributions.                                                             |
| [`var` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.var)                       | Returns the variances for each of the distributions.                                                       |
| [`std` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.std)                       | Returns the standard deviations of each of the distributions.                                              |
| [`moment` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.moment)                 | Returns the nth moment for each of the distributions.                                                      |
| [`entropy` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.entropy)               | Returns the differential entropy for each of the distributions.                                            |
| [`interval` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.interval)             | Returns the intervals corresponding to a confidence level of alpha for each of the distributions.          |
| [`histogramize` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.histogramize)     | Returns integrated histogram bin values for each of the distributions.                                     |
| [`integrate` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.integrate)           | Returns the integral under the PDFs between the given limits for each of the distributions.                |
| [`moment_partial` {octicon}`link;0.9em`](#qp.core.ensemble.Ensemble.moment_partial) | Returns the nth moment over a given range for each of the distributions.                                   |
