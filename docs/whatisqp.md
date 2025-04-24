# What is `qp`?

`qp` is a python package for manipulating and storing probability distributions in a variety of parameterization types. It supports both analytically parameterized distributions, such as Gaussian mixture models, and those that are parameterized by data, such as histograms.

The main object of `qp` stores many distributions together in memory, which can then be used to perform a number of statistical functions (i.e. PDF, CDF, PPF, etc). The object can also be written to file, allowing you to easily share distributions with others. `qp` makes it easy to work with large numbers of distributions based on real world data, that don't have neat analytic representations.

`qp` is currently in use in the [RAIL package](https://rail-hub.readthedocs.io/en/latest/index.html), where it is used to handle the probability density functions of the photometric redshift of galaxies and groups of galaxies.

## Use Cases

`qp` has a variety of use cases in science and data analysis, including:

**Standarized Format for Empirical and Analytic Probability Distributions**
: Loading, Manipulating, and Writing large tables of probability distributions in a standardized, reproducible format, including output to HDF5 and FITS files.

**Rapid Statistical Parameter Estimation**
: Calculating common statistical quantities (means, medians, variances, quantiles) of tables of probability distributions through a common interface.

**Parallel Operations on Probability Distributions**
: Running analyses on large tables of probability distributions in parallel using built-in python multiprocessing or MPI for HPC usage.

**Conversion between Probability Distribution Parameterizations**
: Convert probability distributions to common parameterization schemes (including histogram-based, Gaussian mixture models, and interpolated sampling).
