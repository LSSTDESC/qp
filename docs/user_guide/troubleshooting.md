# Troubleshooting

This page covers a range of common issues and errors that you may encounter.

## Scipy update

This code inherits a number of classes from <inv:#scipy.stats>, most importantly `rv_continuous` and `rv_frozen`. If you are suddenly experiencing issues where the code was previously working, it may be that SciPy has an update that has broken something.

## Conversion

- Converting from Gaussian mixed models to another parameterization using any method that requires sampling (i.e. "samples" method for histogram) is currently not functional, use the default method instead.
- Converting to Gaussian mixed models relies on sampling and then fitting the sampled data, so it does not provide consistent outputs.
- When converting to a quantile parameterization from most other parameterizations, your quantiles cannot include 0 or 1. Give values as close to 0 and 1 as possible to avoid infinite values.

## Parameterizations

- Quantile

  - PDF interpolations can have negative values with "dual spline average" and "cdf spline average" constructors, particularly near the edges of the distribution

- Gaussian mixed models

  - {py:meth}`qp.Ensemble.rvs()` method is not currently functional
