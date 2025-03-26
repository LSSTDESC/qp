# Troubleshooting

This page covers a range of common issues and errors that you may encounter.

## Scipy update

This code inherits a number of classes from `scipy.stats`, specifically `rv_continous` and `rv_frozen`. If you are suddenly experiencing issues where the code was previously working, it may be that SciPy has an update that has broken something.

## Conversion

- Converting from Gaussian mixed models to another parameterization using any method that requires sampling (i.e. "samples" method for histogram) is currently not functional, use the default method instead
- Converting to Gaussian mixed models relies on sampling and fitting the input distribution and does not provide consistent outputs.
- Converting to a quantile parameterization from norm or Gaussian mixed model parameterizations, your quantiles cannot include 0 or 1. Give values as close to 0 and 1 as possible to avoid infinite values

## Parameterizations

- Quantile

  - PDF interpolation can go into negative values unnecessarily with "dual spline average" and "cdf spline average" constructors.

- Gaussian mixed models

  - `rvs()` method is not currently functional
