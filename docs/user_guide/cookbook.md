# Cookbook

## Creating an `Ensemble` from a `qp.stats` distribution

The `create_ensemble` function for the `qp.stats` distributions is slightly different than for the other parameterizations, as it requires a dictionary of the data to create the Ensemble, instead of being able to take individual arguments. See below for an example using the `qp.stats.norm` distribution:

```{doctest}

>>> import qp
>>> import numpy as np
>>> loc = np.linspace(0.5,1,3)
>>> scale = np.linspace(0.25,0.75,3)
>>> data = {"loc": loc, "scale":scale}
>>> ens_n = qp.stats.norm.create_ensemble(data)
>>> ens_n
Ensemble(the_class=norm, shape=(3,1))

```

We provided an array of 3 values each for 'loc' and 'scale' and ended up with an `Ensemble` with 3 distributions. This is due to the automatic reshaping of the input arrays that is done to ensure that the resulting `Ensemble` looks and behaves like the other `Ensembles`. `qp.stats` distributions are analytic, unlike most of the other parameterizations, and therefore only require one value for each of the parameters per distribution. This means that you can input the data either as 1D arrays as above, or as 2D arrays as shown below, and both result in the same outcome.

```{doctest}

>>> loc = np.array([[0.5],[0.75],[1.0]])
>>> scale = np.array([[0.25],[0.5],[0.75]])
>>> data = {"loc": loc, "scale":scale}
>>> ens_n = qp.stats.norm.create_ensemble(data)
>>> ens_n
Ensemble(the_class=norm, shape=(3,1))

```

## Sampling (?)

## Conversion example

- notebook

## Iteration example

- notebook?

## Plotting using x_samples

- notebook?

A useful function for quickly plotting a distribution or distributions in your `Ensemble` is the `x_samples` function. This is meant to provide a series of x values that should cover the range of data given in all distributions in the `Ensemble`, which can be provided to the appropriate function (`pdf` for most parameterizations, `cdf` for quantiles) to get the relevant y values.

:::{note}
This function only does this for the four main supported parameterizations. For the rest it returns just a default set of points between a given minimum and maximum value.
:::

Plotting a CDF from a quantile distribution:

```{doctest}

>>> import qp
>>> import matplotlib.pyplot as plt
>>> quants =
>>> locs =

```

Plotting a pdf from an interpolated distribution:

```{doctest}

>>> xvals =
>>> yvals =

```

## What's in an Ensemble file

- notebook

- creating a qp ensemble from a scipy stats distribution (or instead include this as a parameterization)
- sampling from a pdf
- adding an ensemble to another ensemble (both a one-distribution ensemble and one with more distributions)
- a conversion example, where we convert from one parameterization to another and then back again and compare the output (jupyter notebook)
- iterating over a qp ensemble
- plotting a specific pdf from an ensemble
- read in ensemble file in h5py, look at what's in the file, and then read it in qp
