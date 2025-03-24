# Cookbook

This page provides some more in-depth examples of commonly performed routines with `qp`. The longer ones are given as `jupyter notebooks`, which can be viewed here or downloaded from <project:../nb/index.md>.

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

We provided an array of 3 values each for 'loc' and 'scale' and ended up with an `Ensemble` with 3 distributions. This is due to the automatic reshaping of the input arrays that `qp` does to ensure that the resulting `Ensemble` looks and behaves like the other `Ensemble` types. `qp.stats` distributions are analytic, unlike most of the other parameterizations, so they only require one value for each of the parameters per distribution. This means that you can input the data either as 1D arrays (above), or as 2D arrays (below), and both result in the same outcome.

```{doctest}

>>> loc = np.array([[0.5],[0.75],[1.0]])
>>> scale = np.array([[0.25],[0.5],[0.75]])
>>> data = {"loc": loc, "scale":scale}
>>> ens_n = qp.stats.norm.create_ensemble(data)
>>> ens_n
Ensemble(the_class=norm, shape=(3,1))

```

## Sampling (?)

## Appending an `Ensemble` to another `Ensemble` of the same type

If you have created multiple `Ensembles` with the same parameterization, it can be easier to handle if they are all part of one `Ensemble`. You can use the [`append()`](#qp.Ensemble.append) method of the `Ensemble` to add an `Ensemble` of the same type to an existing `Ensemble`. This can only be done if the metadata for both `Ensembles` are the same. In particular, the coordinates (i.e. "xvals" or "bins") must be the same for both `Ensembles`. For example, to append two histogram `Ensembles` together:

```{doctest}

>>> import qp
>>> bins =
>>> pdfs =
>>> ens_1 = qp.hist.create_ensemble(bins=bins,pdfs=pdfs)
>>> ens_1
Ensemble()
>>> pdfs_2 =
>>> ens_2 = qp.hist.create_ensemble(bins=bins,pdfs=pdfs_2)
>>> ens_2
Ensemble()
>>> ens.append(ens_2)
>>> ens
Ensemble()

```

## Conversion example

- notebook

## Iteration example

See the full tutorial here: <project:../nb/iterator_demo.md>

## Plotting using x_samples

- notebook?

A useful method for quickly plotting a distribution or distributions in your `Ensemble` is the `x_samples()` method. This is meant to provide a series of x values that should cover the range of data given in all distributions in the `Ensemble`, which can be provided to the appropriate method (`pdf()` for most parameterizations, `cdf()` for quantiles) to get the relevant y values.

:::{note}
This method only does this for the four main supported parameterizations. For the rest it returns just a default set of points between a given minimum and maximum value.
:::

Plotting a CDF from a quantile distribution:

```{doctest}

>>> import qp
>>> import matplotlib.pyplot as plt
>>> quants =
>>> locs =
>>> ens_q = qp.quant.create_ensemble(quants=quants, locs=locs)
>>> ens_q.x_samples()

>>> plt.plot(ens_q.x_samples(), ens_q[0].cdf(ens_q.x_samples))
>>> plt.xlabel("x")
>>> plt.ylabel("CDF(x)")
>>> plt.show()

```

Plotting a PDF from an interpolated distribution:

```{doctest}

>>> xvals =
>>> yvals =
>>> ens_i = qp.interp.create_ensemble(xvals=xvals, yvals=yvals)
>>> ens_i.x_samples()

>>> plt.plot(ens_i.x_samples(), ens_i[0].pdf(ens_i.x_samples()))
>>> plt.xlabel("x")
>>> plt.ylabel("P(x)")
>>> plt.show()

```

## What's in an Ensemble file

See the full tutorial here: <project:../nb/ensemble-file.md>

- creating a qp ensemble from a scipy stats distribution (or instead include this as a parameterization)
- sampling from a pdf
- adding an ensemble to another ensemble (both a one-distribution ensemble and one with more distributions)
- a conversion example, where we convert from one parameterization to another and then back again and compare the output (jupyter notebook)
- iterating over a qp ensemble
- plotting a specific pdf from an ensemble
- read in ensemble file in h5py, look at what's in the file, and then read it in qp
