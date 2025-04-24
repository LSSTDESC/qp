# Conversion

This page contains example use cases of converting Ensembles to different parameterizations.

## Converting Ensembles between parameterizations

This tutorial covers converting between a few different parameterizations and how this affects the distributions in the Ensemble across different parameterizations and conversion methods: <project:../../nb/conversion_example.md> (download {download}`here <../../nb/conversion_example.ipynb>`).

## Converting a Gaussian Ensemble to a histogram

First we make our Gaussian Ensemble:

```{doctest}

>>> ens_n = qp.stats.norm.create_ensemble(dict(loc=np.array([0,0.5]),
... scale=np.array([0.5,0.25])))

```

Then we convert it to a histogram Ensemble using the default conversion method, which requires the `bins` argument:

```{doctest}

>>> ens_h = qp.convert(ens_n, 'hist', bins=np.linspace(-1,1,10))
>>> ens_h

```

For more details on the available methods for conversion to a histogram parameterization and how they work, see <project:../parameterizations/hist.md>.

## Converting a Gaussian Ensemble to an interpolation

First we make our Gaussian Ensemble:

```{doctest}

>>> ens_n = qp.stats.norm.create_ensemble(dict(loc=np.array([0,0.5]),
... scale=np.array([0.5,0.25])))

```

Then we convert it to a interpolation Ensemble using the default conversion method, which requires the `xvals` argument:

```{doctest}

>>> ens_i = qp.convert(ens_n, 'interp', xvals=np.linspace(-1,1,10))
>>> ens_i
Ensemble(the_class=interp,shape=(2, 10))

```

For more details on the available methods for conversion to a interpolation parameterization and how they work, see <project:../parameterizations/interp.md>.

## Converting a Gaussian Ensemble to an irregular interpolation

First we make our Gaussian Ensemble:

```{doctest}

>>> ens_n = qp.stats.norm.create_ensemble(dict(loc=np.array([0,1]),
... scale=np.array([0.5,0.25])))

```

Then we convert it to an irregular interpolation Ensemble using the default conversion method, which requires the `xvals` argument:

```{doctest}

>>> ens_i = qp.convert(ens_n, 'interp_irregular', xvals=np.array([np.linspace(-1,1,10),np.linspace(0,2,10)]))
>>> ens_i
Ensemble(the_class=interp_irregular,shape=(2, 10))

```

For more details on the available methods for conversion to an irregular interpolation parameterization and how they work, see <project:../parameterizations/irregularinterp.md>.

## Converting a Gaussian Ensemble to a quantile

First we make our Gaussian Ensemble:

```{doctest}

>>> ens_n = qp.stats.norm.create_ensemble(dict(loc=np.array([0,0.5]),
... scale=np.array([0.5,0.25])))

```

Then we convert it to a quantile Ensemble using the default conversion method, which requires the `quants` argument:

```{doctest}

>>> ens_q = qp.convert(ens_n, 'quant', quants=np.linspace(0.0001, 0.9999, 10))
>>> ens_q
Ensemble(the_class=quant,shape=(2, 12))

```

Since we did not provide 0 or 1, the shape of the quantiles has been expanded to include 0 and 1, resulting in 12 quantiles instead of 10. For more details on the available methods for conversion to a quantile parameterization and how they work, see <project:../parameterizations/quant.md>.

## Converting a histogram Ensemble to a Gaussian mixture model

First we make our histogram Ensemble:

```{doctest}

>>> bins = np.linspace(0,2,10)
>>> pdfs = np.array([[0.1, 0.3, 0.5, 0.6, 0.65, 0.6, 0.5, 0.3, 0.1],
... [0.1, 0.2, 0.25, 0.2, 0.3, 0.4, 0.5, 0.3, 0.1]])
>>> ens_h = qp.hist.create_ensemble(bins=bins,pdfs=pdfs)

```

Then we convert it to a Gaussian mixture model Ensemble using the default conversion method, using the optional argument `ncomps` to determine how many Gaussians will be used for each distribution:

```{doctest}

>>> ens_m = qp.convert(ens_h, "mixmod", ncomps=2)
>>> ens_m
Ensemble(the_class=mixmod,shape=(2, 2))

```

For more details on the available methods for conversion to a Gaussian mixture model parameterization and how they work, see <project:../parameterizations/mixmod.md>.
