# Cookbook

This is for instructions on how to do a few commonly performed operations. If these are short they can just be code examples, but some of the more complicated ones will be links to jupyter notebooks

Some examples:

## Creating an `Ensemble` from a `qp.stats` distribution

```{doctest}

>>> import qp
>>> import numpy as np
>>> loc = np.linspace(0.5,1,3)
>>> scale = np.linspace(0.25,0.75,3)
>>> data = {"loc": loc, "scale":scale}
>>> ens_n = qp.stats.norm.create_ensemble(data)

```

## Sampling (?)

## Conversion example

## Iteration example

## Plotting example

## What's in an Ensemble file

- creating a qp ensemble from a scipy stats distribution (or instead include this as a parameterization)
- sampling from a pdf
- adding an ensemble to another ensemble (both a one-distribution ensemble and one with more distributions)
- a conversion example, where we convert from one parameterization to another and then back again and compare the output (jupyter notebook)
- iterating over a qp ensemble
- plotting a specific pdf from an ensemble
- read in ensemble file in h5py, look at what's in the file, and then read it in qp
