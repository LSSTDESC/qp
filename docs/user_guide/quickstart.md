# Quickstart

## Installation

To install `qp`, you can run the following commands:

```bash

git clone https://github.com/LSSTDESC/qp.git
cd qp
pip install .

```

For more information on alternate installation methods see <project:installation.md>.

## How to use `qp`

`qp` is meant to allow users to store and manipulate sets of probability distributions that can be represented, or parameterized, in different ways. The different parameterization options allow users to store analytically parameterized distributions, such as Gaussian mixture models, in addition to more data-based parameterizations such as histograms or distributions interpolated from a set of $x$ and $y$ values. It also allows for use of any of the `scipy.stats.rv_continuous` distributions as parameterization types under the `qp.stats` module. For a full list of supported parameterizations and more detailed explanations, see <project:./parameterizations/index.md>.

The main object of `qp` that stores these distributions is the {py:class}`qp.Ensemble`. It can store one or more distributions of the same parameterization. It has three main data dictionaries:

- **Metadata** ({py:attr}`qp.Ensemble.metadata`)
  - tells you the shared parameters, including the parameterization type
- **Data values** ({py:attr}`qp.Ensemble.objdata`)
  - the data values for each distribution, where one row = one distribution
- _(optional)_ **Ancillary data table** ({py:attr}`qp.Ensemble.ancil`)
  - any additional data for the distributions, where there must be one row for every distribution

The printed representation of an `Ensemble` tells you the parameterization type and the shape of the arrays in the `objdata`, which is essentially ($n_{pdf}$, $n_{vals}$), where $n_{pdf}$ is the number of distributions and $n_{vals}$ is the number of values or data points for each distribution in the `Ensemble`.

### Creating an Ensemble

To create an `Ensemble` of any number of distributions, you can use the `create_ensemble` method of any of the existing parameterization classes. For example, to create an `Ensemble` of interpolations:

```{doctest}

>>> import qp
>>> import numpy as np
>>> xvals = np.array([0,0.5,1,1.5,2])
>>> yvals = np.array([[0.01,0.2,0.3,0.2,0.01],
... [0.1,0.3,0.5,0.2,0.05]])
>>> ens = qp.interp.create_ensemble(xvals,yvals)
>>> ens
Ensemble(the_class=interp,shape=(2,5))
```

You can also read an `Ensemble` from a file using {py:meth}`qp.read <qp.factory.Factory.read>`. For example:

```{doctest}

>>> ens_r = qp.read("ensemble.hdf5")
>>> ens_r
Ensemble(the_class=interp,shape=(3, 50))

```

### Working with an Ensemble

Now that you have created an `Ensemble`, you can get a sense of what's in it by using some useful attributes:

- {py:attr}`qp.Ensemble.npdf`: Number of distributions in the `Ensemble`
- {py:attr}`qp.Ensemble.shape`: Shape of the data, ($n_{pdf}$, $n_{vals}$)
- {py:attr}`qp.Ensemble.metadata` : The metadata dictionary
- {py:attr}`qp.Ensemble.objdata` : The data dictionary
- {py:attr}`qp.Ensemble.ancil` : The ancillary data (will not work if there is no ancillary data table)

You can also use the available [`Ensemble` methods](methods.md). These allow you to manipulate your `Ensemble`, or get statistical information about your `Ensembles`. For example, here are some useful methods:

- {py:meth}`qp.Ensemble.pdf`: Get the PDF values at specified $x$ values
- {py:meth}`qp.Ensemble.cdf`: Get the CDF values at specified $x$ values
- {py:meth}`qp.Ensemble.convert`: Convert the `Ensemble` to a different parameterization

### Saving an `Ensemble` for later

To write an `Ensemble` to file, simply use the {py:meth}`qp.Ensemble.write_to` method. The only required argument is the full path to the file you would like to write. For example, we can write out the `Ensemble` we created like so:

```{doctest}

>>> ens.write_to("ensemble-test.hdf5")

```

The available file formats can be found in <project:basicusage.md#writing-an-ensemble-to-file>.

## Where to find more information

- <project:qpprimer.md> covers some of the statistics basics necessary for using `qp`
- <project:basicusage.md> provides more detailed explanations of topics discussed here
- <project:./cookbook/index.md> contains detailed examples for specific use cases, i.e. conversion, plotting, iteration
- <project:./parameterizations/index.md> contains more details on specific parameterizations
- <project:troubleshooting.md> for common pitfalls and errors
