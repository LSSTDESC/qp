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

The main object of `qp` is the [`qp.Ensemble`](#qp.Ensemble). This is an object that stores one or more distributions of the same type, or parameterization. It has three main data dictionaries:

- **Metadata** (`Ensemble.metadata`)
  - tells you the shared parameters, including the parameterization type
- **Data values** (`Ensemble.objdata`)
  - the data values for each distribution, where one row = one distribution
- _(optional)_ **Ancillary data table** (`Ensemble.ancil`)
  - any additional data for the distributions, where there must be one row for every distribution

The printed representation of an `Ensemble` tells you the parameterization type and the shape of the arrays in the `objdata`, which is essentially (`npdf`, `nvals`), where `npdf` is the number of distributions and `nvals` is the number of values or data points for each distribution in the `Ensemble`.

### Creating an Ensemble

To create an `Ensemble` of any number of distributions, you can use the `create_ensemble` method of any of the existing parameterizations. For example, to create an `Ensemble` of interpolations:

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

You can also read an `Ensemble` from a file using [`qp.read`](#qp.core.factory.Factory.read). For example:

```{doctest}

>>> ens_r = qp.read("ensemble.hdf5")
>>> ens_r
Ensemble(the_class=interp,shape=(3, 50))

```

### Working with an Ensemble

Now that you have created an `Ensemble`, you can get a sense of what's in it by using some useful attributes:

- `ens.npdf`: Number of distributions in the `Ensemble`
- `ens.shape`: Shape of the data, (`npdf`, `nvals`)
- `ens.metadata` : The metadata dictionary
- `ens.objdata` : The data dictionary
- `ens.ancil` : The ancillary data (will not work if there is no ancillary data table)

You can also use the available [`Ensemble` methods](methods.md). These allow you to manipulate your `Ensemble`, or get statistical information about your `Ensembles`. For example, here are some useful methods:

- [`ens.pdf()`](#qp.Ensemble.pdf)
- [`ens.cdf()`](#qp.Ensemble.cdf)
- [`]

### Saving an `Ensemble` for later

To write an `Ensemble` to file, simply use the [`ens.write_to`](#qp.Ensemble.write_to) method. The only required argument is the full path to the file you would like to write. For example, we can write out the `Ensemble` we created like so:

```{doctest}

>>> ens.write_to("ensemble-test.hdf5")

```

The available file formats can be found in <project:basicusage.md#writing-an-ensemble-to-file>.

## Where to find more information

For a statistical refresher that covers a lot of the concepts used in this documentation, see <project:qpprimer.md>. For more detailed information on how to use `qp`, see <project:basicusage.md>. <project:cookbook.md> has more detailed examples for specific use cases, such as conversion and plotting. If you're looking for more information on one of the supported parameterizations, you can take a look at their documentation pages:

- <project:hist.md>
- <project:quant.md>
- <project:interp.md>
- <project:mixmod.md>
