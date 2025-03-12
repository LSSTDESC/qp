# Basic Usage

## Introduction to the Ensemble

The main object of `qp` is the [`qp.Ensemble`](#qp.core.ensemble.Ensemble). This is a data structure that can store one or more distributions with the same type, or **parameterization**. A **parameterization** is defined in this package as the way that the distribution is represented, for example a histogram represents a data set using bins and values inside those bins.

An `Ensemble` object has three main data components, which exist as dictionaries:

- **Metadata** (`Ensemble.metadata`)
  - This tells you the parameters shared by all distributions in the `Ensemble`, including the name of the parameterization, the version, and the coordinates of that parameterization (i.e. bins for a histogram).
- **Data values** (`Ensemble.objdata`)
  - Contains the data values for each distribution in the `Ensemble`, where each row in a value's array corresponds to a distribution.
- _(optional)_ **Ancillary data table** (`Ensemble.ancil`)
  - Contains any additional parameters pertaining to the data, where there is one value for each distribution. The arrays for each parameter must have the same first dimension as the number of distributions.

```{note}
The exact configuration of the data within these dictionaries differs for each parameterization, so see <project:../advanced_usage/datastructure.md> for more details.
```

## Creating an Ensemble

There are three available methods to create an `Ensemble` from in-memory data, as well as the ability to read in an `Ensemble` from file.

(creating-ensemble-memory)=

### Creating an Ensemble from in-memory data

The first method is to use the `create_ensemble` function that exists for each parameterization. For example, to create an interpolated parameterization, where the distributions are given by a set of x and y values, you can use [`qp.interp.create_ensemble`](#qp.parameterizations.interp.interp.interp_gen.create_ensemble), where the data is passed as arguments to the function. See below for an example:

```{doctest}

>>> import qp
>>> import numpy as np
>>> xvals= np.array([0,0.5,1,1.5,2]),
>>> yvals = np.array([[0.01, 0.2,0.3,0.2,0.01],[0.09,0.25,0.2,0.1,0.01]])
>>> ancil = {'ids':[5,8]}
>>> ens = qp.interp.create_ensemble(xvals, yvals,ancil=ancil)
>>> ens.metadata
{'pdf_name': array([b'interp'], dtype='|S6'),
'pdf_version': array([0]),
'xvals': array([[0. , 0.5, 1. , 1.5, 2. ]])}

```

Another method is to use [`qp.create`](#qp.core.factory.create), which allows you to create an `Ensemble` of any parameterization type. The function requires the parameterization type as an argument, as well as a dictionary of the necessary data, and an optional `ancil` argument for any ancillary data. So to create an `Ensemble` using the same data above, you would use the following commands:

```{doctest}

>>> data = {"xvals": xvals, "yvals": yvals}
>>> ens = qp.create('interp', data=data, ancil=ancil)

```

Finally, you can instantiate the `Ensemble` class directly by using [`qp.Ensemble`](#qp.core.ensemble.Ensemble), which takes the same arguments as the [`qp.create`](#qp.core.factory.create) method, except the parameterization argument must be the actual class, instead of the string name of the class:

```{doctest}

>>> ens = qp.Ensemble(qp.interp, data=data,ancil=ancil)

```

### Reading an Ensemble from file

An `Ensemble` can be read from a file as well, if the file is in the appropriate format. To check if a file can be read in as an `Ensemble`, you can use [`qp.is_qp_file(filename)`](#qp.core.factory.is_qp_file), which returns `True` if it is in a compatible format. Once you have a file that can be read it, you can use [`qp.read`](#qp.core.factory.read) as shown in the example below:

```{doctest}

>>> import qp
>>> ens = qp.read("ensemble.hdf5")
>>> ens
Ensemble(the_class=interp,shape=(2,5))

```

## Working with an Ensemble

### Attributes of an Ensemble

Now that we have an `Ensemble`, we can check the data it contains using `ens.metadata` or `ens.objdata`. These show the dictionaries of data that define our `Ensemble`. Let's check the `objdata` for the `Ensemble` we created above:

```{doctest}

>>> ens.objdata
{'yvals': array([[0.02816901, 0.56338028, 0.84507042, 0.56338028, 0.02816901],
        [0.3       , 0.83333333, 0.66666667, 0.33333333, 0.03333333]])}

```

Note that these `yvals` are different than the ones we provided in the [section above](#creating-ensemble-memory). This is because the data stored is not the input data, but the data used to create the `Ensemble`, and in this case, as is true for many parameterizations, the data is normalized by default. So `objdata` returns the normalized data.

An `Ensemble` has other attributes that provide information about it, including `ens.npdf`, which tells you how many distributions it contains, or `ens.shape`, which returns the shape of the `objdata`, (`npdf`, `ncoord`), where `ncoord` is the number of values that each distribution has, usually corresponding in some way to the number of coordinates in the metadata.

### Important methods

What can you do with an `Ensemble`? <project:methods.md> lists all of the available methods of an `Ensemble` object, and links to their docstrings. Or you can see the [API documentation of the class](#qp.core.ensemble.Ensemble) for a complete list of its attributes and methods all in one place. Here we will go over a few of the most commonly-used methods.

#### Statistical methods

One of the main functions of an `Ensemble` is the ability to calculate the probability distribution function (PDF) or cumulative distribution function (CDF) of the distributions. This can be done via the `ens.pdf` and `ens.cdf` methods, which return values that correspond to the given x values for each distribution. For example, to get the value of the PDFs at a specific x value, one can do the following:

```{doctest}

>>> ens.pdf(1.2)
array([[0.73239437],
       [0.53333333]])

```

This returns an array of shape (`npdf`, `nxval`), where `nxval` is the number of x values given to the function.

#### Conversion

It is possible to convert an `Ensemble` of distributions of one parameterization to a different parameterization. For example, let's say we wanted to convert our `Ensemble` from `interp` to a histogram (`hist`). We can do this using [`qp.convert`](#qp.core.factory.convert), which takes as arguments the `Ensemble` to convert and the name of the parameterization we want to convert to. You can also provide a specific conversion method via the `method` keyword, if the parameterization has more than one conversion method. Most conversion methods also have additional required arguments, which will differ for the parameterization as well as the specific method being used.

To get more information about the existing conversion methods and arguments, we can take a look at the docstrings of the parameterization class (via `qp.hist?` or `help(qp.hist)` in the command line). They tell us that for the [`hist` parameterization](#qp.parameterizations.hist.hist.hist_gen), there are two conversion methods, `extract_hist_values` and `extract_hist_samples`. For this example we'll use `extract_hist_values`, which requires the `bins` argument.

```{doctest}

>>> bins = np.linspace(np.min(xvals),np.max(xvals),10)
>>> ens_hist = qp.convert(ens, 'hist', bins)
>>> ens_hist
Ensemble(the_class=hist,shape=(2, 9))

```

Our new `Ensemble` has a different class and a different shape, as now instead of 5 `xvals` we have 9 `bins` (and 10 bin edges).

However, converting an `Ensemble` does not guarantee that the converted `Ensemble` will have _exactly_ the same distribution shape. For example, we can compare the value of the PDF at `x=1.2` in the `hist` parameterized `Ensemble` to that of the `interp` parameterized `Ensemble`:

```{doctest}

>>> ens_hist.pdf(1.2)
array([[0.70921986],
       [0.54054054]])

>>> ens.pdf(1.2)
array([[0.73239437],
       [0.53333333]])

```

As you can see, these values are slightly different. In this case, they are close enough, but depending on the scenario there can be larger differences. Typically, ensuring that your `Ensembles` have a higher density of coordinate values, and that your conversions have similarly high density, will aid in producing converted distributions that match their initial distributions more closely.

:::{note}

You can only convert to a parameterization that has a conversion method. This means that any parameterization inherited from `scipy` (i.e. any parameterization that starts with `qp.stats`) cannot be converted to.

:::

- objdata, metadata, and ancil tables
- Show a couple of the more important methods that can be called, link to the ensemble methods page which lists all of them
- basics of how to convert to different parameterizations (more detail elsewhere)
  - mention you can't convert to scipy parameterizations
- calculating metrics (basics, table of existing supported metrics)

## Writing an Ensemble to file

- Writing out a qp ensemble
  - format file is normally written to
  - see data structure for more information about what's written
  - see cookbook for example
