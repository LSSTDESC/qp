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

- Creating a qp ensemble

  - from data in a dictionary or table
  - from file

- Working with a qp ensemble

  - objdata, metadata, and ancil tables
  - Show a couple of the more important methods that can be called, link to the ensemble methods page which lists all of them
  - basics of how to convert to different parameterizations (more detail elsewhere)
    - mention you can't convert to scipy parameterizations
  - calculating metrics (basics, table of existing supported metrics)

- Writing out a qp ensemble
  - format file is normally written to
  - see data structure for more information about what's written
  - see cookbook for example
