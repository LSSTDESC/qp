# Histogram

## Introduction

A histogram represents a distribution with two sets of data: a set of bin edges, and a set of data values within those bins. An example is plotted below.

- figure of what an example of this parameterization looks like

### Creating an Ensemble

**Required parameters**: bins, pdfs
**Optional parameters**: ancil, norm, warn

```{doctest}

>>> import qp
>>> import numpy as np
>>> bins = np.linspace(0,1,5)
>>> pdfs = np.array([0.1,0.2,0.2,0.1])
>>> ens = qp.hist.create_ensemble(bins=bins,pdfs=pdfs)

```

- how to make an ensemble of histograms

## Data structure

### Metadata table

The **metadata** table is a dictionary of arrays. For this parameterization it contains the required two keys, as well as the coordinates key "bins", which gives the **bin edges**, not the bin centers. Thus, the length of this array should be `nbins+1`, where `nbins` is the number of bins. An example is shown below:

| key           | value              |
| ------------- | ------------------ |
| "pdf_name"    | `array(b["hist"])` |
| "pdf_version" | `array([0])`       |
| "bins"        | `array([0,1,2,3])` |

### Data table

The **data** table is a dictionary with one key, the "pdfs", which give the values within each bin. The shape of the array is (npdf, nbins), where nbins is the number of bins, or the number of bin edges - 1.

| key    | value                              |
| ------ | ---------------------------------- |
| "pdfs" | `array([[4,5,6],[1,2,3],[7,8,9]])` |

## Conversion

There are two methods that can be used to convert an `Ensemble` to this parameterization: the [default method](#qp.parameterizations.hist.hist_utils.extract_hist_values), and [samples](#qp.parameterizations.hist.hist_utils.extract_hist_samples).

### Default method

**Required argument:** `bins`, where `bins` are the bin edges of the histogram.

The default method works by taking the current distribution, getting the CDF values at each of the bin edges, and then taking the difference across the bins to get the value for each bin.

### Samples method

**Required argument:** `bins`, where `bins` are the bin edges of the histogram.

**Optional argument:** `size`, which is the number of values to sample from the distribution, default = 1000.

This method samples from the distribution, and then uses `np.histogram` to create a histogram from these samples, using the provided `bins` as bin edges. This does mean that too few samples may result in quite a different distribution.

- how to convert an ensemble to this parameterization
  - need `bins`
- details of how this parameterization works
- any known issues
