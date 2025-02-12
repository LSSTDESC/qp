# Code Structure

- diagram of how the code is structured
- short description of the code and how it works, etc

Below is a diagram of the current code structure, and some descriptions of the purpose of files and folders.

```bash

qp
├── __init__.py
├── _version.py
├── core
│   ├── __init__.py
│   ├── ensemble.py
│   ├── factory.py
│   └── lazy_modules.py
├── data # contains sample data for tests (to be moved to tests)
│   ├── CFHTLens_sample.P.npy
│   └── test.hdf5
├── metrics # this module contains functions to calculate metrics on qp objects
│   ├── __init__.py
│   ├── array_metrics.py
│   ├── base_metric_classes.py
│   ├── brier.py
│   ├── concrete_metric_classes.py
│   ├── factory.py
│   ├── goodness_of_fit.py
│   ├── metrics.py
│   ├── parallel_metrics.ipynb
│   ├── pit.py
│   ├── point_estimate_metric_classes.py
│   └── util_funcs.py
├── parameterizations # contains code for all parameterizations
│   ├── __init__.py
│   ├── analytic_parameterizations # code for analytic parameterizations
│   │   ├── __init__.py
│   │   ├── mixmod # mixed model Gaussian parameterization
│   │   │   ├── __init__.py
│   │   │   ├── mixmod_pdf.py
│   │   │   └── mixmod_utils.py
│   │   └── scipy_dists_import.py # imports scipy dists as parameterizations
│   ├── base_parameterization.py # base classes for all parameterizations
│   ├── hist # histogram parameterization
│   │   ├── __init__.py
│   │   ├── hist_pdf.py
│   │   └── hist_utils.py
│   ├── interp # interpolated parameterization
│   │   ├── __init__.py
│   │   ├── interp_pdf.py
│   │   └── interp_utils.py
│   ├── packed_interp # interpolated parameterization stored as packed integers
│   │   ├── __init__.py
│   │   ├── packed_interp_pdf.py
│   │   └── packing_utils.py
│   ├── quant # quantile parameterization
│   │   ├── __init__.py
│   │   ├── abstract_pdf_constructor.py
│   │   ├── cdf_spline_derivative.py
│   │   ├── dual_spline_average.py
│   │   ├── piecewise_constant.py
│   │   ├── piecewise_linear.py
│   │   ├── quant_pdf.py
│   │   └── quant_utils.py
│   ├── sparse_interp # sparse parameterization
│   │   ├── __init__.py
│   │   ├── sparse_pdf.py
│   │   ├── sparse_rep.py
│   │   └── sparse_utils.py
│   └── spline # spline parameterization
│       ├── __init__.py
│       ├── spline_pdf.py
│       └── spline_utils.py
├── plotting.py # creates plots
├── utils # utility functions used throughout qp
│   ├── __init__.py
│   ├── array_utils.py # performing array operations
│   ├── conversion_funcs.py # utilities for converting between parameterizations and unused functions
│   ├── dict_utils.py # performing dictionary operations
│   ├── interp_funcs.py # interpolation functions
│   └── test_data.py # generates test data (to be moved)
└── version.py

```
