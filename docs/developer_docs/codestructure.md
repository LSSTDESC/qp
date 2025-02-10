# Code Structure

- diagram of how the code is structured
- short description of the code and how it works, etc

Below is a diagram of the current code structure, and some descriptions of the purpose of files and folders.

```bash
qp
├── __init__.py
├── _version.py
├── data # contains sample data for tests (to be moved to tests)
│   ├── CFHTLens_sample.P.npy
│   └── test.hdf5
├── ensemble.py # contains the class defining the qp ensemble
├── lazy_modules.py
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
│   └── point_estimate_metric_classes.py
├── parameterizations # contains files or folders for code that set up qp parameterizations
│   ├── __init__.py
│   ├── analytic_parameterizations # folder for analytic qp parameterizations
│   │   ├── __init__.py
│   │   ├── scipy_dists_import.py # imports scipy distributions as qp parameterizations
│   │   └── mixmod_pdf.py # mixed model gaussian parameterization
│   ├── base_parameterization.py # the parameterization classes used to create other qp parameterizations
│   ├── hist_pdf.py # histogram parameterization
│   ├── interp_pdf.py # interpolated parameterization
│   ├── mixmod_pdf.py
│   ├── packed_interp # code for the packed interpolated parameterization
│   │   ├── __init__.py
│   │   ├── packed_interp_pdf.py
│   │   └── packing_utils.py
│   ├── quant # code to create the quantile parameterization
│   │   ├── __init__.py
│   │   ├── abstract_pdf_constructor.py # pdf constructor method
│   │   ├── cdf_spline_derivative.py # pdf constructor method
│   │   ├── dual_spline_average.py # pdf constructor method
│   │   ├── piecewise_constant.py # pdf constructor method
│   │   ├── piecewise_linear.py # pdf constructor method
│   │   └── quant_pdf.py
│   ├── sparse_interp # code to create the sparse interpolated parameterization
│   │   ├── __init__.py
│   │   ├── sparse_pdf.py
│   │   └── sparse_rep.py # support functions
│   └── spline_pdf.py # the spline parameterization
├── plotting.py # plotting qp ensembles
├── utils # utility functions
│   ├── __init__.py
│   ├── conversion_funcs.py # functions to convert between parameterizations
│   ├── dict_utils.py # utility functions for handling dictionaries
│   ├── factory.py # utility functions for dealing with qp ensembles
│   ├── test_data.py # functions to generate test data
│   └── utils.py # generic utility functions
└── version.py
```
