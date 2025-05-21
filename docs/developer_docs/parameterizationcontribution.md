# Creating new parameterizations

Before creating a new parameterization, we recommend you take a look at how some of the existing supported parameterizations are written (i.e. {py:class}`qp.hist <qp.parameterizations.hist.hist.hist_gen>`, {py:class}`qp.interp <qp.parameterizations.interp.interp.interp_gen>`). This will help give you a sense of how a parameterization class should function. Make sure you follow the contribution workflows described in <project:contribution.md>.

## The basics

The [parameterization template](./parameterization_template.py) provides an easy to use starting point for creating a new parameterization. To get started, make a copy of it in a folder inside the `parameterizations/` folder (see <project:contribution.md#naming-and-placement> for instructions on naming). You can copy the file `parameterization_template.py` from the `docs/developer_docs/` folder.

Once you have done so, follow the instructions in the template to get started on your new parameterization. Keep in mind when writing functions that there exist some conversion functions that may be applicable to your parameterization in the `utils/conversion.py` file, and some generically useful interpolation functions that work with the inputs given to `_pdf()` and `_cdf()` functions can be found in `utils/interpolation.py`. Make sure to check the `utils/` directory for a variety of helpful functions when writing your parameterization.

```{tip}
Keep in mind that `ppf(0)` and `ppf(1)` return negative infinity and positive infinity respectively for all of the parameterizations written in `qp`.
```

### Testing

Once you've written your new parameterization, make sure to add a test data generation function in the `tests/helpers/test_data_helper.py` file. You can instead add a test data file to the `tests/test_data/` folder, and then add a function to read it in in the `tests/helpers/test_data_helper.py` file.

There are automatic tests that will run on all parameterizations with test data in the `tests/helpers/test_data_helper.py` file, which will test basic Ensemble functions. We recommend that you write additional tests in a separate test file to ensure that parameterization-specific functionality works as expected.

The test data should be a dictionary of sub-dictionaries, where each sub-dictionary is a test data dictionary. This allows you to include multiple test data sets for each parameterization. For an example, see the `hist_test_data` example in the `tests/helpers/test_data_helper.py` file.

Each test data sub-dictionary should contain the following keys:

- **gen_func**: the parameterization class to use
- **ctor_data**: a dictionary with the data used to create an Ensemble of this parameterization
- **convert_data**: a dictionary with keys that are the arguments you would use to convert to this parameterization (can use **method** to specify which conversion method to use).

Optional keys:

- **ancil**: Any ancillary data that you want to add to the test Ensemble
- **test_pdf**: If `test_auto` will run the basic Ensemble method tests (i.e. {py:meth}`qp.Ensemble.pdf`, {py:meth}`qp.Ensemble.cdf`). By default True.
- **test_persist**: If `test_auto` will test that Ensemble read and write functionality returns the same Ensemble. By default True.
- **test_convert**: If `test_auto` will test conversion to this parameterization from others. By default True.
- **test_plot**: If `test_auto` will test plotting of this parameterization. By default True.
- **test_xvals**: The x values used when testing Ensemble methods (i.e. {py:meth}`qp.Ensemble.pdf`)
- **filekey**: a key that differentiates the files written out for different test data dictionary
- **do_samples**: If true, will run {py:func}`qp.plotting.plot_pdf_samples_on_axes()` during plotting tests. By default False.
- **npdf**: the number of distributions in the Ensemble. If not included will use the built in method to get `npdf`.
- **atol_diff**: Used in conversion tests as the allowed tolerance between converted Ensembles when using {py:meth}`qp.Ensemble.convert_to`, by default $1 \times 10^{-2}$
- **atol_diff2**: Used in the conversion test as the allowed tolerance between converted Ensembles when using {py:meth}`qp.convert() <qp.factory.Factory.convert>`, by default $1 \times 10^{-2}$

## Documentation

Once your parameterization is functional and has been tested, we recommend that you add it to the API documentation. You can add it to the `docs/api_docs/parameterizations.rst` file using the following markdown format:

```markdown
## [One line description of parameterization]

.. autoclass :: qp.[parameterization_name]\_gen
:members:
:show-inheritance:
:undoc-members:

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.[parameterization_name].[parameterization_name]\_utils
:members:
```

We recommend discussing with other developers before adding a new parameterization documentation page to the `docs/user_guide/parameterizations/` folder, however, as that documentation covers only parameterizations that are fully supported by the developers. To add a page, copy the format used for other parameterizations.
