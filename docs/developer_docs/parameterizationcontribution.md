# Creating new parameterizations

Before creating a new parameterization, we recommend you take a look at how some of the existing supported parameterizations are written (i.e. `hist`, `interp`). This will help give you a sense of how a parameterization class should function. Make sure you follow the contribution workflows described in <project:contribution.md>.

## The basics

To

- init function
  - should include functions to store values needed, pass pdf shape to base class constructor
  - should call addmetadata and addobjdata methods
- functions to access each of the data and metadata fields
- \_pdf and \_cdf hook functions
  - useful functions to help implement pdfs can be found in utils files
- implement \_updated_ctor_param function
- define functions to convert other ensembles to this representation
- register class with the factor and make function available

### Testing

Once you've written your new parameterization, make sure to add a test data generation function in the `tests/helpers/test_data_helper.py` file. You can instead add a test data file to the `tests/test_data/` folder, and then add a function to read it in in the `tests/helpers/test_data_helper.py` file.

The test data should be of the following structure:

- dictionary of test data dictionaries. This allows you to have multiple sets of test data for each parameterization.

  - each test data dictionary should contain the following keys:
    - **gen_func**: the parameterization class to use
    - **ctor_data**: a dictionary with the data used to create an `Ensemble` of this parameterization
    - **convert_data**: a dictionary with keys that are the arguments you would use to convert to this parameterization (can use **method** to specify which conversion method to use).
  - optional keys:
    - **ancil**: Any ancillary data that you want to add to the test `Ensemble`
    - **test_pdf**: If `test_auto` will run the basic tests for the parameterization (i.e. testing the basic `Ensemble` methods work). By default True.
    - **test_persist**: If `test_auto` will test out `Ensemble` read and write functionality. By default True.
    - **test_convert**: If `test_auto` will test conversion to this parameterization from others. By default True.
    - **test_plot**: If `test_auto` will test plotting of this parameterization. By default True.
    - **test_xvals**: The x values used when testing `Ensemble` methods (i.e. `.pdf()`)
    - **filekey**: a key that differentiates the files written out for different test data dictionary
    - **do_samples**: If true, will run `qp.plotting.plot_pdf_samples_on_axes()` during plotting tests. By default False.
    - **npdf**: the number of distributions in the `Ensemble`. If not included will use the built in method to get `npdf`.
    - **atol_diff**: Used in conversion tests as the allowed tolerance between converted `Ensembles` when using `ens.convert_to()`, by default $1 \times 10^{-2}$
    - **atol_diff2**: Used in the conversion test as the allowed tolerance between converted `Ensembles` when using `qp.convert()`, by default $1 \times 10^{-2}$

The automatic tests will run on all parameterizations with test data, which will test basic `Ensemble` functions. We recommend that you write additional tests in a separate test file to ensure that parameterization specific functionality works as expected.

## Optional

- \_sf, \_ppf, \_isf, \_rvs functions for faster evaluation
  - Keep in mind that the `_ppf()` function will return -inf at 0 and inf at 1, and the `_ppf()` function will not be called at all to return those values.
- plotting (plot_native)

## Code template

Below is a code template to use that outlines required and optional functionality for parameterizations. You can copy it from the `docs/developer_docs` folder and use it to start your new parameterization, or [download it here](./parameterization_template.py).

```{literalinclude} parameterization_template.py

```
