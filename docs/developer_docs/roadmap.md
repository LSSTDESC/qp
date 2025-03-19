# Roadmap

## This version

### Breaking changes

- file reorganization means that any functions that were previously accessed via full path may not be available
- `qp.parameterization()` no longer aliases to a create function, which means that you cannot create a single distribution object using this call. However, Ensemble creation has been updated to take the class and not the creation function, so `Ensemble` functionality should be unbroken
- changed `qp.Ensemble.objdata()` and `qp.Ensemble.metadata()` calls to be properties, to match `qp.Ensemble.ancil`
- `qp.Ensemble.metadata()['coord']`, where 'coord' is the coordinates value (i.e. bins or xvals), has been changed from a 2D array to a 1D array
- `check_input` has been changed as a parameter to be more descriptive, so it is `norm` or `ensure_extent` as appropriate
- output of `qp.Ensemble` statistics functions, i.e. `pdf`, will have their dimensionality dictated by the dimensionality of the given `Ensemble` and the input value, as in `scipy.stats`.
- when slicing a `qp.Ensemble` for a single distribution, it will return 1D arrays in the `objdata` dictionary.
- for parameterizations based on `scipy.stats` distributions, input is automatically reshaped into arrays that are (npdf, 1) in shape, to ensure that the `Ensemble` object behaves similarly to the other parameterization types

## Moving forward

- separate out `metrics` into its own package
- separate out `irregular_interp` and `interp` parameterizations into separate folders
- possibly change `npdf` to `ndist`, part of larger change away from calling individual distribution objects PDFs
- update the tests to use `pytest` consistently instead of `unittest`
- have a metadata translation layer to user datatypes (i.e. `ens.metadata['pdf_name']` returns a string not an array)
