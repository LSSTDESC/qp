# Roadmap

## This version

### Breaking changes

- file reorganization means that any functions that were previously accessed via full path may not be available
- `qp.[parameterization]` now aliases to the parameterization class instead of its create function. However, Ensemble creation has been updated to take the class and not the creation function, so Ensemble functionality should be unbroken. If you want to specify a specific creation method, it takes the `method` argument.
- changed `qp.Ensemble.objdata()` and `qp.Ensemble.metadata()` calls to be properties, to match `qp.Ensemble.ancil`
- `qp.Ensemble.metadata['coord']`, where 'coord' is the coordinates value (i.e. bins or xvals), has been changed from a 2D array to a 1D array in memory. It will still be changed to a 2D array for writing to file.
- `check_input` has been changed as a parameter to be more descriptive, so it is `norm` or `ensure_extent` as appropriate
- output of `qp.Ensemble` statistics functions, i.e. `pdf`, will have their dimensionality dictated by the dimensionality of the given Ensemble and the input value, as in <inv:#scipy.stats>.
- A single distribution Ensemble will return 1D arrays in the `objdata` dictionary in memory. This is true for any sliced Ensembles as well.
- for parameterizations based on <inv:#scipy.stats> distributions, input is automatically reshaped into arrays that are ($n_{pdf}$, 1) in shape, to ensure that the Ensemble object behaves similarly across all parameterization types
- `pdf_constructor_name` (for the quantile parameterization) is now byte-encoded and able to be written to a file

## Moving forward

- separate out `metrics` into its own package
- separate out `irregular_interp` and `interp` parameterizations into separate folders
- possibly change `qp.Ensemble.npdf` to `qp.Ensemble.ndist`, part of larger change away from calling individual distribution objects PDFs
- update the tests to use `pytest` consistently instead of `unittest`
- have a metadata translation layer to user datatypes (i.e. `ens.metadata['pdf_name']` returns a string not an array of a byte-encoded string)
