# Roadmap

## This version

### Breaking changes

- file reorganization means that any functions that were previously accessed via full path may not be available
- `qp.parameterization()` no longer aliases to a create function, which means that you cannot create a single distribution object using this call. However, Ensemble creation has been updated to take the class and not the creation function, so `Ensemble` functionality should be unbroken
- changed `qp.Ensemble.objdata()` and `qp.Ensemble.metadata()` calls to be properties, to match `qp.Ensemble.ancil`

## Moving forward

- separate out `metrics` into its own package
- separate out `irregular_interp` and `interp` parameterizations into separate folders
- possibly change `npdf` to `ndist`, part of larger change away from calling individual distribution objects PDFs
- update the tests to use `pytest` instead of `unittest`
