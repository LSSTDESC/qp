# Development Priorities

On this page, current known issues and technical debt will be recorded. These issues would make good first contributions for new contributors to the project. You can also take a look at the [issues](https://github.com/LSSTDESC/qp/issues) page for more.

Make sure to check an item off the list if you have completed it.

## Code debt

- [ ] `Ensemble.mix_mod_fit` - a docstring exists but the function is not implemented. Either remove completely or implement.
- [ ] The 'sparse' conversion method for {py:class}`qp.interp_irregular` does not seem to be functional to convert any type of input `Ensemble` to the irregular interpolation parameterization.
- [ ] The {py:class}`qp.mixmod` parameterization is not completely functional. `rvs()` method does not work on `mixmod`. The reason seems to be that `mixmod`'s {py:meth}`ppf() <qp.mixmod._ppf>` method does not allow `CASE_2D` inputs
- [ ] `add_reader_method` is an option within a parameterization class to add a function that will be called when reading in an `Ensemble` of that type. It is supported by {py:meth}`qp.read() <qp.factory.Factory.read>` in `factory.py`, but other functions that read data in (like {py:meth}`qp.iterator() <qp.factory.Factory.iterator>` or {py:meth}`qp.read_dict() <qp.factory.Factory.read_dict>`) do not call this reader method, so if someone were to use `add_reader_method` it would likely result in bugs at the moment. Support needs to be added to the other read functions, or the `add_reader_method` option could be removed.
- [ ] Migrate tests over to `pytest` fully, and remove `unittest` import
