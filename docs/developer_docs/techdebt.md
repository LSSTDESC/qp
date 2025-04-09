# Development Priorities

- Essentially a list of places where things should be updated or changed
- link to the github issues page
- can start out with the easier to do things and then put the larger things in a separate section?

- if you have done one of these things, remember to check it off this list

## Functions that exist but are not implemented/functional

- `Ensemble.mix_mod_fit` - a docstring exists but the function is not implemented
- the 'sparse' conversion method for {py:class}`qp.interp_irregular` does not seem to be functional to convert any type of input `Ensemble` to the irregular interpolation parameterization.

## Functionality that is partially supported

- The {py:class}`qp.mixmod` parameterization is not completely functional. `rvs()` method does not work on `mixmod`. The reason seems to be that `mixmod`'s {py:meth}`ppf() <qp.mixmod._ppf>` method does not allow `CASE_2D` inputs
- `add_reader_method` is an option, and is supported by `qp.read` in `factory.py`, but other functions that read data in (like `iterator` or `read_dict`) do not call this reader method, so if someone were to use the `reader_method` it would likely result in bugs at the moment. Support needs to be added to the other read functions, or the `reader_method` option could be removed.
