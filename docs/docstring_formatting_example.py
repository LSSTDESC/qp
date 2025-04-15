"""Module docstring.

Note that this module has something to do with `myPackage.otherModule.MyOtherClass`
which we can format and link with single backticks.

Alternatively, to only show the final pathe element, a ~ can be added:
`~myPackage.otherModule.MyOtherClass`.

We can link to any external Python function or class (from a package included in the
intersphinx configuration) by adding the full path: `scipy.stats.norm`. The package does
not need to be imported.

In any case, to provide a link title: `title <MyClass>`.
"""

#### Sphinx Configuration

default_role = "py:obj"  # interpret `function` as crossref to the py object 'function'
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

# autodoc configuration
autoclass_content = "class"  # use the class docstring, not the __init__ docstring
autodoc_type_aliases = {
    # requires `from __future__ import annotations` in all modules
    # if autodoc sees these strings as annotations (not inside the docstrings) it
    # replaces them with the assigned value
    # note that this must be an exact match on the entire annotation. "ArrayLike" will
    # not match on "ArrayLike[float]"
    "ArrayLike": "~numpy.typing.ArrayLike",  # required to stop ArrayLike from expanding in the function signature, can't seem to make it link unfortunately
}

# numpydoc configuration
numpydoc_show_class_members = False  # whether to create an autosummary table at the top of the class documentation
numpydoc_validation_checks = {"SS01"}  # don't do numpydoc validation
numpydoc_xref_param_type = True  # save from having to backtick everything
numpydoc_xref_aliases = {
    # if numpydoc sees these in docstrings, it replaces them with the assigned value
    # unlike autodoc, these replacements will apply on subsets of the type: "Optional"
    # *will* match on "Optional[float]"
    # External types (such as those below) can be added to this with their full path and
    # will then become clickable links
    # note that these are in addition to some defined by numpydoc:
    # https://github.com/numpy/numpydoc/blob/b352cd7635f2ea7748722f410a31f937d92545cc/numpydoc/xref.py#L62-L94
    "Optional": "typing.Optional",
    "Mapping": "typing.Mapping",
    "Union": "typing.Union",
    "ArrayLike": "numpy.typing.ArrayLike",
    "Axes": "matplotlib.axes.Axes",
}
# words that might appear in the type section of a parameter or return value that aren't types
numpydoc_xref_ignore = {"optional", "default", "subclass", "of", "or"}


#### Example Module


class MyClass:
    """Class docstring.

    This will appear in the RDT documentation.

    Parameters
    ----------
    parameter : type
        description

    Examples
    --------
    >>> from module import MyClass
    >>> print("Examples should be plural!")
    Examples should be plural!

    """

    def __init__(self, parameter: type) -> None:
        """Constructor docstring. This will not appear in the RTD documentation.

        Parameters
        ----------
        parameter : type
            description

        """
        pass

    def method(self: "MyClass", dictionary: dict[str, int | float]) -> str:
        """Instance method.

        Parameters
        ----------
        self : MyClass
            No formatting needs to be done on "MyClass" since this is the module where
            it is defined.
        dictionary : dict[str , int | float]
            Note the spaces inside the brackets around the comma and pipe.

        Returns
        -------
        str
            Instead of the type/description pattern used here, the Returns section also
            accepts the name/type/description pattern used in the parameters section.

        """
