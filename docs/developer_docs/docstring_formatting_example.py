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
