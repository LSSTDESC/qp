import sys
import os

# Provide path to the python modules we want to run autodoc on
sys.path.insert(0, os.path.abspath("../qp"))

import qp

# set up readthedocs theme
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"


# Avoid imports that may be unsatisfied when running sphinx, see:
# http://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule#15912502
autodoc_mock_imports = ["scipy", "scipy.interpolate", "sklearn"]

# set up extensions

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "attrs_inline",
    "tasklist",
    "deflist",
]
myst_heading_anchors = 5
nb_execution_mode = "auto"
nb_execution_allow_errors = True
exclude_patterns = [
    "_build",
    "_build/jupyter_execute",
    "_build/html/_downloads",
]  # where not to render notebook files to markdown
copybutton_exclude = ".linenos, .gp"  # what to exclude from copied code blocks

# set up autodocs
master_doc = "index"
autosummary_generate = True
autoclass_content = "class"
autodoc_default_flags = ["members", "no-special-members"]
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    # requires `from __future__ import annotations` in all modules
    "ArrayLike": "~numpy.typing.ArrayLike",  # required to stop ArrayLike from expanding in the function signature, can't seem to make it link unfortunately
}

numpydoc_show_class_members = False
numpydoc_validation_checks = {"SS01"}
numpydoc_xref_param_type = True  # save from having to backtick everything
numpydoc_xref_aliases = {
    "Optional": "typing.Optional",  # required for links in docstrings
    "Mapping": "typing.Mapping",  # required for links in docstrings
    "Union": "typing.Union",  # required for links in docstrings
    "Any": "typing.Any",  # required for links in docstrings
    "List": "typing.List",  # required for links in docstrings
    # external
    "ArrayLike": "numpy.typing.ArrayLike",  # required for links in docstrings
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    "sps.gaussian_kde": "scipy.stats.gaussian_kde ",
    "np.uint8": "numpy.uint8",
    # qp classes: lets you write just "ClassName" in files other than where ClassName is defined
    # otherwise you'd need to write what's defined below
    "Pdf_gen": "`~qp.parameterizations.base.Pdf_gen`",
    "Ensemble": "`~qp.Ensemble`",
    "AbstractQuantilePdfConstructor": "`~qp.parameterizations.quant.abstract_pdf_constructor.AbstractQuantilePdfConstructor`",
}
numpydoc_xref_ignore = {
    "optional",
    "default",
    "subclass",
    "of",
    "or",
    "length",
    "size",
    "shape",
    "containing",
    "distributions",
}


# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
default_role = "py:obj"  # interpret `function` as crossref to the py object 'function'


# html_sidebars = {
#     "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"],
# }
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
    "custom.css",
]
html_theme_options = {"style_external_links": True}

project = "qp"
author = "DESC RAIL Team"
copyright = "2025, " + author
version = qp.__version__
release = qp.__version__
