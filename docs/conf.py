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
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "myst_nb",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "dollarmath", "attrs_inline", "tasklist"]
myst_heading_anchors = 5
nb_execution_mode = "auto"
nb_execution_allow_errors = True
exclude_patterns = ["_build", "_build/jupyter_execute", "_build/html/_downloads"]
copybutton_exclude = ".linenos, .gp"

# set up autodocs
master_doc = "index"
autosummary_generate = True
autoclass_content = "class"
autodoc_default_flags = ["members", "no-special-members"]
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
}


# html_sidebars = {
#     "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"],
# }
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]
html_theme_options = {"style_external_links": True}

project = "qp"
author = "Alex Malz and Phil Marshall"
copyright = "2016, " + author
version = "0.1dev"
release = "0.1dev"
# version = qp.__version__
# release = qp.__version__
