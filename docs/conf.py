import sys
import os

# Provide path to the python modules we want to run autodoc on
sys.path.insert(0, os.path.abspath("../qp"))

import qp

# Avoid imports that may be unsatisfied when running sphinx, see:
# http://stackoverflow.com/questions/15889621/sphinx-how-to-exclude-imports-in-automodule#15912502
autodoc_mock_imports = ["scipy", "scipy.interpolate", "sklearn"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_nb",
]

myst_enable_extensions = ["colon_fence"]
jupyter_execute_notebooks = "auto"

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == False

if not on_rtd:
    # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# otherwise, readthedocs.org uses their theme by default, so
# no need to specify it.

master_doc = "index"
autosummary_generate = True
autoclass_content = "class"
autodoc_default_flags = ["members", "no-special-members"]
autodoc_member_order = "bysource"


html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"],
}
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]

project = "qp"
author = "Alex Malz and Phil Marshall"
copyright = "2016, " + author
version = "0.1dev"
release = "0.1dev"
# version = qp.__version__
# release = qp.__version__
