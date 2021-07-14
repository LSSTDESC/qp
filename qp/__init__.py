"""qp is a library for manaing and converting between different representations of distributions"""

import os

try:
    from .version import get_git_version
    __version__ = get_git_version()
except Exception as message: #pragma: no cover
    print(message)

from .spline_pdf import *
from .hist_pdf import *
from .interp_pdf import *
from .quant_pdf import *
from .mixmod_pdf import *
from .sparse_pdf import *
from .scipy_pdfs import *
from .ensemble import Ensemble
from .factory import instance, add_class, create, read, convert, concatenate

from . import utils
from . import test_funcs
