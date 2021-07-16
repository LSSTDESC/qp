"""qp is a library for manaing and converting between different representations of distributions"""

import os
from .version import __version__

from .spline_pdf import *
from .hist_pdf import *
from .interp_pdf import *
from .quant_pdf import *
from .mixmod_pdf import *
from .sparse_pdf import *
from .scipy_pdfs import *
from .ensemble import Ensemble
from .factory import instance, add_class, create, read, convert

from . import utils
from . import test_funcs
