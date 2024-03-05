"""qp is a library for managing and converting between different representations of distributions"""

import os
from .version import __version__

from .spline_pdf import *
from .hist_pdf import *
from .interp_pdf import *
from .quant_pdf import *
from .mixmod_pdf import *
from .sparse_pdf import *
from .scipy_pdfs import *
from .packed_interp_pdf import *
from .ensemble import Ensemble
from .factory import (
    instance,
    add_class,
    create,
    read,
    read_metadata,
    convert,
    concatenate,
    iterator,
    data_length,
    from_tables,
    is_qp_file,
    write_dict,
    read_dict,
)
from .lazy_modules import *

from . import utils

from . import packing_utils

from . import test_funcs
