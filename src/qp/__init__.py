"""qp is a library for managing and converting between different representations of distributions"""

import os

# from ...tests.qp import test_funcs
from .version import __version__

from .parameterizations.spline.spline_pdf import *
from .parameterizations.hist.hist_pdf import *
from .parameterizations.interp.interp_pdf import *
from .parameterizations.quant.quant_pdf import *
from .parameterizations.analytic_parameterizations.mixmod.mixmod_pdf import *
from .parameterizations.sparse_interp.sparse_pdf import *
from .parameterizations.analytic_parameterizations.scipy_dists_import import *
from .parameterizations.packed_interp.packed_interp_pdf import *
from .core.ensemble import Ensemble
from .core.factory import (
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
from .core.lazy_modules import *

from .utils import array_funcs

from .parameterizations.packed_interp import packing_utils

from .utils import dict_funcs

from .core import factory
