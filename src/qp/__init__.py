"""qp is a library for managing and converting between different representations of distributions"""

import os

# from ...tests.qp import test_funcs
from .version import __version__

from .parameterizations.spline.spline import *
from .parameterizations.hist.hist import *
from .parameterizations.interp.interp import *
from .parameterizations.quant.quant import *
from .parameterizations.analytic.mixmod.mixmod import mixmod_gen
from .parameterizations.sparse_interp.sparse import *
from .parameterizations.analytic.scipy_dists_import import *
from .parameterizations.packed_interp.packed_interp import *
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
