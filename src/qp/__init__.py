"""qp is a library for managing and converting between different representations of distributions"""

import os

# from ...tests.qp import test_funcs
from .version import __version__

from .parameterizations.spline.spline import spline, spline_ensemble, spline_gen
from .parameterizations.hist.hist import hist, hist_ensemble, hist_gen
from .parameterizations.interp.interp import (
    interp,
    interp_ensemble,
    interp_gen,
    interp_irregular,
    interp_irregular_ensemble,
    interp_irregular_gen,
)
from .parameterizations.quant.quant import quant, quant_ensemble, quant_gen

from .parameterizations.analytic.mixmod.mixmod import (
    mixmod,
    mixmod_ensemble,
    mixmod_gen,
)
from .parameterizations.sparse_interp.sparse import sparse, sparse_ensemble, sparse_gen

# from .parameterizations.analytic.scipy_testdata import *
from .parameterizations.packed_interp.packed_interp import (
    packed_interp,
    packed_interp_ensemble,
    packed_interp_gen,
)

from .parameterizations import analytic

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


from .utils import array

from .parameterizations.packed_interp import packing_utils

from .utils import dictionary

from .core import factory
