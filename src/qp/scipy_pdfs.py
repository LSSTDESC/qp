"""Module to define qp distributions that inherit from scipy distributions

Notes
-----

In the qp distribtuions the last axis in the
input array shapes is reserved for pdf parameters.

This is because qp deals with numerical representations
of distributions, where some of the input parameters consist
of arrays of values for each pdf.

`scipy.stats` assumes that all input parameters scalars for each pdf.

To ensure that `scipy.stats` based distributions behave the same
as `qp` distributions we are going to insure that the all input
variables have shape either (npdf, 1) or (1)
"""

import numpy as np

from qp.test_data import LOC, SCALE, TEST_XVALS
from qp.factory import stats

# pylint: disable=no-member
stats.norm_gen.test_data = dict(
    norm=dict(
        gen_func=stats.norm,
        ctor_data=dict(loc=LOC, scale=SCALE),
        test_xvals=TEST_XVALS,
        do_samples=True,
        ancil=dict(zmode=LOC),
    ),
    norm_shifted=dict(
        gen_func=stats.norm, ctor_data=dict(loc=LOC, scale=SCALE), test_xvals=TEST_XVALS
    ),
    norm_multi_d=dict(
        gen_func=stats.norm,
        ctor_data=dict(loc=np.array([LOC, LOC]), scale=np.array([SCALE, SCALE])),
        test_xvals=TEST_XVALS,
        do_samples=True,
    ),
)
