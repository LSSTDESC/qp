import pytest
import numpy as np

import qp


def test_irreg():
    x = np.linspace(0, 5, 10)
    y = np.array([0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2])
    data = {"xvals": x, "yvals": y}
    ens_i = qp.interp_irregular.create_ensemble(x, y)

    # ens_i.norm()
    assert ens_i.shape == (1, 10)

    ens_i.set_ancil({"name": ["irreg1"]})

    ens_i.cdf(2)
