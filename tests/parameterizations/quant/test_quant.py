import numpy as np
import pytest

import qp


def test_quant_constructor():
    """Make sure that quant parameterization doesn't work if the constructor name
    is not in the given list."""
    quants = np.linspace(0, 1, 5)
    locs = np.linspace(-1, 1, 5)
    pdf_constructor_name = "piecewise"
    with pytest.raises(ValueError) as exec_info:
        qp.quant.create_ensemble(quants, locs, pdf_constructor_name)
    assert exec_info.type is ValueError
