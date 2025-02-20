import pytest
import qp
import numpy as np


def test_check_input():
    bins = np.linspace(-2, 2, 11)
    print(bins)
    pdfs = np.array([-0, -0.5, -1, -0.5, -0.5, -1.25, -1.5, -0.75, -0.5, -0.2])
    data = {"bins": bins, "pdfs": pdfs, "check_input": False}
    ens_h = qp.hist.create_ensemble(data)
