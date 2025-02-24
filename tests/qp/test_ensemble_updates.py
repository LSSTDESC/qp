import pytest
import qp
import numpy as np


def test_check_input():
    bins = np.linspace(-2, 2, 11)
    print(bins)
    pdfs = np.array([0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2])
    norm = False
    ancil = {"ancil": [3]}
    ens_h = qp.hist.create_ensemble(bins, pdfs, norm=norm, ancil=ancil)

    assert ens_h.npdf == 1

    ens_h.norm()
    assert ens_h.objdata()["pdfs"][0, 1] != pdfs[1]
    assert ens_h.ancil == ancil


def test_norm():
    x = np.linspace(0, 5, 10)
    y = np.array([0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2])
    data = {"xvals": x, "yvals": y}
    ens_i = qp.interp_irregular.create_ensemble(x, y)

    ens_i.norm()


def test_quant():
    quants = np.linspace(0, 1, 5)
    locs = np.linspace(-1, 1, 5)
    pdf_constructor_name = "piecewise"
    ens_q = qp.quant.create_ensemble(quants, locs, pdf_constructor_name)

    assert ens_q.npdf == 1
