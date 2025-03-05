import pytest
import qp
import numpy as np


def test_norm():
    bins = np.linspace(-2, 2, 11)
    print(bins)
    pdfs = np.array(
        [
            [0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2],
            [0.05, 0.09, 0.15, 0.2, 0.3, 0.5, 0.25, 0.15, 0.1, 0.025],
        ]
    )
    norm = False
    ancil = {"ancil": [3, 4]}
    ens_h = qp.hist.create_ensemble(bins, pdfs, norm=norm, ancil=ancil)

    assert ens_h.npdf == 2

    ens_h.norm()
    assert ens_h.objdata["pdfs"][0, 1] != pdfs[0, 1]
    assert ens_h.ancil == ancil


def test_encode_decode_strings(tmp_path):
    """Tests that the encoding and decoding of string ancil data columns for HDF5 files works with
    write_to and read."""
    bins = np.linspace(-2, 2, 11)
    pdfs = np.array(
        [
            [0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2],
            [0.05, 0.09, 0.15, 0.2, 0.3, 0.5, 0.25, 0.15, 0.1, 0.025],
        ]
    )
    names = np.array(["gal1", "gal2"])
    ancil = {"names": names}
    ens_h = qp.hist.create_ensemble(bins, pdfs, ancil=ancil)

    # write to file
    file_path = tmp_path / "test-encode.hdf5"
    ens_h.write_to(file_path)

    # read from file
    new_ens = qp.read(file_path)
    assert new_ens.ancil["names"][0] == names[0]


def test_xsamples():
    """Test that the x_samples functionality is working properly"""
    bins = np.linspace(-2, 2, 11)
    pdfs = np.array(
        [
            [0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2],
            [0.05, 0.09, 0.15, 0.2, 0.3, 0.5, 0.25, 0.15, 0.1, 0.025],
        ]
    )

    ens_h = qp.hist.create_ensemble(bins, pdfs)
    xvals = ens_h.x_samples()

    assert np.min(xvals) > np.min(bins)
    assert len(bins) == (len(xvals) + 1)


def test_irreg():
    x = np.linspace(0, 5, 10)
    y = np.array([0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2])
    data = {"xvals": x, "yvals": y}
    ens_i = qp.interp_irregular.create_ensemble(x, y)

    # ens_i.norm()
    assert ens_i.shape == (1, 10)

    ens_i.set_ancil({"name": ["irreg1"]})

    ens_i.cdf(2)


def test_quant():
    """Make sure that quant parameterization doesn't work if the constructor name
    is not in the given list."""
    quants = np.linspace(0, 1, 5)
    locs = np.linspace(-1, 1, 5)
    pdf_constructor_name = "piecewise"
    with pytest.raises(ValueError) as exec_info:
        qp.quant.create_ensemble(quants, locs, pdf_constructor_name)
    assert exec_info.type is ValueError


def test_stats_norm():
    scale = np.array([[1, 0.5], [1, 0.5]])
    loc = np.array([[0, 2], [1.0, 1.5]])
    ens_n = qp.stats.norm.create_ensemble(data={"scale": scale, "loc": loc})

    assert ens_n.npdf == 2
