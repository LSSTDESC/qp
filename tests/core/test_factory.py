import pytest
import numpy as np
import qp
from tests.helpers.test_data_helper import NPDF


def test_create(hist_test_data, norm_test_data):
    """Make sure that qp create works when the actual class is passed."""

    test_data = hist_test_data["hist"]["ctor_data"]
    ens_h = qp.create(qp.hist, test_data)

    assert ens_h.npdf == NPDF
    assert ens_h.metadata["pdf_name"][0].decode() == "hist"

    test_data_n = norm_test_data["norm"]["ctor_data"]
    ens_n = qp.create(qp.stats.norm, test_data_n)
    assert ens_n.metadata["pdf_name"][0].decode() == "norm"


@pytest.mark.parametrize(
    "filename, expected",
    [("test-quant.h5", True), ("test-quant-nocheckinput.h5", False)],
)
def test_qp_read_files_with_check_input(filename, expected, test_data_dir):
    """Make sure that qp read works with Ensembles with the check_input parameter."""

    filepath = test_data_dir / filename
    ens_q = qp.read(filepath)

    assert ens_q.metadata["ensure_extent"] == expected


def test_qp_read_with_fmt(test_data_dir):
    """Make sure that qp read works when given the fmt argument"""

    filepath = test_data_dir / "test.hdf5"
    ens = qp.read(filepath, fmt="hdf5")

    assert ens.metadata["pdf_name"][0].decode() == "mixmod"


@pytest.mark.parametrize(
    "ancil1,ancil2,ancil3",
    [
        ({"ids": np.arange(0, 1)}, {"ids": np.arange(1, 5)}, {"ids": np.arange(5, 13)}),
        (
            {"ids": np.array([[0]])},
            {"ids": np.array([[1], [2], [3], [4]])},
            {"ids": np.array([[5], [6], [7], [8], [9], [10], [11], [12]])},
        ),
    ],
)
def test_ensemble_concatenation(ancil1, ancil2, ancil3, hist_ensemble):
    """Tests that ensemble concatenation works with 1D and 2D ancillary data, and
    ensembles with 1 or more distributions."""

    # create 3 ensembles with ancillary data

    ens_1 = hist_ensemble[0:1]
    ens_2 = hist_ensemble[1:5]
    ens_3 = hist_ensemble[1:9]

    ens_1.set_ancil(ancil1)
    ens_2.set_ancil(ancil2)
    ens_3.set_ancil(ancil3)
    new_ens = qp.concatenate([ens_1, ens_2, ens_3])

    assert new_ens.npdf == ens_1.npdf + ens_2.npdf + ens_3.npdf
    assert len(new_ens.ancil["ids"]) == new_ens.npdf
