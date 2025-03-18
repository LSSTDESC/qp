import pytest
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
