import pytest
import qp
import numpy as np


@pytest.mark.parametrize(
    "data_dir",
    [
        ({"test": np.array(["one", "two"]), "test2": np.array([1, 2.5])}),
        ({"test": ["gal1", "gal2"]}),
        ({"test": np.array(["galaxy1"]), "test2": np.array([0])}),
    ],
)
def test_decode_strings(data_dir):
    """Test that decode strings works with multiple input types."""

    # encode strings
    data_dir["test"] = np.strings.encode(data_dir["test"], "utf-8")
    decoded = qp.array.decode_strings(data_dir)

    assert decoded["test"].dtype.kind == "U"
    if "test2" in data_dir.keys():
        assert np.allclose(decoded["test2"], data_dir["test2"])
