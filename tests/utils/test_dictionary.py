import pytest
import qp
import numpy as np


@pytest.mark.parametrize(
    "ancil1,ancil2",
    [
        ({"ids": np.arange(0, 5)}, {"ids": np.arange(5, 13)}),
        (
            {"ids": np.array([[0], [1], [2], [3], [4]])},
            {"ids": np.array([[5], [6], [7], [8], [9], [10], [11], [12]])},
        ),
    ],
)
def test_ensemble_concatenation(ancil1, ancil2, hist_ensemble):
    """Tests that ensemble concatenation works with 1D and 2D ancillary data"""

    # create 2 ensembles with ancillary data

    ens_1 = hist_ensemble[0:5]
    ens_2 = hist_ensemble[1:9]

    ens_1.set_ancil(ancil1)
    ens_2.set_ancil(ancil2)
    new_ens = qp.concatenate([ens_1, ens_2])

    assert new_ens.npdf == ens_1.npdf + ens_2.npdf
    assert len(new_ens.ancil["ids"]) == new_ens.npdf
