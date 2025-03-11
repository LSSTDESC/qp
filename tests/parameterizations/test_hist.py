import pytest
import numpy as np
import qp


@pytest.fixture
def hist_ensemble(hist_test_data):
    ens_h = qp.hist.create_ensemble(**hist_test_data["hist"]["ctor_data"])
    return ens_h


def test_norm(hist_test_data):
    """Test that the norm method works for a histogram ensemble"""

    norm = False
    ancil = {"ancil": np.linspace(0, 11, 11)}
    ens_h = qp.hist.create_ensemble(
        **hist_test_data["hist"]["ctor_data"], norm=norm, ancil=ancil
    )

    assert ens_h.npdf == 11

    ens_h.norm()
    # TODO: test here that integrate over the range returns close to 1
    assert (
        ens_h.objdata["pdfs"][0, 1] != hist_test_data["hist"]["ctor_data"]["pdfs"][0, 1]
    )
    assert ens_h.ancil == ancil


def test_xsamples(hist_ensemble):

    x_samps = hist_ensemble.x_samples()
    assert len(hist_ensemble.metadata["bins"]) - 1 == len(x_samps)
