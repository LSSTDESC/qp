import pytest
import qp
import numpy as np

from tests.helpers import test_data_helper as t_data


def test_repr(hist_ensemble):
    """Test that the representation method works as expected."""

    assert (
        hist_ensemble.__repr__()
        == f"Ensemble(the_class=hist,shape=({t_data.NPDF}, {t_data.NBIN -1}))"
    )


def test_len(hist_ensemble):
    """Make sure that the .__len__ method returns the number of distributions."""

    assert len(hist_ensemble) == hist_ensemble.npdf


@pytest.mark.parametrize(
    "ancil_data", [(np.array(["gal1", "gal2"])), (["test1", "test2"])]
)
def test_encode_decode_strings(tmp_path, ancil_data):
    """Tests that the encoding and decoding of string ancil data columns for HDF5 files works with
    write_to and read."""

    bins = np.linspace(-2, 2, 11)
    pdfs = np.array(
        [
            [0, 0.5, 1, 0.5, 0.5, 1.25, 1.5, 0.75, 0.5, 0.2],
            [0.05, 0.09, 0.15, 0.2, 0.3, 0.5, 0.25, 0.15, 0.1, 0.025],
        ]
    )
    ancil = {"names": ancil_data}
    ens_h = qp.hist.create_ensemble(bins, pdfs, ancil=ancil)

    # write to file
    file_path = tmp_path / "test-encode.hdf5"
    ens_h.write_to(file_path)

    # read from file
    new_ens = qp.read(file_path)
    assert new_ens.ancil["names"][0] == ancil_data[0]


# TODO: don't need this test?
@pytest.mark.parametrize(
    "loc,scale,npdf",
    [
        (np.array([1.0, 1.5]), np.array([1.0, 1.5]), 2),
        (np.array([[1], [0.5]]), np.array([[1], [0.5]]), 2),
    ],
)
def test_stats_arg_reshape(loc, scale, npdf):
    """Tests that the arg reshape and broadcasting for scipy classes works as expected"""

    ens_chi = qp.stats.chi2.create_ensemble(
        data={"scale": scale, "loc": loc, "df": 0.5}
    )

    assert ens_chi.npdf == npdf


def test_ensemble_objdata_dims(hist_test_data):
    """Ensure that the objdata arrays' dimensionality is reduced when slicing for 1 object."""

    key = "hist"
    ens_h = qp.Ensemble(
        hist_test_data[key]["gen_func"], hist_test_data[key]["ctor_data"]
    )

    assert np.ndim(ens_h[1].objdata["pdfs"]) == 1

    single_ens = ens_h[1]

    assert np.ndim(single_ens[0].objdata["pdfs"]) == 1
    with pytest.raises(IndexError) as exec_info:
        single_ens[1].objdata["pdfs"]
    assert exec_info.type is IndexError

    maxvals = np.max(hist_test_data[key]["ctor_data"]["pdfs"], axis=1)
    ancil = dict(maxvals=maxvals)
    ens_h.set_ancil(ancil)

    single_ens2 = ens_h[2]

    assert np.ndim(single_ens2[0].objdata["pdfs"]) == 1
    assert np.ndim(single_ens2[0].ancil["maxvals"]) == 0

    with pytest.raises(IndexError) as exec_info:
        single_ens2[2].ancil
    assert exec_info.type is IndexError
