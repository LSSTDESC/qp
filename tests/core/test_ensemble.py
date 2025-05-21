import pytest
import qp
import numpy as np

from tests.helpers import test_data_helper as t_data
from tests.helpers.test_funcs import assert_all_close


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


def test_norm_error_raised(norm_ensemble):
    """Make sure that the appropriate error is raised when a parameterizatoin doesn't have the normalize method."""

    with pytest.raises(
        AttributeError,
        match="This parameterization does not have a normalization function.",
    ):
        norm_ensemble.norm()


def test_x_samples(norm_ensemble):
    """Make sure that x_samples does as expected when the parameterization doesn't have an x_samples method."""

    min = 0
    max = 2
    n = 100
    x_samps = norm_ensemble.x_samples(min=min, max=max, n=n)

    assert_all_close(x_samps, np.linspace(min, max, n))


def test_functions_one_ensemble(hist_ensemble):
    """Make sure that the various Ensemble functions return the appropriate dimensions when given 1 distribution."""

    assert np.ndim(hist_ensemble[0].pdf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].logpdf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].cdf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].logcdf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].ppf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].sf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].logsf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].isf(1.0)) == 0
    assert np.ndim(hist_ensemble[0].mean()) == 0
    assert np.ndim(hist_ensemble[0].median()) == 0
    assert np.ndim(hist_ensemble[0].std()) == 0
    assert np.ndim(hist_ensemble[0].var()) == 0
    assert np.ndim(hist_ensemble[0].moment(1)) == 0
    assert np.ndim(hist_ensemble[0].entropy()) == 0


def test_ancil_dimension(hist_ensemble):
    """Test that the ancillary data table dimensions make sense"""

    ancil = {"ids": np.ones((t_data.NPDF, 3))}
    hist_ensemble.set_ancil(ancil)
    assert_all_close(hist_ensemble.ancil["ids"], ancil["ids"])


def test_writeHdf5Chunk_single_ens(hist_ensemble, tmp_path):
    """Test that the writeHdf5Chunk function works with a chunk of 1 distribution."""

    single_ens = hist_ensemble[0]
    file_path = tmp_path / "test.hdf5"
    groups, fout = single_ens.initializeHdf5Write(file_path, single_ens.npdf)

    single_ens.writeHdf5Chunk(groups, 0, 1)
    single_ens.finalizeHdf5Write(fout)
