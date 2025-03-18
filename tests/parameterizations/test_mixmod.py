import pytest
import qp
import numpy as np


@pytest.fixture
def mixmod_ensemble(mixmod_test_data) -> qp.Ensemble:
    ens = qp.mixmod.create_ensemble(**mixmod_test_data["mixmod"]["ctor_data"])
    return ens


def test_norm(mixmod_ensemble):
    """Test that calling norm raises the appropriate error."""

    with pytest.raises(
        RuntimeError,
        match="The distributions in a mixmod parameterization are already normalized",
    ):
        mixmod_ensemble.norm()


def test_x_samples(mixmod_ensemble):
    """Test that x_samples works as expected."""

    xsamps = mixmod_ensemble.x_samples()

    xmin = np.min(mixmod_ensemble.objdata["means"]) - np.max(
        mixmod_ensemble.objdata["stds"]
    )
    xmax = np.max(mixmod_ensemble.objdata["means"]) + np.max(
        mixmod_ensemble.objdata["stds"]
    )

    assert np.min(xsamps) <= xmin
    assert np.max(xsamps) >= xmax
    assert len(xsamps) >= 50
    assert len(xsamps) <= 10000


@pytest.mark.parametrize(
    ("means_big,stds_small, stds_big"),
    [((-25, 25), (3, 7), (25, 35)), ((-100, 100), (0.05, 0.5), (50, 100))],
)
def test_x_samples_with_more_npts(means_big, stds_small, stds_big):
    """Make sure that the x_smaples method works when npts is used and when npts_max is used."""

    npdf = 5

    mean = np.vstack(
        [
            np.linspace(-2, 2, npdf),
            np.linspace(-10, 10, npdf),
            np.linspace(*means_big, num=npdf),
        ]
    ).T
    std = np.vstack(
        [
            np.linspace(*stds_small, num=npdf),
            np.linspace(5, 10, npdf),
            np.linspace(*stds_big, num=npdf),
        ]
    ).T
    weights = np.vstack(
        [0.7 * np.ones((npdf)), 0.2 * np.ones((npdf)), 0.1 * np.ones((npdf))]
    ).T

    ens_m = qp.mixmod.create_ensemble(means=mean, stds=std, weights=weights)

    xmin = np.min(ens_m.objdata["means"]) - np.max(ens_m.objdata["stds"])
    xmax = np.max(ens_m.objdata["means"]) + np.max(ens_m.objdata["stds"])

    dx = np.min(ens_m.objdata["stds"]) / 2.0
    npts = (xmax - xmin) // dx

    xsamps = ens_m.x_samples()

    # make sure limits of xsamples makes sense
    assert xmin == np.min(xsamps)
    assert xmax == np.max(xsamps)

    # make sure the number of points makes sense
    if npts > 10000:
        assert len(xsamps) == 10000
    else:
        assert len(xsamps) == npts


@pytest.mark.parametrize(
    "means, stds, weights, match_string",
    [
        (
            np.array([0, 0.5, np.nan]),
            np.array([0.25, 0.5, 1.0]),
            np.array([1, 1, 1]),
            "The given means contain non-finite values for the following distributions",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([0.25, 0.5, np.inf]),
            np.array([1, 0.5, 1]),
            "here are non-finite values in the stds for the following distributions",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([0.25, 0.5, 1.0]),
            np.array([1, 1, np.nan]),
            "There are non-finite values in the weights for the following distributions",
        ),
        (
            np.array([0, 0.5, 1]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            "The following distributions have all stds",
        ),
    ],
)
def test_warnings(means, stds, weights, match_string):
    """Test that warnings when building a mixmod Ensemble work."""

    with pytest.warns(RuntimeWarning, match=match_string):
        ens = qp.mixmod.create_ensemble(means=means, stds=stds, weights=weights)


@pytest.mark.parametrize(
    "means, stds, weights, match_string",
    [
        (
            np.array([0, 0.5]),
            np.array([0.25, 0.5, 1.0]),
            np.array([1, 1, 1]),
            "must have the same shape",
        ),
        (
            np.array([0, 0.5, 1.0]),
            np.array([0.25, 0.5, 1.0]),
            np.array([-0.2, 0, 0.5]),
            "Invalid input: All weights need to be larger than or equal to 0",
        ),
        (
            np.array([0, 0.5, 1.0]),
            np.array([0.25, -0.5, 1.0]),
            np.array([1, 0.5, 0.5]),
            "Invalid input: All standard deviations",
        ),
    ],
)
def test_invalid_input(means, stds, weights, match_string):
    """Tests that the appropriate errors are raised when invalid input is given to build a mixmod Ensemble."""

    with pytest.raises(ValueError, match=match_string):
        ens = qp.mixmod.create_ensemble(means=means, stds=stds, weights=weights)
