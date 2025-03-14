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


def test_xsamples(mixmod_ensemble):
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
