import pytest
from pathlib import Path
import numpy as np
import scipy.stats as sps
from tests.helpers import test_data_helper as t_data
import qp
from qp.parameterizations.packed_interp.packing_utils import PackingType


@pytest.fixture
def test_dir() -> Path:
    """Return path to test directory

    Returns
    -------
    Path
        Path to test directory
    """
    return Path(__file__).resolve().parent


@pytest.fixture
def test_data_dir(test_dir) -> Path:
    return test_dir / "test_data"


@pytest.fixture
def hist_test_data():
    """Writes out test data and parameters for the histogram parameterization"""

    return t_data.hist_test_data


@pytest.fixture
def interp_test_data():
    """Writes out test data and parameters for the interp parameterizations"""

    return t_data.interp_test_data


@pytest.fixture
def interp_irreg_test_data():

    return t_data.interp_irregular_test_data


@pytest.fixture
def mixmod_test_data():
    """Writes out test data and parameters for the mixmod parameterization"""

    return t_data.mixmod_test_data


@pytest.fixture
def quant_test_data():
    """Test data and parameters for the quant parameterization"""

    return t_data.quant_test_data


@pytest.fixture
def norm_test_data():
    """Writes out test data and parameters for the norm parameterization"""

    return t_data.norm_test_data


@pytest.fixture
def packed_interp_test_data():
    """Writes out test data and parameters for the packed interp parameterization"""

    return t_data.packed_interp_test_data


@pytest.fixture
def sparse_test_data():
    """Writes out test data and parameters for the sparse parameterization"""

    return t_data.sparse_test_data


@pytest.fixture
def spline_test_data():
    """Writes out test data and parameters for the spline parameterization"""

    return t_data.spline_test_data


## Fixtures to create ensembles
@pytest.fixture
def hist_ensemble(hist_test_data):
    ens_h = qp.hist.create_ensemble(**hist_test_data["hist"]["ctor_data"])
    return ens_h


@pytest.fixture
def norm_ensemble(norm_test_data):
    ens_n = qp.stats.norm.create_ensemble(norm_test_data["norm"]["ctor_data"])
    return ens_n
