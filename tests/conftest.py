import pytest
from pathlib import Path
import numpy as np
import scipy.stats as sps
import qp


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
