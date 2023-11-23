"""
Unit tests for PDF class
"""
import os
import unittest

import h5py
import numpy as np
import pytest
from mpi4py import MPI

import qp
from qp.test_funcs import build_ensemble


@pytest.mark.skipif(
    not h5py.get_config().mpi, reason="Do not have parallel version of hdf5py"
)
class EnsembleTestCase(unittest.TestCase):
    """Class to test qp.Ensemble functionality"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """

    def tearDown(self):
        "Clean up any mock data files created by the tests."

    @staticmethod
    def _run_ensemble_funcs(ens_type, ens, _xpts):
        """Run the test for a practicular class"""
        comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        zmode = ens.mode(grid=np.linspace(-3, 3, 100))
        ens.set_ancil(dict(zmode=zmode, ones=np.ones(ens.npdf)))
        group, fout = ens.initializeHdf5Write(
            f"testwrite_{ens_type}.hdf5", ens.npdf * mpi_size, comm
        )
        ens.writeHdf5Chunk(group, mpi_rank * ens.npdf, (mpi_rank + 1) * ens.npdf)
        ens.finalizeHdf5Write(fout)

        readens = qp.read(f"testwrite_{ens_type}.hdf5")
        assert sum(readens.ancil["ones"]) == mpi_size * ens.npdf
        assert len(readens.ancil["zmode"]) == mpi_size * ens.npdf
        assert readens.metadata().keys() == ens.metadata().keys()
        assert readens.objdata().keys() == ens.objdata().keys()

        test_grid = np.linspace(-3, 3, 100)
        itr = qp.iterator(f"testwrite_{ens_type}.hdf5", 10, mpi_rank, mpi_size)
        for start, end, ens_i in itr:
            assert np.allclose(readens[start:end].pdf(test_grid), ens_i.pdf(test_grid))

        if mpi_rank == 0:
            os.remove(f"testwrite_{ens_type}.hdf5")

    def test_parallel_norm(self):
        """Run the ensemble tests on an ensemble of scipy.stats.norm distributions"""
        key = "norm"
        cls_test_data = qp.stats.norm_gen.test_data[key]  # pylint: disable=no-member
        ens_norm = build_ensemble(cls_test_data)
        self._run_ensemble_funcs("norm", ens_norm, cls_test_data["test_xvals"])

    def test_parallel_hist(self):
        """Run the ensemble tests on an ensemble of qp.hist distributions"""
        key = "hist"
        qp.hist_gen.make_test_data()
        cls_test_data = qp.hist_gen.test_data[key]
        ens_h = build_ensemble(cls_test_data)
        self._run_ensemble_funcs("hist", ens_h, cls_test_data["test_xvals"])

    def test_parallel_interp(self):
        """Run the ensemble tests on an ensemble of qp.interp distributions"""
        key = "interp"
        qp.interp_gen.make_test_data()
        cls_test_data = qp.interp_gen.test_data[key]
        ens_i = build_ensemble(cls_test_data)
        self._run_ensemble_funcs("interp", ens_i, cls_test_data["test_xvals"])


if __name__ == "__main__":
    unittest.main()
