"""
Unit tests for PDF class
"""
import copy
import numpy as np
import unittest
import qp
from qp import test_data
import os
import sys

from qp.test_funcs import assert_all_small, assert_all_close, build_ensemble
from qp.plotting import init_matplotlib
from mpi4py import MPI

class EnsembleTestCase(unittest.TestCase):
    """ Class to test qp.Ensemble functionality """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """

    def tearDown(self):
        "Clean up any mock data files created by the tests."

    @staticmethod
    def _run_ensemble_funcs(ens, xpts):
        """Run the test for a practicular class"""
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        zmode = ens.mode(grid=np.linspace(-3,3,100))
        ens.set_ancil(dict(zmode=zmode,ones=np.ones(ens.npdf)))
        group, fout = ens.initializeHdf5Write("testwrite.hdf5", ens.npdf*mpi_size, comm)
        ens.writeHdf5Chunk(group, mpi_rank*ens.npdf, (mpi_rank+1)*ens.npdf)
        ens.finalizeHdf5Write(fout)
        readens = qp.read("testwrite.hdf5")
        assert sum(readens.ancil['ones']) == mpi_size*ens.npdf
        assert len(readens.ancil['zmode']) == mpi_size*ens.npdf
        assert readens.metadata().keys() == ens.metadata().keys()
        assert readens.objdata().keys() == ens.objdata().keys()
        if mpi_rank == 0:
            os.remove("testwrite.hdf5")


    def test_norm(self):
        """ Run the ensemble tests on an ensemble of scipy.stats.norm distributions """
        key = 'norm'
        cls_test_data = qp.stats.norm_gen.test_data[key]  #pylint: disable=no-member
        ens_norm = build_ensemble(cls_test_data)
        self._run_ensemble_funcs(ens_norm, cls_test_data['test_xvals'])


    def test_hist(self):
        """ Run the ensemble tests on an ensemble of qp.hist distributions """
        key = 'hist'
        qp.hist_gen.make_test_data()
        cls_test_data = qp.hist_gen.test_data[key]
        ens_h = build_ensemble(cls_test_data)
        self._run_ensemble_funcs(ens_h, cls_test_data['test_xvals'])

    def test_interp(self):
        """ Run the ensemble tests on an ensemble of qp.interp distributions """
        key = 'interp'
        qp.interp_gen.make_test_data()
        cls_test_data = qp.interp_gen.test_data[key]
        ens_i = build_ensemble(cls_test_data)
        self._run_ensemble_funcs(ens_i, cls_test_data['test_xvals'])


if __name__ == '__main__':
    unittest.main()
