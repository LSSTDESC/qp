"""
Unit tests for PDF class
"""

import os
import unittest

import h5py
import numpy as np
import pytest
from mpi4py import MPI
import tempfile
from pathlib import Path

import qp
from tests.helpers.test_funcs import build_ensemble
from tests.helpers import test_data_helper as t_data


@pytest.mark.skipif(
    not h5py.get_config().mpi, reason="Do not have parallel version of hdf5py"
)
class EnsembleTestCase(unittest.TestCase):
    """Class to test qp.Ensemble functionality"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.tmpdir = tempfile.TemporaryDirectory()

        self.norm_test_data = dict(
            norm=dict(
                gen_func=qp.stats.norm,
                ctor_data=dict(loc=t_data.LOC, scale=t_data.SCALE),
                test_xvals=t_data.TEST_XVALS,
                do_samples=True,
                ancil=dict(zmode=t_data.LOC),
            ),
            norm_shifted=dict(
                gen_func=qp.stats.norm,
                filekey="norm_shifted",
                ctor_data=dict(loc=t_data.LOC, scale=t_data.SCALE),
                test_xvals=t_data.TEST_XVALS,
            ),
            norm_multi_d=dict(
                gen_func=qp.stats.norm,
                filekey="norm_multi_d",
                ctor_data=dict(
                    loc=np.array([t_data.LOC, t_data.LOC]),
                    scale=np.array([t_data.SCALE, t_data.SCALE]),
                ),
                test_xvals=t_data.TEST_XVALS,
                do_samples=True,
                npdf=22,
            ),
        )
        # make hist test data
        self.hist_test_data = dict(
            hist=dict(
                gen_func=qp.hist,
                ctor_data=dict(bins=t_data.XBINS, pdfs=t_data.HIST_DATA),
                convert_data=dict(bins=t_data.XBINS),
                atol_diff=1e-1,
                atol_diff2=1e-1,
                test_xvals=t_data.TEST_XVALS,
            ),
            hist_samples=dict(
                gen_func=qp.hist,
                filekey="hist_samples",
                ctor_data=dict(bins=t_data.XBINS, pdfs=t_data.HIST_DATA),
                convert_data=dict(
                    bins=t_data.XBINS, method="samples", size=t_data.NSAMPLES
                ),
                atol_diff=1e-1,
                atol_diff2=1e-1,
                test_xvals=t_data.TEST_XVALS,
                do_samples=True,
            ),
        )
        self.interp_test_data = dict(
            interp=dict(
                gen_func=qp.interp,
                ctor_data=dict(xvals=t_data.XBINS, yvals=t_data.YARRAY),
                convert_data=dict(xvals=t_data.XBINS),
                test_xvals=t_data.TEST_XVALS,
            )
        )

    def tearDown(self):
        "Clean up any mock data files created by the tests."

        self.tmpdir.cleanup()

    @staticmethod
    def _run_ensemble_funcs(ens_type, ens, _xpts, tmpdir):
        """Run the test for a practicular class"""
        comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        zmode = ens.mode(grid=np.linspace(-3, 3, 100))
        ens.set_ancil(dict(zmode=zmode, ones=np.ones(ens.npdf)))

        filename = Path(tmpdir) / f"testwrite_{ens_type}.hdf5"
        group, fout = ens.initializeHdf5Write(filename, ens.npdf * mpi_size, comm)
        ens.writeHdf5Chunk(group, mpi_rank * ens.npdf, (mpi_rank + 1) * ens.npdf)
        ens.finalizeHdf5Write(fout)

        readens = qp.read(filename)
        assert sum(readens.ancil["ones"]) == mpi_size * ens.npdf
        assert len(readens.ancil["zmode"]) == mpi_size * ens.npdf
        assert readens.metadata.keys() == ens.metadata.keys()
        assert readens.objdata.keys() == ens.objdata.keys()

        test_grid = np.linspace(-3, 3, 100)
        itr = qp.iterator(filename, 10, mpi_rank, mpi_size)
        for start, end, ens_i in itr:
            assert np.allclose(readens[start:end].pdf(test_grid), ens_i.pdf(test_grid))

        # if mpi_rank == 0:
        #     os.remove(f"testwrite_{ens_type}.hdf5")

    def test_parallel_norm(self):
        """Run the ensemble tests on an ensemble of scipy.stats.norm distributions"""
        key = "norm"
        cls_test_data = self.norm_test_data[key]  # pylint: disable=no-member
        ens_norm = build_ensemble(cls_test_data)
        self._run_ensemble_funcs(
            "norm", ens_norm, cls_test_data["test_xvals"], self.tmpdir.name
        )

    def test_parallel_hist(self):
        """Run the ensemble tests on an ensemble of qp.hist distributions"""
        key = "hist"
        # qp.hist_gen.make_test_data()
        cls_test_data = self.hist_test_data[key]
        ens_h = build_ensemble(cls_test_data)
        self._run_ensemble_funcs(
            "hist", ens_h, cls_test_data["test_xvals"], self.tmpdir.name
        )

    def test_parallel_interp(self):
        """Run the ensemble tests on an ensemble of qp.interp distributions"""
        key = "interp"
        # qp.interp_gen.make_test_data()
        cls_test_data = self.interp_test_data[key]
        ens_i = build_ensemble(cls_test_data)
        self._run_ensemble_funcs(
            "interp", ens_i, cls_test_data["test_xvals"], self.tmpdir.name
        )


if __name__ == "__main__":
    unittest.main()
