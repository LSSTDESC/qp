"""
Unit tests for PDF class
"""

import copy
import os
import unittest
from pathlib import Path
import tempfile
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats as sps
import qp
from qp import test_data
from qp.plotting import init_matplotlib
from qp.parameterizations.packed_interp.packing_utils import PackingType
from tests.helpers.test_funcs import (
    assert_all_close,
    assert_all_small,
    build_ensemble,
)
from tests.helpers import test_data_helper as t_data


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
                ctor_data=dict(loc=t_data.LOC, scale=t_data.SCALE),
                test_xvals=t_data.TEST_XVALS,
            ),
            norm_multi_d=dict(
                gen_func=qp.stats.norm,
                ctor_data=dict(
                    loc=np.array([t_data.LOC, t_data.LOC]),
                    scale=np.array([t_data.SCALE, t_data.SCALE]),
                ),
                test_xvals=t_data.TEST_XVALS,
                do_samples=True,
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

        ypacked_lin, ymax_lin, ypacked_log, ymax_log = t_data.calc_ypacked()

        self.packed_interp_test_data = dict(
            lin_packed_interp=dict(
                gen_func=qp.packed_interp,
                ctor_data=dict(
                    packing_type=PackingType.linear_from_rowmax,
                    xvals=t_data.XBINS,
                    ypacked=ypacked_lin,
                    ymax=ymax_lin,
                ),
                convert_data=dict(
                    xvals=t_data.XBINS,
                    packing_type=PackingType.linear_from_rowmax,
                ),
                test_xvals=t_data.TEST_XVALS,
            ),
            log_packed_interp=dict(
                gen_func=qp.packed_interp,
                ctor_data=dict(
                    packing_type=PackingType.log_from_rowmax,
                    xvals=t_data.XBINS,
                    ypacked=ypacked_log,
                    ymax=ymax_log,
                    log_floor=-3.0,
                ),
                convert_data=dict(
                    xvals=t_data.XBINS,
                    packing_type=PackingType.log_from_rowmax,
                    log_floor=-3.0,
                ),
                test_xvals=t_data.TEST_XVALS,
            ),
        )

    def tearDown(self):
        "Clean up any mock data files created by the tests."

        self.tmpdir.cleanup()

    @staticmethod
    def _run_ensemble_funcs(ens: qp.Ensemble, xpts, tmpdir):
        """Run the test for a particular class"""

        pdfs = ens.pdf(xpts)
        cdfs = ens.cdf(xpts)
        logpdfs = ens.logpdf(xpts)
        logcdfs = ens.logcdf(xpts)

        if hasattr(ens.gen_obj, "npdf"):
            assert ens.npdf == ens.gen_obj.npdf

        with np.errstate(all="ignore"):
            assert np.allclose(np.log(pdfs), logpdfs, atol=1e-9)
            assert np.allclose(np.log(cdfs), logcdfs, atol=1e-9)

        binw = xpts[1:] - xpts[0:-1]
        check_cdf = ((pdfs[:, 0:-1] + pdfs[:, 1:]) * binw / 2).cumsum(axis=1) - cdfs[
            :, 1:
        ]
        assert_all_small(check_cdf, atol=5e-2, test_name="cdf")

        hist = ens.histogramize(xpts)[1]
        hist_check = ens.frozen.histogramize(xpts)[1]
        assert_all_small(hist - hist_check, atol=1e-5, test_name="hist")

        ppfs = ens.ppf(t_data.QUANTS)
        check_ppf = ens.cdf(ppfs) - t_data.QUANTS
        assert_all_small(check_ppf, atol=2e-2, test_name="ppf")

        sfs = ens.sf(xpts)
        check_sf = sfs + cdfs
        assert_all_small(check_sf - 1, atol=2e-2, test_name="sf")

        _ = ens.isf(t_data.QUANTS)
        check_isf = ens.cdf(ppfs) + t_data.QUANTS[::-1]
        assert_all_small(check_isf - 1, atol=2e-2, test_name="isf")

        samples = ens.rvs(size=1000)
        assert samples.shape[0] == ens.frozen.npdf
        assert samples.shape[1] == 1000

        median = ens.median()
        mean = ens.mean()
        var = ens.var()
        std = ens.std()
        entropy = ens.entropy()

        _ = ens.stats()
        modes = ens.mode(xpts)

        assert median.size == ens.npdf
        assert mean.size == ens.npdf
        assert np.std(mean) > 1e-8
        assert var.size == ens.npdf
        assert std.size == ens.npdf
        assert entropy.size == ens.npdf
        assert modes.size == ens.npdf

        integral = ens.integrate(limits=(ens.gen_obj.a, ens.gen_obj.a))
        interval = ens.interval(0.05)

        assert integral.size == ens.npdf
        assert interval[0].size == ens.npdf

        for N in range(3):
            moment_partial = ens.moment_partial(N, limits=(t_data.XMIN, t_data.XMAX))
            calc_moment = qp.metrics.calculate_moment(
                ens, N, limits=(t_data.XMIN, t_data.XMAX)
            )
            assert_all_close(
                moment_partial,
                calc_moment,
                rtol=5e-2,
                test_name="moment_partial_%i" % N,
            )

            sps_moment = ens.moment(N)
            assert sps_moment.size == ens.npdf
            # assert_all_close(sps_moment.flatten(), moment_partial.flatten(),
            #    rtol=5e-2, test_name="moment_%i" % N)
            # pmf = ens.pmf(N)
            # logpmf = ens.logpmf(N)

        init_matplotlib()
        axes = ens.plot(xlim=(xpts[0], xpts[-1]))
        ens.plot_native(axes=axes)

        red_ens = ens[np.arange(5)]
        red_pdf = red_ens.pdf(xpts)

        check_red = red_pdf - pdfs[0:5]
        assert_all_small(check_red, atol=1e-5, test_name="red")

        if hasattr(ens.gen_obj, "npdf"):  # skip scipy norm
            commList = [None]
            try:
                import mpi4py.MPI  # pylint: disable=import-outside-toplevel

                commList.append(
                    mpi4py.MPI.COMM_WORLD
                )  # pylint: disable=c-extension-no-member
            except ImportError:
                pass
            for comm in commList:
                filepath = Path(tmpdir) / "testwrite.hdf5"
                try:
                    group, fout = ens.initializeHdf5Write(filepath, ens.npdf, comm)
                except TypeError:
                    continue
                ens.writeHdf5Chunk(group, 0, ens.npdf)
                ens.finalizeHdf5Write(fout)
                readens = qp.read(filepath)
                assert readens.metadata.keys() == ens.metadata.keys()
                assert readens.objdata.keys() == ens.objdata.keys()
                # os.remove("testwrite.hdf5")

    @staticmethod
    def _run_merge_tests(ens: qp.Ensemble, xpts):
        npdf = ens.npdf
        pdf_orig = ens.pdf(xpts)

        ens_cat = qp.concatenate([ens, ens])
        ens.append(ens)

        pdf_cat = ens_cat.pdf(xpts)

        modes = np.array([xpts[idx] for idx in np.squeeze(np.argmax(pdf_cat, axis=1))])

        ens_cat.set_ancil({"mode": modes})
        pdf_app = ens.pdf(xpts)

        mask = np.concatenate([np.ones((npdf), "bool"), np.zeros((npdf), "bool")])
        ens_check = ens_cat[mask]
        pdf_check = ens_check.pdf(xpts)

        assert_all_close(pdf_cat, pdf_app, atol=5e-8, test_name="merge_1")
        assert_all_close(pdf_orig, pdf_check, atol=5e-8, test_name="merge_2")
        assert_all_close(
            ens_cat.ancil["mode"][mask], modes[mask], atol=5e-8, test_name="mode"
        )

    def test_norm(self):
        """Run the ensemble tests on an ensemble of scipy.stats.norm distributions"""
        key = "norm"
        # cls_test_data = qp.stats.norm_gen.test_data[key]  # pylint: disable=no-member
        cls_test_data = self.norm_test_data[key]
        ens_norm = build_ensemble(cls_test_data)
        assert hasattr(ens_norm, "gen_func")
        assert isinstance(
            ens_norm.gen_obj, qp.stats.norm_gen  # pylint: disable=no-member
        )
        assert "loc" in ens_norm.frozen.kwds
        self._run_ensemble_funcs(ens_norm, cls_test_data["test_xvals"], self.tmpdir)
        self._run_merge_tests(ens_norm, cls_test_data["test_xvals"])

    def test_hist(self):
        """Run the ensemble tests on an ensemble of qp.hist distributions"""
        key = "hist"

        # qp.hist_gen.make_test_data()
        # cls_test_data = qp.hist_gen.test_data[key]
        cls_test_data = self.hist_test_data[key]
        ens_h = build_ensemble(cls_test_data)
        assert isinstance(ens_h.gen_obj, qp.hist_gen)
        self._run_ensemble_funcs(ens_h, cls_test_data["test_xvals"], self.tmpdir)
        self._run_merge_tests(ens_h, cls_test_data["test_xvals"])

        pdfs_mod = copy.copy(ens_h.dist.pdfs)
        pdfs_mod[:, 7] = 0.5 * pdfs_mod[:, 7]
        ens_h.update_objdata(dict(pdfs=pdfs_mod))

    def test_interp(self):
        """Run the ensemble tests on an ensemble of qp.interp distributions"""
        key = "interp"
        # qp.interp_gen.make_test_data()
        # cls_test_data = qp.interp_gen.test_data[key]
        cls_test_data = self.interp_test_data[key]
        ens_i = build_ensemble(cls_test_data)
        assert isinstance(ens_i.gen_obj, qp.interp_gen)
        self._run_ensemble_funcs(ens_i, cls_test_data["test_xvals"], self.tmpdir)

    def test_packed_interp(self):
        """Run the ensemble tests on an ensemble of qp.packed_interp distributions"""
        key = "lin_packed_interp"

        # qp.packed_interp_gen.make_test_data()
        # cls_test_data = qp.packed_interp_gen.test_data[key]
        cls_test_data = self.packed_interp_test_data[key]
        ens_i = build_ensemble(cls_test_data)
        assert isinstance(ens_i.gen_obj, qp.packed_interp_gen)
        self._run_ensemble_funcs(ens_i, cls_test_data["test_xvals"], self.tmpdir)
        assert np.isfinite(ens_i.dist.yvals).all()

    def test_iterator(self):
        """Test the iterated read"""
        # QP_DIR = os.path.abspath(os.path.dirname(qp.__file__))
        TEST_DIR = Path(__file__).resolve().parent
        data_file = os.path.join(TEST_DIR.parent, "test_data", "test.hdf5")
        ens = qp.read(data_file)
        data_length = qp.data_length(data_file)
        assert data_length == ens.npdf
        itr = qp.iterator(data_file, 10)
        test_grid = np.linspace(0.0, 1.0, 11)
        for start, end, ens_i in itr:
            check_vals = ens[start:end].pdf(test_grid)
            test_vals = ens_i.pdf(test_grid)
            assert np.allclose(check_vals, test_vals)

    def test_mixmod_with_negative_weights(self):
        """Verify that an exception is raised when setting up a mixture model with negative weights"""
        means = np.array([0.5, 1.1, 2.9])
        sigmas = np.array([0.15, 0.13, 0.14])
        weights = np.array([1, 0.5, -0.25])
        with self.assertRaises(ValueError):
            _ = qp.mixmod(weights=weights, means=means, stds=sigmas)

    def test_dictionary_output(self):
        """Test that writing and reading a dictionary of ensembles works as expected."""
        key = "hist"
        # qp.hist_gen.make_test_data()
        # cls_test_data = qp.hist_gen.test_data[key]
        cls_test_data = self.hist_test_data[key]
        ens_h = build_ensemble(cls_test_data)

        key = "interp"
        # qp.interp_gen.make_test_data()
        # cls_test_data = qp.interp_gen.test_data[key]
        cls_test_data = self.interp_test_data[key]
        ens_i = build_ensemble(cls_test_data)

        output_dict = {
            "hist": ens_h,
            "interp": ens_i,
        }

        filepath = Path(self.tmpdir) / "test_dict.hdf5"
        qp.write_dict(filepath, output_dict)

        input_dict = qp.read_dict(filepath)

        assert input_dict.keys() == output_dict.keys()

        XVALS = np.linspace(0, 3, 100)
        for ens_type in ["hist", "interp"]:
            assert_array_equal(
                input_dict[ens_type].metadata["pdf_name"],
                output_dict[ens_type].metadata["pdf_name"],
            )

            assert_array_almost_equal(
                input_dict[ens_type].pdf(XVALS), output_dict[ens_type].pdf(XVALS)
            )


if __name__ == "__main__":
    unittest.main()
