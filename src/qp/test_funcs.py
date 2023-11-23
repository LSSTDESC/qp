"""This small module implements generic tests for distributions to ensure 
that they have been implemented consistently
"""

import os
import numpy as np

from qp.ensemble import Ensemble
from qp.plotting import plot_native, plot, plot_pdf_samples_on_axes
from qp.test_data import NPDF
from qp.factory import read, read_metadata, convert


def assert_all_close(arr, arr2, **kwds):
    """A slightly more informative version of asserting allclose"""
    test_name = kwds.pop("test_name", "test")
    if not np.allclose(arr, arr2, **kwds):  # pragma: no cover
        raise ValueError(
            "%s %.2e %.2e %s"
            % (test_name, (arr - arr2).min(), (arr - arr2).max(), kwds)
        )


def assert_all_small(arr, **kwds):
    """A slightly more informative version of asserting allclose"""
    test_name = kwds.pop("test_name", "test")
    if not np.allclose(arr, 0, **kwds):  # pragma: no cover
        raise ValueError("%s %.2e %.2e %s" % (test_name, arr.min(), arr.max(), kwds))


def build_ensemble(test_data):
    """Build an ensemble from test data in a class"""
    gen_func = test_data["gen_func"]
    ctor_data = test_data["ctor_data"]
    try:
        ens = Ensemble(gen_func, data=ctor_data)
        ancil = test_data.get("ancil")
        if ancil is not None:
            ens.set_ancil(ancil)
        return ens
    except Exception as exep:  # pragma: no cover
        print("Failed to make %s %s %s" % (gen_func, ctor_data, exep))
        raise ValueError from exep


def pdf_func_tests(pdf, test_data, short=False, check_props=True):
    """Run the test for a practicular class"""

    xpts = test_data["test_xvals"]

    if check_props:
        # if we used the c'tor, make sure the class keeps the data used in the c'tor
        for kv in test_data["ctor_data"].keys():
            test_val = pdf.kwds.get(kv, None)
            if test_val is None:
                if not hasattr(pdf.dist, kv):  # pragma: no cover
                    raise ValueError("%s %s" % (pdf.dist, kv))
                _ = getattr(pdf.dist, kv)

    if hasattr(pdf.dist, "npdf"):
        assert pdf.dist.npdf == pdf.npdf

    assert pdf.shape[-1] == NPDF

    pdfs = pdf.pdf(xpts)
    if pdf.ndim == 1:
        xslice = xpts[5]
        pdfs_slice = pdf.pdf(xslice)
        check_pdf = pdfs[:, 5].flatten() - pdfs_slice.flatten()
        assert_all_small(check_pdf, atol=2e-2, test_name="pdf")

        xslice = np.expand_dims(xpts[np.arange(pdf.npdf)], -1)
        pdfs_slice = pdf.pdf(xslice)
        pdf_check = np.array([pdfs[i, i] for i in range(pdf.npdf)])
        check_pdfs_slice = pdf_check.flatten() - pdfs_slice.flatten()
        assert_all_small(check_pdfs_slice, atol=2e-2, test_name="pdf_slice")

    if short:  # pragma: no cover
        return pdf

    cdfs = pdf.cdf(xpts)
    quants = np.linspace(0.01, 0.99, 50)

    binw = xpts[1:] - xpts[0:-1]
    if pdf.ndim == 1:
        check_cdf = ((pdfs[:, 0:-1] + pdfs[:, 1:]) * binw / 2).cumsum(axis=-1) - cdfs[
            :, 1:
        ]
    else:
        check_cdf = ((pdfs[:, :, 0:-1] + pdfs[:, :, 1:]) * binw / 2).cumsum(
            axis=-1
        ) - cdfs[:, :, 1:]

    assert_all_small(check_cdf, atol=2e-1, test_name="cdf")

    ppfs = pdf.ppf(quants)
    check_ppf = pdf.cdf(ppfs) - quants
    assert_all_small(check_ppf, atol=2e-2, test_name="ppf")

    if pdf.ndim == 1:
        quants_slice = np.expand_dims(quants[np.arange(pdf.npdf)], -1)
        ppfs_slice = pdf.ppf(quants_slice)
        _ = np.array([ppfs[i, i] for i in range(pdf.npdf)])

        check_ppfs_slice = pdf.cdf(ppfs_slice) - quants_slice
        assert_all_small(check_ppfs_slice, atol=2e-2, test_name="ppf_slice")

    sfs = pdf.sf(xpts)
    check_sf = sfs + cdfs
    assert_all_small(check_sf - 1, atol=1e-5, test_name="sf")

    _ = pdf.isf(quants)
    check_isf = pdf.cdf(ppfs) + quants[::-1]
    assert_all_small(check_isf - 1, atol=5e-2, test_name="isf")
    return pdf


def run_pdf_func_tests(test_class, test_data, short=False, check_props=True):
    """Run the test for a practicular class"""

    method = test_data.get("method", None)
    ctor_func = test_class.creation_method(method)
    if ctor_func is None:  # pragma: no cover
        raise KeyError(
            "failed to find creation method %s for class %s" % (method, test_class)
        )
    try:
        pdf = ctor_func(**test_data["ctor_data"])
    except Exception as exep:  # pragma: no cover
        print("Failed to make %s %s %s" % (ctor_func, test_data["ctor_data"], exep))
        raise ValueError from exep

    alloc_kwds = pdf.dist.get_allocation_kwds(pdf.npdf, **test_data["ctor_data"])
    for key, val in alloc_kwds.items():
        assert np.product(val[0]) == np.size(test_data["ctor_data"][key])

    return pdf_func_tests(pdf, test_data, short=short, check_props=check_props)


def persist_func_test(ensemble, test_data):
    """Run loopback persistence tests on an ensemble"""
    # ftypes = ['fits', 'hdf5', 'pq']
    if ensemble.ndim == 1:
        ftypes = ["fits", "hf5", "h5", "pq", "hdf5"]
    else:
        ftypes = ["fits", "hf5"]

    for ftype in ftypes:
        filename = "test_%s.%s" % (ensemble.gen_class.name, ftype)
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
        ensemble.write_to(filename)
        meta = read_metadata(filename)
        ens_r = read(filename)
        meta2 = ens_r.metadata()
        # check that reading metadata and main file get same metadata items
        for k, _v in meta.items():
            # we can't actually do a better check than this because the build_tables
            # method in the ensemble class may add extra metadata items and change their type
            assert k in meta2

        diff = ensemble.pdf(test_data["test_xvals"]) - ens_r.pdf(
            test_data["test_xvals"]
        )
        assert_all_small(diff, atol=1e-5, test_name="persist")
        if ensemble.ancil is not None:
            if ftype in ["pq"]:
                continue
            diff2 = ensemble.ancil["zmode"] - ens_r.ancil["zmode"]
            assert_all_small(diff2, atol=1e-5, test_name="persist_ancil")
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
        try:
            os.unlink(filename.replace(".%s" % ftype, "data.%s" % ftype))
            os.unlink(filename.replace(".%s" % ftype, "meta.%s" % ftype))
        except FileNotFoundError:
            pass


def run_persist_func_tests(test_data):
    """Run the test for a practicular class"""
    ens = build_ensemble(test_data)
    persist_func_test(ens, test_data)


def run_convert_tests(ens_orig, gen_class, test_data, **kwargs):
    """Run the test for a practicular class"""

    xpts = test_data["test_xvals"]

    binw = np.mean(xpts[1:] - xpts[0:-1])
    atol = kwargs.get("atol_diff", 1e-2) / binw
    atol2 = kwargs.get("atol_diff2", 1e-2) / binw

    ens1 = ens_orig.convert_to(gen_class, **test_data["convert_data"])
    diffs = ens_orig.pdf(xpts) - ens1.pdf(xpts)
    assert_all_small(diffs, atol=atol, test_name="convert")

    ens2 = convert(ens_orig, gen_class.name, **test_data["convert_data"])
    diffs = ens_orig.pdf(xpts) - ens2.pdf(xpts)
    assert_all_small(diffs, atol=atol2, test_name="convert2")

    assert ens_orig.shape[:-1] == ens1.shape[:-1]
    assert ens_orig.shape[:-1] == ens2.shape[:-1]
    assert ens_orig.frozen.shape[:-1] == ens2.shape[:-1]
    if hasattr(ens2.dist, "shape"):
        assert ens2.dist.shape[:-1] == ens2.shape[:-1]


def plotting_func_tests(ensemble, do_samples=False):
    """Run the test for a practicular class"""
    pdf = ensemble[0]
    fig, axes = plot_native(pdf, xlim=(-5, 5))
    assert fig is not None
    assert axes is not None
    fig, axes = plot(pdf, axes=axes)
    assert fig is not None
    assert axes is not None
    fig, axes = plot_native(pdf.frozen, xlim=(-5, 5))
    assert fig is not None
    assert axes is not None
    fig, axes = plot(pdf.frozen, xlim=(-5, 5))
    assert fig is not None
    assert axes is not None
    if do_samples:
        samples = pdf.rvs(size=1000)
        plot_pdf_samples_on_axes(axes, pdf, samples)


def run_plotting_func_tests(test_data, do_samples=False):
    """Run the test for a practicular class"""
    ens = build_ensemble(test_data)
    if ens.ndim != 1:
        return
    plotting_func_tests(ens, do_samples=do_samples)
