"""Tests for the DEmP 30-Gaussian-quantile parameterization."""

import tempfile
import os
from pathlib import Path

import numpy as np
import pytest
import qp

SAMPLE_FITS    = Path(__file__).resolve().parents[1] / "test_data" / "pdzq30_sample.fits"
REFERENCE_FITS = Path(__file__).resolve().parents[1] / "test_data" / "pdzq30_reference.fits"

from qp.parameterizations.demp_quant.demp_quant_utils import (
    NQ,
    KNOT_Y,
    LEVELS,
    decode_locs,
    encode_locs,
    build_knots,
    pchip_slopes,
)

NPDF = 11
ZLO, ZHI = -0.005, 6.005
rng = np.random.default_rng(1234)
LOCS = np.sort(rng.uniform(0.1, 3.0, (NPDF, NQ)), axis=1)


@pytest.fixture
def ensemble():
    return qp.demp_quant.create_ensemble(LOCS, zlo=ZLO, zhi=ZHI)


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------


def test_levels_shape():
    assert LEVELS.shape == (NQ,)
    assert KNOT_Y.shape == (NQ + 2,)
    assert KNOT_Y[0] == 0.0
    assert KNOT_Y[-1] == 1.0
    assert np.all(np.diff(KNOT_Y) > 0)


def test_encode_decode_roundtrip():
    codes = encode_locs(LOCS, ZLO, ZHI)
    assert codes.dtype == np.uint16
    assert codes.shape == (NPDF, NQ)
    locs_rt = decode_locs(codes, ZLO, ZHI)
    # Quantization step = (ZHI - ZLO) / 65535; expect at most half a step
    max_err = (ZHI - ZLO) / 65535 * 0.5 + 1e-9
    assert np.max(np.abs(locs_rt - LOCS)) < max_err + 1e-9


def test_int16_decode():
    """FITS files may deliver uint16 data as int16 when TZERO is absent."""
    codes_u16 = encode_locs(LOCS, ZLO, ZHI)
    codes_i16 = codes_u16.view(np.int16)
    locs_from_i16 = decode_locs(codes_i16, ZLO, ZHI)
    locs_from_u16 = decode_locs(codes_u16, ZLO, ZHI)
    np.testing.assert_array_equal(locs_from_i16, locs_from_u16)


def test_build_knots_monotone():
    knot_x = build_knots(LOCS, ZLO, ZHI)
    assert knot_x.shape == (NPDF, NQ + 2)
    assert np.all(np.diff(knot_x, axis=1) > 0), "knot_x must be strictly increasing"


def test_pchip_slopes_shape():
    knot_x = build_knots(LOCS, ZLO, ZHI)
    slopes = pchip_slopes(knot_x)
    assert slopes.shape == (NPDF, NQ + 2)
    assert np.all(slopes >= 0), "PCHIP slopes should be non-negative for monotone CDF"


# ------------------------------------------------------------------
# Ensemble construction
# ------------------------------------------------------------------


def test_create_ensemble(ensemble):
    assert ensemble.npdf == NPDF
    assert "zlo" in ensemble.metadata
    assert "zhi" in ensemble.metadata
    assert ensemble.dist.zlo == ZLO
    assert ensemble.dist.zhi == ZHI
    assert ensemble.dist.locs.shape == (NPDF, NQ)


def test_create_ensemble_from_codes():
    codes = encode_locs(LOCS, ZLO, ZHI)
    ens = qp.demp_quant.create_ensemble_from_codes(codes, ZLO, ZHI)
    assert ens.npdf == NPDF
    # Locs should match the decoded values
    np.testing.assert_allclose(ens.dist.locs, decode_locs(codes, ZLO, ZHI))


def test_to_codes(ensemble):
    codes = ensemble.dist.to_codes()
    assert codes.dtype == np.uint16
    assert codes.shape == (NPDF, NQ)


# ------------------------------------------------------------------
# CDF / PDF / PPF
# ------------------------------------------------------------------


def test_cdf_monotone(ensemble):
    z = np.linspace(0.1, 2.9, 50)
    for i in range(NPDF):
        cdf = ensemble[i].cdf(z)
        assert np.all(np.diff(cdf) >= 0), f"CDF not monotone for PDF {i}"


def test_cdf_bounds(ensemble):
    z_lo = np.array([ZLO - 1.0])
    z_hi = np.array([ZHI + 1.0])
    for i in range(NPDF):
        assert ensemble[i].cdf(z_lo) >= 0.0
        assert ensemble[i].cdf(z_hi) <= 1.0


def test_pdf_nonnegative(ensemble):
    z = np.linspace(0.0, 3.0, 100)
    for i in range(NPDF):
        pdf = ensemble[i].pdf(z)
        assert np.all(pdf >= 0), f"PDF negative for PDF {i}"


def test_pdf_outside_support_is_zero(ensemble):
    # Outside knot range the PDF should be 0
    z_far = np.array([ZLO - 2.0, ZHI + 2.0])
    for i in range(NPDF):
        assert np.all(ensemble[i].pdf(z_far) == 0.0)


@pytest.mark.skipif(not SAMPLE_FITS.exists(), reason="sample FITS file not present")
def test_ppf_flat_tail_accuracy():
    """ppf is accurate in flat tails where pdf ≈ 0 and Newton stalls.

    ~10 % of objects in the sample have pdf = 0 at the extension knot.
    For those objects, check that cdf(ppf(p)) ≈ p at extreme quantiles.
    """
    ens = qp.read(str(SAMPLE_FITS))
    ens.dist._ensure_knots()
    # Objects with pdf=0 at left extension knot
    bad_idx = np.where(ens.dist._knot_d[:, 0] == 0.0)[0]
    assert len(bad_idx) > 0, "fixture has no flat-tail objects — update test"

    p_extreme = np.array([1e-4, 1e-6, 1e-8, 1e-10])
    for obj in bad_idx[:20]:
        z = ens[int(obj)].ppf(p_extreme).ravel()
        cdf_rt = ens[int(obj)].cdf(z).ravel()
        np.testing.assert_allclose(
            cdf_rt, p_extreme, atol=1e-12,
            err_msg=f"PPF inaccurate in flat tail for object {obj}",
        )


def test_ppf_monotone(ensemble):
    p = np.linspace(0.01, 0.99, 50)
    for i in range(NPDF):
        ppf = ensemble[i].ppf(p)
        assert np.all(np.diff(ppf) >= 0), f"PPF not monotone for PDF {i}"


def test_cdf_ppf_roundtrip(ensemble):
    # Only test z values strictly within each PDF's PCHIP knot range
    # (outside the range CDF == 0 or 1, so PPF returns ±inf by convention)
    ensemble.dist._ensure_knots()
    for i in range(NPDF):
        kx_i = ensemble.dist._knot_x[i]
        z_test = np.linspace(kx_i[1], kx_i[-2], 15)  # skip endpoint knots
        z_rt = ensemble[i].ppf(ensemble[i].cdf(z_test))
        np.testing.assert_allclose(z_rt, z_test, atol=1e-4)


def test_ppf_cdf_roundtrip(ensemble):
    # Avoid p=0 and p=1 (which map to ±inf in z)
    p_test = np.linspace(0.05, 0.95, 20)
    for i in range(NPDF):
        p_rt = ensemble[i].cdf(ensemble[i].ppf(p_test))
        np.testing.assert_allclose(p_rt, p_test, atol=1e-4)


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------


def test_pdf_integrates_to_one(ensemble):
    z_edges = np.linspace(-0.005, 6.005, 601)
    _, h = ensemble[0].histogramize(z_edges)
    np.testing.assert_allclose(h.sum(), 1.0, atol=1e-3)


# ------------------------------------------------------------------
# HDF5 round-trip
# ------------------------------------------------------------------


def test_hdf5_roundtrip(ensemble):
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        fname = f.name
    try:
        ensemble.write_to(fname)
        ens_rt = qp.read(fname)
        assert ens_rt.npdf == NPDF
        assert ens_rt.dist.zlo == ZLO
        assert ens_rt.dist.zhi == ZHI
        # locs are stored as uint16 codes; first round-trip introduces at most
        # half a quantisation step of error, subsequent round-trips are exact.
        quant_tol = (ZHI - ZLO) / 65535 * 0.5 + 1e-9
        np.testing.assert_allclose(ens_rt.dist.locs, ensemble.dist.locs, atol=quant_tol)
        z = np.array([0.5, 1.0, 2.0])
        for i in range(NPDF):
            np.testing.assert_allclose(ens_rt[i].cdf(z), ensemble[i].cdf(z), atol=1e-3)
    finally:
        os.unlink(fname)


# ------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------


def test_convert_to_hist(ensemble):
    bins = np.linspace(0.0, 3.0, 61)
    ens_h = ensemble.convert_to(qp.hist, bins=bins)
    assert ens_h.npdf == NPDF


def test_convert_to_quant(ensemble):
    quants = np.linspace(0.01, 0.99, 20)
    ens_q = ensemble.convert_to(qp.quant, quants=quants)
    assert ens_q.npdf == NPDF


# ------------------------------------------------------------------
# Sample FITS file (tests/test_data/pdzq30_sample.fits)
# qp-format file: readable with qp.read(); written from the original
# DEmP FITS via create_ensemble_from_codes + write_to.
# ------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_FITS.exists(), reason="sample FITS file not present")
def test_read_sample_fits():
    """qp.read() loads the sample file and returns a valid demp_quant Ensemble."""
    ens = qp.read(str(SAMPLE_FITS))

    assert isinstance(ens.dist, qp.demp_quant_gen)
    assert ens.npdf > 0
    assert ens.dist.locs.shape == (ens.npdf, NQ)
    assert ens.ancil is not None and "ids" in ens.ancil

    zlo, zhi = ens.dist.zlo, ens.dist.zhi
    assert zlo < zhi

    # CDF is monotone for a handful of objects spread across the ensemble
    indices = np.linspace(0, ens.npdf - 1, 5, dtype=int)
    z = np.linspace(zlo + 0.01, zhi - 0.01, 50)
    for i in indices:
        cdf = ens[i].cdf(z)
        assert np.all(np.diff(cdf) >= 0), f"CDF not monotone for object {i}"

    # PDF integrates to ~1 over the full stored z range
    z_edges = np.linspace(zlo, zhi, 801)
    _, h_bins = ens[0].histogramize(z_edges)
    np.testing.assert_allclose(h_bins.sum(), 1.0, atol=1e-3)


@pytest.mark.skipif(not SAMPLE_FITS.exists(), reason="sample FITS file not present")
def test_sample_fits_roundtrip(tmp_path):
    """Write and re-read the sample ensemble; data should be identical."""
    ens = qp.read(str(SAMPLE_FITS))

    out = tmp_path / "roundtrip.fits"
    ens.write_to(str(out))
    ens2 = qp.read(str(out))

    assert ens2.npdf == ens.npdf
    assert ens2.dist.zlo == ens.dist.zlo
    assert ens2.dist.zhi == ens.dist.zhi
    np.testing.assert_array_equal(ens2.dist.locs, ens.dist.locs)
    np.testing.assert_array_equal(ens2.ancil["ids"], ens.ancil["ids"])


# ------------------------------------------------------------------
# Reference comparison (pdzq30_reference.fits)
# Verifies qp histogramize matches the demp_quantiles.decode output
# (the original DEmP reference implementation) for the same codes.
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not SAMPLE_FITS.exists() or not REFERENCE_FITS.exists(),
    reason="sample or reference FITS file not present",
)
def test_demp_matches_reference():
    """qp CDF-diff agrees with demp_quantiles reference PDFs to < 1e-6 per bin.

    For each reference object: evaluate qp CDF at the 802 bin edges
    (bin_center ± 0.005), take differences, renormalise to sum 1, then
    compare to the reference row.  Empirical max difference is ~2e-8.
    """
    from astropy.io import fits

    ens = qp.read(str(SAMPLE_FITS))
    id_to_idx = {int(v): i for i, v in enumerate(ens.ancil["ids"])}

    with fits.open(str(REFERENCE_FITS)) as h:
        ref_ids     = np.asarray(h["REFERENCE"].data["ID"])
        ref_pdfs    = np.asarray(h["REFERENCE"].data["PDF"], dtype=np.float64)
        bin_centers = np.asarray(h["BINS"].data["BINS"])

    dz = bin_centers[1] - bin_centers[0]
    bin_edges = np.concatenate([[bin_centers[0] - dz / 2], bin_centers + dz / 2])

    for ref_id, ref_pdf in zip(ref_ids, ref_pdfs):
        idx = id_to_idx[int(ref_id)]
        cdf_vals = ens[idx].cdf(bin_edges).ravel()      # (802,)
        probs    = np.clip(np.diff(cdf_vals), 0, None)  # (801,)
        probs   /= probs.sum()
        np.testing.assert_allclose(
            probs, ref_pdf, atol=1e-6,
            err_msg=f"ID {ref_id}: qp CDF-diff differs from reference",
        )
