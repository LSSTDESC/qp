"""DEmP photo-z PDFs stored as 30 Gaussian-level quantiles (uint16 encoding).

Parameterization stores float quantile z-positions; the PCHIP CDF is reconstructed
lazily on first evaluation.  Use ``create_ensemble_from_codes`` to load directly from
DEmP FITS files.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike

from .demp_quant_utils import NQ, KNOT_Y, decode_locs, encode_locs, build_knots, pchip_slopes
from ..base import Pdf_rows_gen
from ...core.factory import add_class
from ...core.ensemble import Ensemble
from ...utils.array import reshape_to_pdf_size


class demp_quant_gen(Pdf_rows_gen):
    """DEmP photo-z PDFs stored as 30 Gaussian-level quantiles.

    Quantile levels are L_k = Phi(s_k) where s_k = linspace(-3, 3, 30) and Phi is
    the standard-normal CDF (range ~0.00135 to ~0.99865).  The CDF is reconstructed
    by PCHIP (Fritsch-Carlson monotone cubic) interpolation through extended knots
    (a, 0), (q_1, L_1), ..., (q_30, L_30), (b, 1).

    Parameters
    ----------
    locs : ArrayLike, shape (npdf, 30)
        Quantile z-positions: the redshift where the PDF's CDF reaches each of the
        30 Gaussian quantile levels.
    zlo : float
        Lower outer edge of the redshift range (used to clip the CDF endpoint
        extension).  Typically the QZLO header value from the DEmP FITS file.
    zhi : float
        Upper outer edge of the redshift range.  Typically QZHI.

    Notes
    -----
    To load from a DEmP FITS file that stores uint16 codes, use
    :meth:`create_ensemble_from_codes` rather than constructing directly.

    The PCHIP knots and slopes are built lazily on first CDF/PDF/PPF evaluation
    and cached for the lifetime of the object.
    """

    name = "demp_quant"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        locs: ArrayLike,
        zlo: float,
        zhi: float,
        *args,
        **kwargs,
    ):
        self._zlo = float(zlo)
        self._zhi = float(zhi)
        raw = np.asarray(locs)
        if raw.dtype.kind in ("u", "i") and raw.dtype.itemsize == 2:
            # int16/uint16 codes from file → decode to float64 locs
            self._locs = reshape_to_pdf_size(decode_locs(raw, self._zlo, self._zhi), -1)
        else:
            self._locs = reshape_to_pdf_size(np.asarray(raw, np.float64), -1)

        self._xmin = float(np.min(self._locs))
        self._xmax = float(np.max(self._locs))

        # PCHIP knots/slopes are computed lazily
        self._knot_x = None
        self._knot_d = None

        kwargs["shape"] = self._locs.shape
        super().__init__(*args, **kwargs)

        self._addmetadata("zlo", self._zlo)
        self._addmetadata("zhi", self._zhi)
        # Store as uint16 codes (60 bytes/object) via int16 view for FITS/HDF5
        codes = encode_locs(self._locs, self._zlo, self._zhi)
        self._addobjdata("locs", codes.view(np.int16))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def locs(self) -> np.ndarray:
        """Quantile z-positions, shape (npdf, 30)."""
        return self._locs

    @property
    def zlo(self) -> float:
        """Lower outer edge of the redshift range."""
        return self._zlo

    @property
    def zhi(self) -> float:
        """Upper outer edge of the redshift range."""
        return self._zhi

    # ------------------------------------------------------------------
    # PCHIP construction (lazy)
    # ------------------------------------------------------------------

    def _ensure_knots(self):
        """Build and cache PCHIP knot positions and slopes (once per object)."""
        if self._knot_x is None:
            self._knot_x = build_knots(self._locs, self._zlo, self._zhi)
            self._knot_d = pchip_slopes(self._knot_x)

    # ------------------------------------------------------------------
    # Core PCHIP evaluators
    # ------------------------------------------------------------------

    def _find_segments(self, x_flat, kx):
        """Return segment indices k such that kx[:, k] <= x_flat < kx[:, k+1].

        Parameters
        ----------
        x_flat : ndarray, shape (M,)
            Query z-values (assumed clipped to [kx[:, 0], kx[:, -1]]).
        kx : ndarray, shape (M, 32)
            Per-row knot positions.

        Returns
        -------
        k : ndarray of int, shape (M,)
            Segment index in [0, NQ].
        """
        # Count how many interior knots (indices 1..NQ) are <= x_flat
        return (x_flat[:, None] >= kx[:, 1 : NQ + 1]).sum(axis=1)

    @staticmethod
    def _broadcast_row(x_flat, row_flat):
        """Broadcast row_flat to the same length as x_flat if needed."""
        if row_flat.shape != x_flat.shape:
            row_flat = np.broadcast_to(row_flat, x_flat.shape).copy()
        return row_flat

    def _eval_cdf(self, x_flat, row_flat):
        """Evaluate the PCHIP CDF for given (x, row) pairs.

        Parameters
        ----------
        x_flat : ndarray, shape (M,)
        row_flat : ndarray of int, shape (M,) or broadcastable to (M,)

        Returns
        -------
        cdf : ndarray, shape (M,), clipped to [0, 1]
        """
        self._ensure_knots()
        row_flat = self._broadcast_row(x_flat, row_flat)
        kx = self._knot_x[row_flat]  # (M, 32)
        kd = self._knot_d[row_flat]  # (M, 32)

        x_c = np.clip(x_flat, kx[:, 0], kx[:, -1])
        k = self._find_segments(x_c, kx)  # (M,)
        r = np.arange(len(x_c))

        x0, x1 = kx[r, k], kx[r, k + 1]
        y0, y1 = KNOT_Y[k], KNOT_Y[k + 1]
        d0, d1 = kd[r, k], kd[r, k + 1]

        hh = x1 - x0
        t = (x_c - x0) / hh
        t2, t3 = t * t, t * t * t

        C = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * hh * d0
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * hh * d1
        )
        return np.clip(C, 0.0, 1.0)

    def _eval_pdf(self, x_flat, row_flat):
        """Evaluate the PCHIP PDF (d(CDF)/dx) for given (x, row) pairs.

        Returns zero outside the knot range.
        """
        self._ensure_knots()
        row_flat = self._broadcast_row(x_flat, row_flat)
        kx = self._knot_x[row_flat]  # (M, 32)
        kd = self._knot_d[row_flat]  # (M, 32)

        in_range = (x_flat >= kx[:, 0]) & (x_flat <= kx[:, -1])
        x_c = np.clip(x_flat, kx[:, 0], kx[:, -1])
        k = self._find_segments(x_c, kx)
        r = np.arange(len(x_c))

        x0, x1 = kx[r, k], kx[r, k + 1]
        y0, y1 = KNOT_Y[k], KNOT_Y[k + 1]
        d0, d1 = kd[r, k], kd[r, k + 1]

        hh = x1 - x0
        t = (x_c - x0) / hh
        t2 = t * t

        # Derivative of PCHIP polynomial w.r.t. t, divided by hh = d/dx
        dC_dt = (
            (6.0 * t2 - 6.0 * t) * y0
            + (3.0 * t2 - 4.0 * t + 1.0) * hh * d0
            + (-6.0 * t2 + 6.0 * t) * y1
            + (3.0 * t2 - 2.0 * t) * hh * d1
        )
        pdf = np.maximum(0.0, dC_dt / hh)
        return np.where(in_range, pdf, 0.0)

    def _eval_ppf(self, p_flat, row_flat):
        """Evaluate the PPF (inverse CDF) using Newton / bisection hybrid.

        Each p value is bracketed within its PCHIP segment (where the CDF is
        guaranteed to be monotone).  Newton's method is used when the PDF is
        large enough and the step stays inside the bracket; otherwise the
        bracket midpoint is used (bisection fallback).  50 iterations give
        full double precision even in flat tails where Newton stalls.

        At exactly the 30 stored quantile levels the initial linear estimate
        already lands on the knot, dz = 0, and the bracket check keeps z
        there, so no extra logic is needed for those points.
        """
        self._ensure_knots()

        row_flat = self._broadcast_row(p_flat, row_flat)

        # Find segment k such that KNOT_Y[k] <= p < KNOT_Y[k+1]
        k = np.clip(
            (p_flat[:, None] > KNOT_Y[None, :]).sum(axis=1) - 1, 0, NQ
        )  # (M,) in [0, NQ]

        y0, y1 = KNOT_Y[k], KNOT_Y[k + 1]
        # Initial bracket: the per-row knot x-positions for segment k
        z_lo = self._knot_x[row_flat, k]
        z_hi = self._knot_x[row_flat, k + 1]

        # Initial estimate: linear interpolation within the segment
        dy = y1 - y0
        t = np.where(dy > 0.0, (p_flat - y0) / dy, 0.5)
        z = z_lo + t * (z_hi - z_lo)

        for _ in range(50):
            cdf_z = self._eval_cdf(z, row_flat)
            pdf_z = self._eval_pdf(z, row_flat)
            dz = np.where(pdf_z > 1e-12, (cdf_z - p_flat) / pdf_z, 0.0)
            z_newton = z - dz
            # Tighten bracket around the current point
            above = cdf_z > p_flat
            z_lo = np.where(above, z_lo, z)
            z_hi = np.where(above, z, z_hi)
            # Accept Newton step when pdf is large and step stays in bracket
            # (>= / <= so a zero step at a knot boundary is accepted as-is);
            # fall back to bisection midpoint for flat tails or overshoot.
            use_newton = (pdf_z > 1e-12) & (z_newton >= z_lo) & (z_newton <= z_hi)
            z = np.where(use_newton, z_newton, 0.5 * (z_lo + z_hi))

        return z

    # ------------------------------------------------------------------
    # scipy rv_continuous overrides
    # ------------------------------------------------------------------

    def _pdf(self, x, row):
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        row_flat = np.asarray(row, dtype=np.int64).ravel()
        return self._eval_pdf(x_flat, row_flat).ravel()

    def _cdf(self, x, row):
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        row_flat = np.asarray(row, dtype=np.int64).ravel()
        return self._eval_cdf(x_flat, row_flat).ravel()

    def _ppf(self, p, row):
        p_flat = np.asarray(p, dtype=np.float64).ravel()
        row_flat = np.asarray(row, dtype=np.int64).ravel()
        return self._eval_ppf(p_flat, row_flat).ravel()

    def _updated_ctor_param(self):
        dct = super()._updated_ctor_param()
        dct["locs"] = encode_locs(self._locs, self._zlo, self._zhi).view(np.int16)
        dct["zlo"] = self._zlo
        dct["zhi"] = self._zhi
        return dct

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def x_samples(self) -> np.ndarray:
        """Return representative x values for plotting all PDFs."""
        return np.linspace(self._xmin, self._xmax, 200)

    def to_codes(self) -> np.ndarray:
        """Return the uint16 DEmP encoding of the stored quantile z-positions.

        Returns
        -------
        codes : ndarray, shape (npdf, 30), uint16
        """
        return encode_locs(self._locs, self._zlo, self._zhi)

    # ------------------------------------------------------------------
    # qp class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_allocation_kwds(cls, npdf: int, **kwargs) -> dict:
        """Return kwds for allocating an empty HDF5 file with ``npdf`` entries."""
        return dict(locs=((npdf, NQ), np.int16))

    @classmethod
    def add_mappings(cls) -> None:
        """Register creation methods with the qp factory."""
        cls._add_creation_method(cls.create, None)

    @classmethod
    def create_ensemble(
        cls,
        locs: ArrayLike,
        zlo: float,
        zhi: float,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Create an Ensemble from quantile z-positions.

        Parameters
        ----------
        locs : ArrayLike, shape (npdf, 30)
            Quantile z-positions at the 30 Gaussian CDF levels.
        zlo : float
            Lower outer edge of the redshift range.
        zhi : float
            Upper outer edge of the redshift range.
        ancil : Mapping, optional
            Ancillary data, e.g. ``{'ids': object_id_array}``.

        Returns
        -------
        Ensemble

        Examples
        --------
        >>> import qp, numpy as np
        >>> locs = np.random.uniform(0.5, 1.5, (100, 30))
        >>> locs = np.sort(locs, axis=1)
        >>> ens = qp.demp_quant.create_ensemble(locs, zlo=-0.005, zhi=6.005)
        """
        data = {"locs": locs, "zlo": zlo, "zhi": zhi}
        return Ensemble(cls, data, ancil)

    @classmethod
    def create_ensemble_from_codes(
        cls,
        codes: ArrayLike,
        zlo: float,
        zhi: float,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Create an Ensemble from uint16 DEmP encoded codes.

        This is the primary entry point when loading from a DEmP FITS file.

        Parameters
        ----------
        codes : ArrayLike, shape (npdf, 30), dtype uint16 or int16
            Encoded quantile values from a DEmP FITS file (column ``PDF_Q``).
            int16 (FITS files without TZERO) is accepted and reinterpreted.
        zlo : float
            Lower outer edge of the redshift range (header keyword ``QZLO``;
            use -0.005 for the legacy ``QUANT30G16`` format).
        zhi : float
            Upper outer edge of the redshift range (header keyword ``QZHI``;
            use 6.005 for the legacy ``QUANT30G16`` format).
        ancil : Mapping, optional
            Ancillary data, e.g. ``{'ids': np.asarray(h[1].data['ID'])}``.

        Returns
        -------
        Ensemble

        Examples
        --------
        >>> from astropy.io import fits
        >>> import qp
        >>> with fits.open('PDzQ30.fits') as h:
        ...     zlo = float(h[1].header.get('QZLO', -0.005))
        ...     zhi = float(h[1].header.get('QZHI', 6.005))
        ...     ids = np.asarray(h[1].data['ID'])
        ...     codes = np.asarray(h[1].data['PDF_Q'])
        >>> ens = qp.demp_quant.create_ensemble_from_codes(
        ...     codes, zlo, zhi, ancil={'ids': ids}
        ... )
        """
        locs = decode_locs(codes, zlo, zhi)
        return cls.create_ensemble(locs, zlo, zhi, ancil=ancil)


demp_quant = demp_quant_gen
add_class(demp_quant_gen)
