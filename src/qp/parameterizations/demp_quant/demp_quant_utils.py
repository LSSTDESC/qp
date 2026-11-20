"""Utility functions for the DEmP 30-Gaussian-quantile parameterization.

The DEmP format stores photo-z PDFs as 30 CDF quantile values encoded as uint16.
Quantile levels are L_k = Phi(s_k), s_k = linspace(-3, 3, 30), where Phi is the
standard normal CDF (range ~0.00135 to ~0.99865).

The CDF is reconstructed via PCHIP (Fritsch-Carlson monotone cubic) interpolation
through extended knots (a, 0), (q_1, L_1), ..., (q_30, L_30), (b, 1).
"""

import numpy as np
from scipy.special import ndtr

NQ = 30
LEVELS = ndtr(np.linspace(-3.0, 3.0, NQ))  # (30,) Gaussian CDF levels
KNOT_Y = np.concatenate([[0.0], LEVELS, [1.0]])  # (32,) CDF values at PCHIP knots
SCALE = 65535  # uint16 max value used for encoding


def decode_locs(codes, zlo, zhi):
    """Convert uint16 DEmP codes to float64 quantile z-positions.

    Parameters
    ----------
    codes : array_like, shape (N, 30), dtype uint16 or int16
        Encoded quantile positions from a DEmP FITS file.  FITS may read
        uint16 as int16 when the TZERO keyword is absent; both are accepted.
    zlo, zhi : float
        Redshift range outer edges (from QZLO, QZHI header keywords).

    Returns
    -------
    locs : ndarray, shape (N, 30), float64
        Quantile z-positions.
    """
    u = np.asarray(codes)
    if u.dtype.kind == "i" and u.dtype.itemsize == 2:
        u = u.view(u.dtype.str.replace("i", "u"))
    return zlo + u.astype(np.float64) / SCALE * (zhi - zlo)


def encode_locs(locs, zlo, zhi):
    """Convert float64 quantile z-positions to uint16 DEmP codes.

    Parameters
    ----------
    locs : array_like, shape (N, 30), float64
        Quantile z-positions.
    zlo, zhi : float
        Redshift range outer edges.

    Returns
    -------
    codes : ndarray, shape (N, 30), uint16
    """
    q = np.asarray(locs, np.float64)
    u = np.round((q - zlo) / (zhi - zlo) * SCALE)
    return np.clip(u, 0, SCALE).astype(np.uint16)


def build_knots(locs, zlo, zhi):
    """Build PCHIP CDF knot x-positions from float quantile z-positions.

    Clips the quantile positions to [zlo, zhi], extends the first and last
    quantile segments linearly to z-values where CDF = 0 and CDF = 1
    respectively (clipped to [zlo, zhi]), then enforces strict monotonicity
    by accumulating maxima and adding a small offset.

    Parameters
    ----------
    locs : array_like, shape (N, 30), float64
        Quantile z-positions (decoded from uint16 or provided directly).
    zlo, zhi : float
        Redshift range outer edges.

    Returns
    -------
    knot_x : ndarray, shape (N, 32), float64
        Strictly increasing knot x-positions for each of the N PDFs.
    """
    q = np.clip(np.asarray(locs, np.float64), zlo, zhi)
    L = LEVELS
    # Extrapolate first segment to CDF = 0
    a = np.maximum(
        zlo,
        q[:, 0] - np.maximum(q[:, 1] - q[:, 0], 1e-4) * L[0] / (L[1] - L[0]),
    )
    # Extrapolate last segment to CDF = 1
    b = np.minimum(
        zhi,
        q[:, -1] + np.maximum(q[:, -1] - q[:, -2], 1e-4) * (1.0 - L[-1]) / (L[-1] - L[-2]),
    )
    x = np.concatenate([a[:, None], q, b[:, None]], axis=1)  # (N, 32)
    # Enforce strictly increasing knots (handles equal quantiles from narrow PDFs)
    x = np.maximum.accumulate(x, axis=1) + np.arange(NQ + 2) * 1e-7
    return x


def pchip_slopes(knot_x):
    """Compute PCHIP slopes at the CDF knots (Fritsch-Carlson algorithm).

    The CDF y-values at the knots are always KNOT_Y (identical for all PDFs),
    so only the x-positions differ between rows.

    Parameters
    ----------
    knot_x : ndarray, shape (N, 32), float64
        Strictly increasing knot x-positions from :func:`build_knots`.

    Returns
    -------
    slopes : ndarray, shape (N, 32), float64
        PCHIP slopes (d(CDF)/dz) at each knot.
    """
    h = np.diff(knot_x, axis=1)        # (N, 31) intervals
    dy = np.diff(KNOT_Y)               # (31,) CDF differences — shared across rows
    m = dy[None, :] / h                # (N, 31) chord slopes (all positive)
    d = np.empty_like(knot_x)          # (N, 32) output slopes

    # Interior slopes: Fritsch-Carlson weighted harmonic mean
    w1 = 2.0 * h[:, 1:] + h[:, :-1]   # (N, 30)
    w2 = h[:, 1:] + 2.0 * h[:, :-1]   # (N, 30)
    d[:, 1:-1] = (w1 + w2) / (w1 / m[:, :-1] + w2 / m[:, 1:])

    # Left endpoint slope (Hyman 1983 non-oscillatory condition)
    h0, h1, m0, m1 = h[:, 0], h[:, 1], m[:, 0], m[:, 1]
    de = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    d[:, 0] = np.where(de < 0.0, 0.0, de)

    # Right endpoint slope
    h0, h1, m0, m1 = h[:, -1], h[:, -2], m[:, -1], m[:, -2]
    de = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    d[:, -1] = np.where(de < 0.0, 0.0, de)

    return d
