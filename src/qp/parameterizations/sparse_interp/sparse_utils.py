from __future__ import annotations

from typing import Any

import numpy as np
from scipy import interpolate as sciinterp

from .sparse_rep import build_sparse_representation


def extract_sparse_from_xy(
    in_dist: "Ensemble", **kwargs
) -> dict[str, Any]:  # pragma: no cover
    """Extract sparse representation from an xy interpolated representation

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    xvals : ArrayLike
        Used to override the y-values
    xvals : ArrayLike
        Used to override the x-values
    nvals : int
        Used to override the number of bins

    Returns
    -------
    metadata : dict[str, Any]
        Dictionary with data for sparse representation

    Notes
    -----
    This function will rebin to a grid more suited to the in_dist support by
    removing x-values corrsponding to y=0
    """
    default = in_dist.objdata["yvals"]
    yvals = kwargs.pop("yvals", default)
    default = in_dist.metadata["xvals"][0]
    xvals = kwargs.pop("xvals", default)
    nvals = kwargs.pop("nvals", 300)
    # rebin to a grid more suited to the in_dist support
    xmin = np.min(xvals)
    _, j = np.where(yvals > 0)
    xmax = np.max(xvals[j])
    newx = np.linspace(xmin, xmax, nvals)
    interp = sciinterp.interp1d(xvals, yvals, assume_sorted=True)
    newpdf = interp(newx)
    sparse_indices, metadata, _ = build_sparse_representation(newx, newpdf)
    metadata["xvals"] = newx
    metadata["sparse_indices"] = sparse_indices
    metadata.pop("Ntot")
    return metadata
