"""This module implements a PDT distribution sub-class using a Gaussian mixture model"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.stats import rv_continuous
from scipy import integrate as sciint
from scipy import interpolate as sciinterp
from typing import Mapping, Optional
from numpy.typing import ArrayLike

from . import sparse_rep
from ...core.factory import add_class
from ...core.ensemble import Ensemble
from ..interp.interp import interp_gen
from .sparse_utils import extract_sparse_from_xy


class sparse_gen(interp_gen):
    """Sparse based distribution. The final behavior is similar to interp_gen, but the constructor
    takes a sparse representation to build the interpolator.
    Attempt to inherit from interp_gen : this is failing

    Notes
    -----
    This implements a qp interface to the original code SparsePz from M. Carrasco-Kind.

    """

    # pylint: disable=protected-access

    name = "sparse"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self, xvals, mu, sig, dims, sparse_indices, *args, **kwargs
    ):  # pylint: disable=too-many-arguments
        self.sparse_indices = sparse_indices
        self._xvals = xvals
        self.mu = mu
        self.sig = sig
        self.dims = dims
        cut = kwargs.pop("cut", 1.0e-5)
        # recreate the basis array from the metadata
        sparse_meta = dict(xvals=xvals, mu=mu, sig=sig, dims=dims)
        A = sparse_rep.create_basis(sparse_meta, cut=cut)
        # decode the sparse indices into basis indices and weights
        basis_indices, weights = sparse_rep.decode_sparse_indices(sparse_indices)
        # retrieve the weighted array of basis functions for each object
        pdf_y = A[:, basis_indices] * weights
        # normalize and sum the weighted pdfs
        x = sparse_meta["xvals"]
        y = pdf_y.sum(axis=-1)
        norms = sciint.trapezoid(y.T, x)
        y /= norms
        kwargs.setdefault("xvals", x)
        kwargs.setdefault("yvals", y.T)
        super().__init__(*args, **kwargs)

        self._clearobjdata()
        self._addmetadata("xvals", self._xvals)
        self._addmetadata("mu", self.mu)
        self._addmetadata("sig", self.sig)
        self._addmetadata("dims", self.dims)
        self._addobjdata("sparse_indices", self.sparse_indices)

    def _updated_ctor_param(self):
        """
        Add the two constructor's arguments for the Factory
        """
        dct = super()._updated_ctor_param()
        dct["sparse_indices"] = self.sparse_indices
        dct["xvals"] = self._xvals
        dct["mu"] = self.mu
        dct["sig"] = self.sig
        dct["dims"] = self.dims
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf, **kwargs
    ) -> dict[str, tuple[tuple[int, int], str]]:
        if "dims" not in kwargs:
            raise ValueError("required argument dims not in kwargs")  # pragma: no cover
        nsp = np.array(kwargs["dims"]).flatten()[4]
        return dict(sparse_indices=((npdf, nsp), "i8"))

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_sparse_from_xy, None)

    @classmethod
    def create_ensemble(
        self, sparse_indices, xvals, mu, sig, dims, ancil: Optional[Mapping] = None
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as interpolations, constructed from a sparse representation.


        Parameters
        ----------
        sparse_indices:
        xvals :
        mu :
        sig :
        dims :
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have the same length as the number of distributions, by default None

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.
        """
        data = {
            "sparse_indices": sparse_indices,
            "xvals": xvals,
            "mu": mu,
            "sig": sig,
            "dims": dims,
        }
        return Ensemble(self, data, ancil)


sparse = sparse_gen

add_class(sparse_gen)
