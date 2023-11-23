"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""
import os
import sys
import numpy as np
from scipy.stats import rv_continuous
from scipy import integrate as sciint
from scipy import interpolate as sciinterp
from qp import sparse_rep
from qp.factory import add_class
from qp.interp_pdf import interp_gen
from qp.conversion_funcs import extract_sparse_from_xy
from qp.test_data import TEST_XVALS, NPDF


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

    def __init__(self, xvals, mu, sig, dims, sparse_indices, *args, **kwargs):  # pylint: disable=too-many-arguments
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
        norms = sciint.trapz(y.T, x)
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
    def get_allocation_kwds(cls, npdf, **kwargs):
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

    @staticmethod
    def build_test_data():
        """build a test case out of real pdfs"""
        qproot = sys.modules["qp"].__path__[0]
        filein = os.path.join(qproot, "./data/CFHTLens_sample.P.npy")
        # FORMAT FILE, EACH ROW IS THE PDF FOR EACH GALAXY, LAST ROW IS THE REDSHIFT POSITION
        P = np.load(filein)
        z = P[-1]
        P = P[:NPDF]
        P = P / sciint.trapz(P, z).reshape(-1, 1)
        minz = np.min(z)
        nz = 301
        _, j = np.where(P > 0)
        maxz = np.max(z[j + 1])
        newz = np.linspace(minz, maxz, nz)
        interp = sciinterp.interp1d(z, P, assume_sorted=True)
        newpdf = interp(newz)
        newpdf = newpdf / sciint.trapz(newpdf, newz).reshape(-1, 1)
        sparse_idx, meta, _ = sparse_rep.build_sparse_representation(
            newz, newpdf, verbose=False
        )
        return sparse_idx, meta

    @classmethod
    def make_test_data(cls):
        SPARSE_IDX, META = cls.build_test_data()

        cls.test_data = dict(
            sparse=dict(
                gen_func=sparse,
                ctor_data=dict(
                    xvals=META["xvals"],
                    mu=META["mu"],
                    sig=META["sig"],
                    dims=META["dims"],
                    sparse_indices=SPARSE_IDX,
                ),
                test_xvals=TEST_XVALS,
            ),
        )


sparse = sparse_gen.create

add_class(sparse_gen)
