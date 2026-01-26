"""This module implements a PDT distribution sub-class using splines"""

from __future__ import annotations
import numpy as np

from scipy.interpolate import splev, splint, splrep, interp1d
from scipy.special import errstate  # pylint: disable=no-name-in-module
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike

from .spline_utils import (
    extract_samples,
    spline_extract_xy_vals,
    build_kdes,
    evaluate_kdes,
    normalize_spline,
    build_splines,
)
from ...core.factory import add_class
from ...core.ensemble import Ensemble
from ..base import Pdf_rows_gen
from ...plotting import get_axes_and_xlims, plot_pdf_on_axes
from ...utils.array import reshape_to_pdf_size


class spline_gen(Pdf_rows_gen):
    """Spline based distribution

    Notes
    -----
    This implements PDFs using a set of splines

    The relevant data members are:

    - `splx`:  (npdf, n) spline-knot x-values
    - `sply`:  (npdf, n) spline-knot y-values
    - `spln`:  (npdf) spline-knot order parameters

    The pdf() for the ith pdf will return the result of
    `scipy.interpolate.splev(x, splx[i], sply[i], spln[i))`

    The cdf() for the ith pdf will return the result of
    `scipy.interpolate.splint(x, splx[i], sply[i], spln[i))`

    The ppf() will use the default scipy implementation, which
    inverts the cdf() as evaluated on an adaptive grid.
    """

    # pylint: disable=protected-access

    name = "spline"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Create a new distribution using the given histogram

        Parameters
        --------
        splx : ArrayLike
          The x-values of the spline knots
        sply : ArrayLike
          The y-values of the spline knots
        spln : ArrayLike, optional
          The order of the spline knots, by default None

        Notes
        -----
        Either (splx, sply and spln) must be provided.
        """
        splx = kwargs.pop("splx", None)
        sply = kwargs.pop("sply", None)
        spln = kwargs.pop("spln", None)

        if splx is None:  # pragma: no cover
            raise ValueError("splx must be provided")
        if splx.shape != sply.shape:  # pragma: no cover
            raise ValueError(
                "Shape of xvals (%s) != shape of yvals (%s)" % (splx.shape, sply.shape)
            )
        # kwargs['a'] = self.a = np.min(splx)
        # kwargs['b'] = self.b = np.max(splx)
        self._xmin = np.min(splx)
        self._xmax = np.max(splx)
        # kwargs["shape"] = splx.shape[:-1]
        self._splx = reshape_to_pdf_size(splx, -1)
        self._sply = reshape_to_pdf_size(sply, -1)
        self._spln = reshape_to_pdf_size(spln, -1)

        kwargs["shape"] = self._splx.shape
        super().__init__(*args, **kwargs)
        self._addobjdata("splx", self._splx)
        self._addobjdata("sply", self._sply)
        self._addobjdata("spln", self._spln)

    @staticmethod
    def build_normed_splines(xvals, yvals, **kwargs):
        """
        Build a set of normalized splines using the x and y values

        Parameters
        ----------
        xvals : ArrayLike
          The x-values used to do the interpolation
        yvals : ArrayLike
          The y-values used to do the interpolation

        Returns
        -------
        splx : ArrayLike
          The x-values of the spline knots
        sply : ArrayLike
          The y-values of the spline knots
        spln : ArrayLike
          The order of the spline knots
        """
        if xvals.shape != yvals.shape:  # pragma: no cover
            raise ValueError(
                "Shape of xvals (%s) != shape of yvals (%s)"
                % (xvals.shape, yvals.shape)
            )
        xmin = np.min(xvals)
        xmax = np.max(xvals)

        # make sure xvals and yvals are 2d
        if np.ndim(xvals) == 1:
            xvals = np.expand_dims(xvals, axis=0)
        if np.ndim(yvals) == 1:
            yvals = np.expand_dims(yvals, axis=0)

        yvals = normalize_spline(xvals, yvals, limits=(xmin, xmax), **kwargs)
        return build_splines(xvals, yvals)

    @classmethod
    def create_from_xy_vals(cls, xvals, yvals, **kwargs):
        """
        Create a new distribution using the given x and y values

        Parameters
        ----------
        xvals : ArrayLike
          The x-values used to do the interpolation
        yvals : ArrayLike
          The y-values used to do the interpolation

        Returns
        -------
        pdf_obj : spline_gen
            The requested PDF
        """
        splx, sply, spln = spline_gen.build_normed_splines(xvals, yvals, **kwargs)
        gen_obj = cls(splx=splx, sply=sply, spln=spln)
        return gen_obj(**kwargs)

    @classmethod
    def create_from_samples(cls, xvals, samples, **kwargs):
        """
        Create a new distribution using the given x and y values

        Parameters
        ----------
        xvals : ArrayLike
          The x-values used to do the interpolation
        samples : ArrayLike
          The sample values used to build the KDE

        Returns
        -------
        pdf_obj : spline_gen
            The requested PDF
        """
        kdes = build_kdes(samples)
        kwargs.pop("yvals", None)
        yvals = evaluate_kdes(xvals, kdes)
        xvals_expand = (np.expand_dims(xvals, -1) * np.ones(samples.shape[0])).T
        return cls.create_from_xy_vals(xvals_expand, yvals, **kwargs)

    @property
    def splx(self) -> np.ndarray:
        """Return x-values of the spline knots"""
        return self._splx

    @property
    def sply(self) -> np.ndarray:
        """Return y-values of the spline knots"""
        return self._sply

    @property
    def spln(self) -> np.ndarray:
        """Return order of the spline knots"""
        return self._spln

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        def pdf_row(xv, irow):
            return splev(
                xv, (self._splx[irow], self._sply[irow], self._spln[irow].item())
            )

        with errstate(all="ignore"):
            vv = np.vectorize(pdf_row)
        return vv(x, row).ravel()

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        def cdf_row(xv, irow):
            return splint(
                self._xmin,
                xv,
                (self._splx[irow], self._sply[irow], self._spln[irow].item()),
            )

        with errstate(all="ignore"):
            vv = np.vectorize(cdf_row)
        return vv(x, row).ravel()

    def ppf(self, quants):
        # FIXME: remove this function once the issue with spline ppf is fixed
        raise NotImplementedError(
            "This function is buggy and currently not working properly, and will be restored once it's been fixed."
        )

    def _ppf(self, quants, row):
        # pylint: disable=arguments-differ

        # get the cdfs on a grid
        n_pts = 1001
        grid = np.linspace(self._xmin, self._xmax, n_pts)
        unique_rows = np.unique(row)
        cdf_vals = self._cdf(np.expand_dims(grid, -1), unique_rows).reshape(
            len(unique_rows), n_pts
        )

        def ppf_row(quantsv, irow):
            cdf_row = cdf_vals[irow]
            # Filter out the bits where it fluctuations down
            arg_sorted = np.argsort(cdf_row)
            sorted_vals = cdf_row[arg_sorted]
            sorted_grid = grid[arg_sorted]
            mask = np.zeros((len(sorted_grid)), dtype=bool)
            mask[1:] = sorted_grid[1:] > sorted_grid[0:-1]
            mask[1:] &= sorted_vals[1:] > sorted_vals[0:-1]
            sorted_masked_vals = sorted_vals[mask]
            sorted_masked_grid = sorted_grid[mask]
            sorted_masked_vals /= sorted_masked_vals[-1]
            # Build an interpolater, but reverse x and y to get the inverse function
            interp = interp1d(
                np.squeeze(sorted_vals[mask]),
                sorted_grid[mask],
                bounds_error=False,
                fill_value=(sorted_grid[0], sorted_grid[-1]),
            )
            return interp(quantsv)

        with errstate(all="ignore"):
            vv = np.vectorize(ppf_row)
        ret_vals = vv(quants, row).ravel()
        return ret_vals

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct["splx"] = self._splx
        dct["sply"] = self._sply
        dct["spln"] = self._spln
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf: int, **kwargs
    ) -> dict[str, tuple[tuple[int, int], str]]:
        """
        Return the keywords necessary to create an 'empty' hdf5 file with npdf entries
        for iterative file writeout.  We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.

        Parameters
        ----------
        npdf : int
            number of *total* PDFs that will be written out
        kwargs
            dictionary of kwargs needed to create the ensemble

        Returns
        -------
        dict[str, tuple[tuple[int, int], str]]
        """
        if "splx" not in kwargs:  # pragma: no cover
            raise ValueError("required argument splx not included in kwargs")

        shape = np.shape(kwargs["splx"])
        return dict(splx=(shape, "f4"), sply=(shape, "f4"), spln=((shape[0], 1), "i4"))

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a spline this shows the spline knots
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        xvals = pdf.dist.splx[pdf.kwds["row"]]
        return plot_pdf_on_axes(axes, pdf, xvals, **kw)

    @classmethod
    def add_mappings(cls) -> None:
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_creation_method(cls.create_from_xy_vals, "xy")
        cls._add_creation_method(cls.create_from_samples, "samples")
        cls._add_extraction_method(spline_extract_xy_vals, "xy")
        cls._add_extraction_method(extract_samples, "samples")

    @classmethod
    def create_ensemble(
        self,
        splx: ArrayLike,
        sply: ArrayLike,
        spln: Optional[ArrayLike] = None,
        ancil: Optional[Mapping] = None,
        method: Optional[str] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as via a set of splines.


        Parameters
        ----------
        splx : ArrayLike
          The x-values of the spline knots
        sply : ArrayLike
          The y-values of the spline knots
        spln : ArrayLike, optional
          The order of the spline knots, by default None
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have the same length as the number of
            distributions, by default None
        method : Optional[str], optional
            The string of the creation method to use, by default None.

        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.


        """
        data = {"splx": splx, "sply": sply, "spln": spln}
        return Ensemble(self, data, ancil, method)


spline = spline_gen
spline_from_xy = spline_gen.create_from_xy_vals
spline_from_samples = spline_gen.create_from_samples

add_class(spline_gen)
