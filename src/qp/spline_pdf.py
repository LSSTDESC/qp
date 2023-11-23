"""This module implements a PDT distribution sub-class using splines
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splev, splint, splrep
from scipy.special import errstate  # pylint: disable=no-name-in-module
from scipy.stats import rv_continuous

from qp.conversion_funcs import extract_samples, extract_xy_vals
from qp.factory import add_class
from qp.pdf_gen import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.test_data import SAMPLES, TEST_XVALS, XARRAY, YARRAY
from qp.utils import build_kdes, evaluate_kdes, reshape_to_pdf_size


def normalize_spline(xvals, yvals, limits, **kwargs):
    """
    Normalize a set of 1D interpolators

    Parameters
    ----------
    xvals : array-like
        X-values used for the spline
    yvals : array-like
        Y-values used for the spline
    limits : tuple (2)
        Lower and Upper limits of integration

    Keywords
    --------
    Passed to the `scipy.quad` intergation function

    Returns
    -------
    ynorm: array-like
        Normalized y-vals
    """

    def row_integral(irow):
        def spl(xv):
            return splev(xv, splrep(xvals[irow], yvals[irow]))

        return quad(spl, limits[0], limits[1], **kwargs)[0]

    vv = np.vectorize(row_integral)
    with errstate(all="ignore"):
        integrals = vv(np.arange(xvals.shape[0]))
    return (yvals.T / integrals).T


def build_splines(xvals, yvals):
    """
    Build a set of 1D spline representations

    Parameters
    ----------
    xvals : array-like
        X-values used for the spline
    yvals : array-like
        Y-values used for the spline

    Returns
    -------
    splx : array-like
        Spline knot xvalues
    sply : array-like
        Spline knot yvalues
    spln : array-like
        Spline knot order paramters
    """
    l_x = []
    l_y = []
    l_n = []
    for xrow, yrow in zip(xvals, yvals):
        rep = splrep(xrow, yrow)
        l_x.append(rep[0])
        l_y.append(rep[1])
        l_n.append(rep[2])
    return np.vstack(l_x), np.vstack(l_y), np.vstack(l_n)


class spline_gen(Pdf_rows_gen):
    """Spline based distribution

    Notes
    -----
    This implements PDFs using a set of splines

    The relevant data members are:

    splx:  (npdf, n) spline-knot x-values

    sply:  (npdf, n) spline-knot y-values

    spln:  (npdf) spline-knot order paramters

    The pdf() for the ith pdf will return the result of
    scipy.interpolate.splev(x, splx[i], sply[i], spln[i))

    The cdf() for the ith pdf will return the result of
    scipy.interpolate.splint(x, splx[i], sply[i], spln[i))

    The ppf() will use the default scipy implementation, which
    inverts the cdf() as evaluated on an adaptive grid.
    """

    # pylint: disable=protected-access

    name = "spline"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, *args, **kwargs):
        """
        Create a new distribution using the given histogram

        Keywords
        --------
        splx : array_like
          The x-values of the spline knots
        sply : array_like
          The y-values of the spline knots
        spln : array_like
          The order of the spline knots

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
        kwargs["shape"] = splx.shape[:-1]
        self._splx = reshape_to_pdf_size(splx, -1)
        self._sply = reshape_to_pdf_size(sply, -1)
        self._spln = reshape_to_pdf_size(spln, -1)
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
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation

        Returns
        -------
        splx : array_like
          The x-values of the spline knots
        sply : array_like
          The y-values of the spline knots
        spln : array_like
          The order of the spline knots
        """
        if xvals.shape != yvals.shape:  # pragma: no cover
            raise ValueError(
                "Shape of xvals (%s) != shape of yvals (%s)"
                % (xvals.shape, yvals.shape)
            )
        xmin = np.min(xvals)
        xmax = np.max(xvals)
        yvals = normalize_spline(xvals, yvals, limits=(xmin, xmax), **kwargs)
        return build_splines(xvals, yvals)

    @classmethod
    def create_from_xy_vals(cls, xvals, yvals, **kwargs):
        """
        Create a new distribution using the given x and y values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation

        Returns
        -------
        pdf_obj : `spline_gen`
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
        xvals : array_like
          The x-values used to do the interpolation
        samples : array_like
          The sample values used to build the KDE

        Returns
        -------
        pdf_obj : `spline_gen`
            The requested PDF
        """
        kdes = build_kdes(samples)
        kwargs.pop("yvals", None)
        yvals = evaluate_kdes(xvals, kdes)
        xvals_expand = (np.expand_dims(xvals, -1) * np.ones(samples.shape[0])).T
        return cls.create_from_xy_vals(xvals_expand, yvals, **kwargs)

    @property
    def splx(self):
        """Return x-values of the spline knots"""
        return self._splx

    @property
    def sply(self):
        """Return y-values of the spline knots"""
        return self._sply

    @property
    def spln(self):
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
    def get_allocation_kwds(cls, npdf, **kwargs):
        """
        Return the keywords necessary to create an 'empty' hdf5 file with npdf entries
        for iterative file writeout.  We only need to allocate the objdata columns, as
        the metadata can be written when we finalize the file.

        Parameters
        ----------
        npdf: int
            number of *total* PDFs that will be written out
        kwargs: dict
            dictionary of kwargs needed to create the ensemble
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
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_creation_method(cls.create_from_xy_vals, "xy")
        cls._add_creation_method(cls.create_from_samples, "samples")
        cls._add_extraction_method(extract_xy_vals, "xy")
        cls._add_extraction_method(extract_samples, "samples")

    @classmethod
    def make_test_data(cls):
        """Make data for unit tests"""
        SPLX, SPLY, SPLN = cls.build_normed_splines(XARRAY, YARRAY)
        cls.test_data = dict(
            spline=dict(
                gen_func=spline,
                ctor_data=dict(splx=SPLX, sply=SPLY, spln=SPLN),
                test_xvals=TEST_XVALS[::10],
            ),
            spline_kde=dict(
                gen_func=spline_from_samples,
                ctor_data=dict(samples=SAMPLES, xvals=np.linspace(0, 5, 51)),
                convert_data=dict(xvals=np.linspace(0, 5, 51), method="samples"),
                test_xvals=TEST_XVALS,
                atol_diff2=1.0,
                test_pdf=False,
            ),
            spline_xy=dict(
                gen_func=spline_from_xy,
                ctor_data=dict(xvals=XARRAY, yvals=YARRAY),
                convert_data=dict(xvals=np.linspace(0, 5, 51), method="xy"),
                test_xvals=TEST_XVALS,
                test_pdf=False,
            ),
        )


spline = spline_gen.create
spline_from_xy = spline_gen.create_from_xy_vals
spline_from_samples = spline_gen.create_from_samples

add_class(spline_gen)
