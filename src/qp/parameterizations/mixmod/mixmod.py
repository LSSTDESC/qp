"""This module implements a PDT distribution sub-class using a Gaussian mixture model"""

from __future__ import annotations
import numpy as np
from scipy import stats as sps
from scipy.stats import rv_continuous
from typing import Mapping, Optional
from numpy.typing import ArrayLike
import warnings

from .mixmod_utils import extract_mixmod_fit_samples
from ...core.factory import add_class
from ..base import Pdf_rows_gen
from ...utils.array import (
    get_eval_case,
    reshape_to_pdf_size,
)
from ...utils.interpolation import interpolate_multi_x_y
from ...core.ensemble import Ensemble


class mixmod_gen(Pdf_rows_gen):
    """Parameterizes distributions using a Gaussian Mixture model.

    There are `ncomp` Gaussians in the model, and `npdf` distributions contained
    in the object.

    Parameters
    ----------
    means : ArrayLike
        The means of the Gaussians, with shape (npdf, ncomp)
    stds : ArrayLike
        The standard deviations of the Gaussians, with shape (npdf, ncomp)
    weights : ArrayLike
        The weights to attach to the Gaussians, with shape (npdf, ncomp).
        Weights should sum up to one. If not, the weights are interpreted
        as relative weights.
    warn : bool, optional
        If True, raises warnings if input is not finite. If False, no warnings
        are raised. By default True.


    Notes
    -----

    All distributions must have the same number of Gaussian components, `ncomp`.
    Use 0 as a fill value instead of `Nan`, which will result in errors in the
    PDF construction.


    Converting to this parameterization:

    This table contains the available methods to convert to this parameterization,
    their required arguments, and their method keys. If the key is `None`, this is
    the default conversion method.

    +------------------------------+--------------------------------------------+------------+
    | Function                     | Arguments                                  | Method key |
    +------------------------------+--------------------------------------------+------------+
    |`.extract_mixmod_fit_samples` | ncomps=3, nsamples=1000, random_state=None | None       |
    +------------------------------+--------------------------------------------+------------+

    Implementation Notes:

    The `pdf()` and `cdf()` are exact, and are computed as a weighted sum of
    the `pdf()` and `cdf()` of the component Gaussians.

    The `ppf()` is computed by computing the `cdf()` values on a fixed
    grid and interpolating the inverse function using `scipy.interp1d` with the
    default interpolation method (linear). `ppf(0)` returns negative infinity and
    `ppf(1)` returns positive infinity.


    """

    # pylint: disable=protected-access

    name = "mixmod"
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(
        self,
        means: ArrayLike,
        stds: ArrayLike,
        weights: ArrayLike,
        warn: bool = True,
        *args,
        **kwargs,
    ):
        """
        Create a new distribution or distributions using a Gaussian Mixture Model.
        There are ncomp Gaussians in the model, and npdf distributions contained
        in the object.

        Parameters
        ----------
        means : ArrayLike
            The means of the Gaussians, with shape (npdf, ncomp)
        stds : ArrayLike
            The standard deviations of the Gaussians, with shape (npdf, ncomp)
        weights : ArrayLike
            The weights to attach to the Gaussians, with shape (npdf, ncomp).
            Weights should sum up to one. If not, the weights are interpreted
            as relative weights.
        warn : bool, optional
            If True, raises warnings if input is not finite. If False, no warnings
            are raised. By default True.
        """
        self._scipy_version_warning()
        self._means = reshape_to_pdf_size(np.asarray(means), -1)
        self._stds = reshape_to_pdf_size(np.asarray(stds), -1)
        self._weights = reshape_to_pdf_size(np.asarray(weights), -1)
        kwargs["shape"] = self._means.shape  # means.shape
        self._ncomps = means.shape[-1]

        # validate input
        if (
            self._means.shape != self._stds.shape
            or self._stds.shape != self._weights.shape
        ):
            raise ValueError(
                f"Invalid input: means {np.shape(means)}, stds {np.shape(stds)}, and weights {np.shape(weights)} must have the same shape."
            )
        if np.any(self._weights < 0):
            raise ValueError(
                "Invalid input: All weights need to be larger than or equal to 0"
            )
        if np.any(self._stds < 0):
            raise ValueError(
                "Invalid input: All standard deviations (stds) must be greater than or equal to 0."
            )

        # raise warnings if input data is not finite
        self._warn = warn
        if self._warn:
            self._input_warnings()

        super().__init__(*args, **kwargs)

        self._weights = self._weights / self._weights.sum(axis=1)[:, None]
        self._addobjdata("weights", self._weights)
        self._addobjdata("stds", self._stds)
        self._addobjdata("means", self._means)

    def _scipy_version_warning(self):
        import scipy  # pylint: disable=import-outside-toplevel

        scipy_version = scipy.__version__
        vtuple = scipy_version.split(".")
        if int(vtuple[0]) > 1 or int(vtuple[1]) > 7:
            return
        raise DeprecationWarning(
            f"Mixmod_gen will not work correctly with scipy version < 1.8.0, you have {scipy_version}"
        )  # pragma: no cover

    def _input_warnings(self):
        """Raise warnings if input is not finite or any distribution has all stds <= 0"""
        if not np.all(np.isfinite(self._means)):
            indices = np.where(np.isfinite(self._means) != True)
            warnings.warn(
                f"The given means contain non-finite values for the following distributions: {indices}",
                RuntimeWarning,
            )
        if not np.all(np.isfinite(self._stds)):
            indices = np.where(np.isfinite(self._stds) != True)
            warnings.warn(
                f"There are non-finite values in the stds for the following distributions: {indices}",
                RuntimeWarning,
            )
        if not np.all(np.isfinite(self._weights)):
            indices = np.where(np.isfinite(self._weights) != True)
            warnings.warn(
                f"There are non-finite values in the weights for the following distributions: {indices}",
                RuntimeWarning,
            )
        if np.any(np.all(self._stds <= 0, axis=1)):
            indices = np.where(np.all(self._stds <= 0, axis=1))
            warnings.warn(
                f"The following distributions have all stds <= 0: {indices}",
                RuntimeWarning,
            )

    def normalize(self):
        raise RuntimeError(
            "The distributions in a mixmod parameterization are already normalized"
        )

    @property
    def weights(self) -> np.ndarray[float]:
        """Return weights to attach to the Gaussians"""
        return self._weights

    @property
    def means(self) -> np.ndarray[float]:
        """Return means of the Gaussians"""
        return self._means

    @property
    def stds(self) -> np.ndarray[float]:
        """Return standard deviations of the Gaussians"""
        return self._stds

    def x_samples(self) -> np.ndarray[float]:
        """Return a set of x values that can be used to plot all the PDFs."""

        # make sure the number of points is reasonable
        npts_min = 50
        npts_max = 10000

        # calculate bounds and dx
        dx = np.min(self._stds) / 2.0
        xmin = np.min(self._means) - np.max(self._stds)
        xmax = np.max(self._means) + np.max(self._stds)
        npts = (xmax - xmin) // dx
        if npts < npts_min:
            return np.linspace(xmin, xmax, npts_min)
        elif npts >= npts_min and npts <= npts_max:
            return np.linspace(xmin, xmax, int(npts))
        else:  # npts > npts_max
            return np.linspace(xmin, xmax, npts_max)

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        if np.ndim(x) > 1:  # pragma: no cover
            x = np.expand_dims(x, -2)
        return (
            self.weights[row].swapaxes(-2, -1)
            * sps.norm(
                loc=self._means[row].swapaxes(-2, -1),
                scale=self._stds[row].swapaxes(-2, -1),
            ).pdf(x)
        ).sum(axis=0)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if np.ndim(x) > 1:  # pragma: no cover
            x = np.expand_dims(x, -2)
        return (
            self.weights[row].swapaxes(-2, -1)
            * sps.norm(
                loc=self._means[row].swapaxes(-2, -1),
                scale=self._stds[row].swapaxes(-2, -1),
            ).cdf(x)
        ).sum(axis=0)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        min_val = np.min(self._means - 6 * self._stds)
        max_val = np.max(self._means + 6 * self._stds)
        grid = np.linspace(min_val, max_val, 201)
        case_idx, _, rr = get_eval_case(x, row)
        if case_idx == 1:
            cdf_vals = self.cdf(grid, rr)
        elif case_idx == 3:
            cdf_vals = self.cdf(grid, np.expand_dims(rr, -1))
        else:  # pragma: no cover
            raise ValueError(
                f"Oops, we can't handle this kind of input to mixmod._ppf {case_idx}"
            )
        return interpolate_multi_x_y(
            x, row, cdf_vals, grid, bounds_error=False, fill_value=(min_val, max_val)
        ).ravel()

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super()._updated_ctor_param()
        dct["means"] = self._means
        dct["stds"] = self._stds
        dct["weights"] = self._weights
        dct["warn"] = self._warn
        return dct

    @classmethod
    def get_allocation_kwds(
        cls, npdf, **kwargs
    ) -> dict[str, tuple[tuple[int, int], str]]:
        """Return the kwds necessary to create an `empty` HDF5 file with ``npdf`` entries
        for iterative write. We only need to allocate the data columns, as
        the metadata will be written when we finalize the file.

        The number of data columns is calculated based on the length or shape of the
        metadata, ``n``. For example, the number of columns is ``nbins-1``
        for a histogram.

        Parameters
        ----------
        npdf : int
            Total number of distributions that will be written out
        kwargs :
            The keys needed to construct the shape of the data to be written.

        Returns
        -------
        dict[str, tuple[tuple[int, int], str]]
            A dictionary with a key for the objdata, a tuple with the shape of that data,
            and the data type of the data as a string.
            i.e. ``{objdata_key = ( (npdf, n), "f4" )}``

        Raises
        ------
        ValueError
            Raises an error if the means are not provided.
        """
        if "means" not in kwargs:  # pragma: no cover
            raise ValueError("Required argument `means` not included in kwargs")

        ncomp = np.shape(kwargs["means"])[-1]
        return dict(
            means=((npdf, ncomp), "f4"),
            stds=((npdf, ncomp), "f4"),
            weights=((npdf, ncomp), "f4"),
        )

    @classmethod
    def add_mappings(cls) -> None:
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_mixmod_fit_samples, None)

    @classmethod
    def create_ensemble(
        self,
        means: ArrayLike,
        stds: ArrayLike,
        weights: ArrayLike,
        warn: bool = True,
        ancil: Optional[Mapping] = None,
    ) -> Ensemble:
        """Creates an Ensemble of distributions parameterized as Gaussian Mixture models.

        `npdf` = the number of distributions

        `ncomp` = the number of Gaussians in the mixture model

        Parameters
        ----------
        means : ArrayLike
            The means of the Gaussians, with shape (npdf, ncomp)
        stds : ArrayLike
            The standard deviations of the Gaussians, with shape (npdf, ncomp)
        weights : ArrayLike
            The weights to attach to the Gaussians, with shape (npdf, ncomp).
            Weights should sum up to one. If not, the weights are interpreted
            as relative weights.
        warn : bool, optional
            If True, raises warnings if input is not finite. If False, no warnings
            are raised. By default True.
        ancil : Optional[Mapping], optional
            A dictionary of metadata for the distributions, where any arrays have
            the same length as the number of distributions, by default None


        Returns
        -------
        Ensemble
            An Ensemble object containing all of the given distributions.

        Examples
        --------

        To create an Ensemble of two distributions with associated ids:

        >>> import qp
        >>> means = np.array([[0.35, 0.55],[0.23,0.81]])
        >>> stds = np.array([[0.2, 0.25],[0.21, 0.19]])
        >>> weights = np.array([[0.4, 0.6],[0.3,0.7]])}
        >>> ancil = {'ids': [200, 205]}
        >>> ens = qp.mixmod.create_ensemble(means, stds, weights, ancil)
        >>> ens.metadata
        {'pdf_name': array([b'mixmod'], dtype='|S6'), 'pdf_version': array([0])}


        """
        data = {"means": means, "stds": stds, "weights": weights, "warn": warn}
        return Ensemble(self, data, ancil)


mixmod = mixmod_gen

add_class(mixmod_gen)
