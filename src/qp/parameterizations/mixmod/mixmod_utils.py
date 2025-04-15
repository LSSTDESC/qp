from __future__ import annotations

import numpy as np

from ...core.lazy_modules import mixture


def extract_mixmod_fit_samples(
    in_dist: "Ensemble", **kwargs
) -> dict[str, np.ndarray[float]]:
    """Convert to a mixture model using a set of values sampled from the pdf

    Parameters
    ----------
    in_dist : Ensemble
        Input distributions

    Other Parameters
    ----------------
    ncomps : int
        Number of components in mixture model to use
    nsamples : int
        Number of samples to generate
    random_state : int
        Used to reproducibly generate random variate from in_dist

    Returns
    -------
    data : dict[str, np.ndarray[float]]
        The extracted data
    """
    n_comps = kwargs.pop("ncomps", 3)
    n_sample = kwargs.pop("nsamples", 1000)
    random_state = kwargs.pop("random_state", None)
    samples = in_dist.rvs(size=n_sample, random_state=random_state)

    def mixmod_helper(samps):
        estimator = mixture.GaussianMixture(n_components=n_comps)
        estimator.fit(samps.reshape(-1, 1))
        weights = estimator.weights_
        means = estimator.means_[:, 0]
        stdevs = np.sqrt(estimator.covariances_[:, 0, 0])
        ov = np.vstack([weights, means, stdevs])
        return ov

    vv = np.vectorize(mixmod_helper, signature="(%i)->(3,%i)" % (n_sample, n_comps))
    fit_vals = vv(samples)
    return dict(
        weights=fit_vals[:, 0, :], means=fit_vals[:, 1, :], stds=fit_vals[:, 2, :]
    )
