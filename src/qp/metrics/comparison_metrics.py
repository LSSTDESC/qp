from scipy import stats
import numpy as np


def calculate_kolmogorov_smirnov(rvs, cdf, **kwargs):
    """ Use scipy.stats.kstest to compute the Kolmogorov-Smirnov test statistic for
        the PIT values by comparing with a uniform distribution between 0 and 1. """
    return stats.kstest(rvs, cdf, **kwargs)

def calculate_cramer_von_mises(rvs, cdf, **kwargs):
    """ Use scipy.stats.cramervonmises to compute the Cramer-von Mises statistic for
    the PIT values by comparing with a uniform distribution between 0 and 1. """
    return stats.cramervonmises(rvs, cdf, **kwargs)
