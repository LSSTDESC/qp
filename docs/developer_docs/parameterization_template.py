from qp.parameterizations.base_parameterization import Pdf_rows_gen
from scipy.stats import rv_continuous


class hist_gen(Pdf_rows_gen):
    """This class parameterizes a distribution as a histogram.

    Notes
    -----

    The relevant data members are:

    bins:  n+1 bin edges (shared for all PDFs)

    pdfs:  (npdf, n) bin values

    Inside a given bin the pdf() will return the pdf value.
    Outside the range bins[0], bins[-1] the pdf() will return 0.

    Inside a given bin the cdf() will use a linear interpolation across the bin
    Outside the range bins[0], bins[-1] the cdf() will return (0 or 1), respectively

    The ppf() is computed by inverting the cdf().
    ppf(0) will return bins[0]
    ppf(1) will return bins[-1]
    """

    name = "hist"
    version = 0

    _support_mask = rv_continuous._support_mask
