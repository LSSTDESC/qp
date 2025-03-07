# Testing out the conversion function as well as changing the PDF constructor
import numpy as np
import qp


def test_pdf_constructor():

    loc1 = np.array([[0]])
    scale1 = np.array([[1]])
    norm_dist1 = qp.stats.norm.create(loc=loc1, scale=scale1)

    xvals = np.linspace(-5, 5, 11)

    # Define the quantile values to compute the locations for
    quants = np.linspace(0.01, 0.99, 7)
    print(quants)

    # Compute the corresponding locations using the inverse CDF (percent point function)
    locs = np.squeeze(norm_dist1.ppf(quants))  # Ensure locs is 1D

    # Construct the quantile distribution
    quant_dist = qp.quant.create(quants=quants, locs=locs)
    quant_vals = quant_dist.pdf(xvals)

    quant_dist1 = qp.quant.create(
        quants=np.atleast_1d(quants), locs=np.atleast_2d(locs)
    )
    # quant_dist1.dist.pdf_constructor_name = 'piecewise_linear'
    # quant_dist1.dist.pdf_constructor_name = "piecewise_constant"
    # quant_dist1.dist.pdf_constructor_name = "cdf_spline_derivative"
    quant_dist1.dist.pdf_constructor_name = "dual_spline_average"
    print(quant_dist1.dist.pdf_constructor_name)
    quant_dist1.dist.quants
    # PDF and CDF values
    pdf_vals = np.squeeze(quant_dist1.pdf(xvals))
    cdf_vals = np.squeeze(quant_dist1.cdf(xvals))


def test_mixmod():

    qp.mixmod_gen.make_test_data()
    cls_test_data = qp.mixmod_gen.test_data["mixmod"]
    gen_func = cls_test_data["gen_func"]
    ctor_data = cls_test_data["ctor_data"]
    ens = qp.Ensemble(gen_func, ctor_data)
    pdf_vals = ens.pdf(cls_test_data["test_xvals"])
