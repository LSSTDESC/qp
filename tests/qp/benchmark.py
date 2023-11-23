""" Timing for qp distributions """

import time

import numpy as np

import qp

testfile = "qp_test_ensemble.hdf5"


def time_ensemble(ens):
    """Time main scipy functions"""

    zv = np.linspace(0.0, 2.5, 201)
    quants = np.linspace(0.01, 0.99, 50)
    nsamples = 100

    print(
        "Timing %s on %i PDFS, with %i grid points, %i quantiles and %i samples"  # pylint: disable=bad-string-format-type
        % (
            type(ens.gen_obj),
            ens.frozen.npdf,
            zv.size,
            quants.size,
            nsamples,
        )
    )

    t0 = time.time()
    _ = ens.pdf(zv)
    t1 = time.time()
    print("pdf  %.2f s" % (t1 - t0))

    t0 = time.time()
    _ = ens.cdf(zv)
    t1 = time.time()
    print("cdf  %.2f s" % (t1 - t0))

    t0 = time.time()
    _ = ens.ppf(quants)
    t1 = time.time()
    print("ppf  %.2f s" % (t1 - t0))

    t0 = time.time()
    _ = ens.sf(zv)
    t1 = time.time()
    print("sf   %.2f s" % (t1 - t0))

    t0 = time.time()
    _ = ens.isf(zv)
    t1 = time.time()
    print("isf  %.2f s" % (t1 - t0))

    t0 = time.time()
    _ = ens.rvs(size=nsamples)
    t1 = time.time()
    print("rvs  %.2f s" % (t1 - t0))


def time_convert(ens, cls_to, **kwds):
    """Time conversion function"""
    t0 = time.time()
    ens_out = ens.convert_to(cls_to, **kwds)
    t1 = time.time()
    print(
        "Convert %s to %s with %i pdfs in %.2f s"
        % (type(ens.gen_obj), cls_to, ens.frozen.npdf, t1 - t0)
    )
    return ens_out


def main():
    """Main"""
    t0 = time.time()
    ens_orig = qp.read(testfile)
    t1 = time.time()
    print("Read %.2f s" % (t1 - t0))

    time_ensemble(ens_orig)

    bins = np.linspace(0.0, 2.5, 101)
    quants = np.linspace(0.01, 0.99, 50)

    ens_i = time_convert(ens_orig, qp.interp_gen, xvals=bins)
    time_ensemble(ens_i)

    ens_h = time_convert(ens_orig, qp.hist_gen, bins=bins)
    time_ensemble(ens_h)

    ens_q = time_convert(ens_orig, qp.quant_gen, quants=quants)
    time_ensemble(ens_q)

    # ens_s = time_convert(ens_orig, qp.spline_gen, xvals=bins)
    # skip this, it sucks
    # time_ensemble(ens_s)


if __name__ == "__main__":
    main()
