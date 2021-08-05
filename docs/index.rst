============================================
qp : quantile-parametrized PDF approximation
============================================

In a scientific inference we typically seek to characterize the
posterior probability density function (PDF) for our parameter(s),
which means we need to fund a suitable, calculable approximation to it.
Popular choices include an ensemble of samples, a histogram estimator
based on those samples, or (in 1 dimensional problems) a tabulation of
the PDF on a regular parameter grid. `qp` is a python package that
supports these approximations, as well as the "quantile
parameterization" from which the package gets its name.


Quick examples
--------------


Building an ensemble

.. code-block:: python

    #Here we will create 100 Gaussians with means distributed between -1 and 1
    #and widths distributed between 0.9 and 1.
    
    locs = 2* (np.random.uniform(size=(100,1))-0.5)
    scales = 1 + 0.2*(np.random.uniform(size=(100,1))-0.5)
    ens_n = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))


    #Here we create a 100 PDF using the histogram representation and
    #61 bins running from 0 to 3, filled with random values
    bins = np.linspace(0, 3, 61)
    pdfs = np.random.random((100, 61))

    ens_h = qp.Ensemble(qp.hist, data=dict(bins=bins, pdfs=pdfs))


Evaluating functions of the distribution

.. code-block:: python

    # Here we will evaluate the pdfs() and cdfs() on a grid of 501 points from 0 to 3.
    # Since there are 100 pdfs, this will return an array of shape (100, 501)
    grid = np.linscape(0, 3, 501)
    pdf_vals = ens_n.pdf(grid)
    cdf_vals = ens_n.cdf(grid)


Converting an ensemble to a different representation

.. code-block:: python

    #Here we convert our ensemble of Gaussians to an representation using an interpolated grid
    #with 201 grid points between 0 and 3
    
    xvals = np.linspace(0, 3, 201)
    ens_i = ens_n.convert_to(qp.interp_gen, xvals=vals)
 

Reading and writing ensembles

.. code-block:: python

    #Here we write our interpolated grid representation ensemble to an hdf5 file and read it back
    ens_i.writeto("qp_interp_ensemble.hdf5")

    ens_read = qp.read("qp_interp_ensemble.hdf5")
    

Computing a point estimate and storing it with the ensemble

.. code-block:: python

    #Here compute the mode (i.e., the location of the maximum) on a grid of 501 values
    modes = ens_i.mode(grid=np.linspace(0, 3, 501)
    ens_i.set_ancil(dict(modes=modes))


Sampling the pdf and storing the samples

.. code-block:: python

    #Pull 5 samples from each PDF, this will return an array of shape (100, 5)
    samples = ens_i.rvs(size=5)
    ens_i.set_ancil(dict(samples=samples))



Documentation Contents
----------------------

.. toctree::
   :includehidden:
   :maxdepth: 3

   install
   tutorials
   contributing
   qp
   
