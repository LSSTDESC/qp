.. _contributing:


Contributing to qp
==================


Making a new distribution type
------------------------------

Here is a checklist of things that you will need to include in a class that implements a new type of distritubion.

1.  What type of distribution are you making?
    
   1. A "simple" distribution, i.e., a distribution that is defined by
      a fixed set of parameters.  In that case you should implement
      your class as a sub-class of `scipy.stats.rv_continuous` and
      then use the `qp.factory._make_scipy_wrapped_class` class to
      extend it to qp pdf class.
   2. A "row-based" distribution i.e., distributions that are
      configured by providing a variable sized set of paramters, where
      each row corresponds to one PDF, such as a histogram or grid.
      In that case you should inherit from the `qp.Pdf_rows_gen`
      class.

2.  In the static block before the first class method you should
    define the name the class will go by, the version of the class
    (used to convert older version when reading them back from disk),
    and the mask across which the distribution is supported.
    
.. code-block:: python
		
    name = 'hist'
    version = 0

    _support_mask = rv_continuous._support_mask

    
      
3.  In the constuctor of the class you should store whatever information you will need to evaluate the PDF and make sure that it is consistent.  Here is an example from the histogram implementation.   Note how the `check_input` keyword is used to allow you to skip the normalization step of the PDF, if you know that they are normalized.

.. code-block:: python
    
    self._hbins = np.asarray(bins)
    self._nbins = self._hbins.size - 1
    self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
    if np.shape(pdfs)[-1] != self._nbins: # pragma: no cover
	raise ValueError("Number of bins (%i) != number of values (%i)" % (self._nbins, np.shape(pdfs)[-1]))
    check_input = kwargs.pop('check_input', True)
    if check_input:
        sums = np.sum(pdfs*self._hbin_widths, axis=1)
	self._hpdfs = (pdfs.T / sums).T
    else: #pragma: no cover
	self._hpdfs = pdfs

4.  In the constructor of the class you should extract the number of PDF and pass them to the base class constructor.
   
.. code-block:: python

    kwargs['npdf'] = pdfs.shape[0]
    super(hist_rows_gen, self).__init__(*args, **kwargs)

5.  In the constructor you should define which data members of the class are "data" and "metadata".   In this context, "data" means quantites that are defined for each PDF, and "metadata" means quantities that are shared between all the PDFs.   This should be the minimal set of data need to reconstruct the class instance. 
    
.. code-block:: python

    self._addmetadata('bins', self._hbins)
    self._addobjdata('pdfs', self._hpdfs)

6.  You should provide properties to access each of the "data" and "metadata" fields.

.. code-block:: python

    @property
    def bins(self):
        """Return the histogram bin edges"""
        return self._hbins

    @property
    def pdfs(self):
        """Return the histogram bin values"""
        return self._hpdfs

7.  At a minimum you need to implement either the `_pdf` `_cdf` scipy hook functions to evaluate the PDF.  Optionally you can implement the `_sf`, `_ppf`, `_isf`, `_rvs` functions as well, for faster evaluate.   See below for some comments on how to make these evaluation functions fast.

.. code-block:: python

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        return evaluate_unfactored_hist_x_multi_y(x, row, self._hbins, self._hpdfs)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._hcdfs is None: #pragma: no cover
            self._compute_cdfs()
        if np.shape(x)[:-1] == np.shape(row)[:-1]:
            return interpolate_unfactored_x_multi_y(x, row, self._hbins, self._hcdfs, bounds_error=False, fill_value=(0.,1.))
        return interp1d(self._hbins, self._hcdfs[np.squeeze(row)], bounds_error=False, fill_value=(0.,1.))(x)  # pragma: no cover

8.  You should implement the `_updated_ctor_param` function that scipy needs in order to copy distributions.   This should make a dictionary of all the constructor parameters.

.. code-block:: python

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(hist_rows_gen, self)._updated_ctor_param()
        dct['bins'] = self._hbins
        dct['pdfs'] = self._hpdfs
        return dct


9.  You should define functions to convert other ensembles to this
    representation.  Doing that requires two things: 1) a function to
    extract values for the orignal representation, and 2) a function to 
    to use those values to create a new ensemble.  Finally, you have to
    add those mappings to the dictionaries that the class carries with it.
    conversions happen. `None` is used as a wildcard to catch any
    values that are not explicitly defined.
    
.. code-block:: python
    
    @classmethod
    def add_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(convert_using_hist_values, None)
        cls._add_extraction_method(convert_using_hist_samples, "samples")


10.  If you want, you can define a particular method for plotting
     distributions of the class that better capture the representation
     of the PDF by adding a `plot_native` method to the class.

.. code-block:: python
     
    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a histogram this shows the bin edges
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        vals = pdf.dist.pdfs[pdf.kwds['row']]
        return plot_pdf_histogram_on_axes(axes, hist=(pdf.dist.bins, vals), **kw)
     
11.  After the class definiton, you need to register the class with
     the factory, and make the creation function available.

.. code-block:: python

    hist = hist_gen.create
    add_class(hist_gen)


12.  After the class definition, you can also add test data to the
     class so that it will be tested in the automatically generated
     tests.   The test data takes the form of a multi-level
     dictionary.  At the top level each key-value pair will be used
     for four tests:

     1. Creating a distribution and making sure that the
	pdf functions are well-behaved.
     2. Writing the distribution to disk
	and reading it back and making sure it is the same, 
     3. Converting a normal distribution to a distribution of this
	type and making sure it is reasonably close to the original.
     4. Testing the plotting functions. 	
     
.. code-block:: python
		
    @classmethod
    def make_test_data(cls):
        """ Make data for unit tests """
        hist_gen.test_data = dict(hist=dict(gen_func=hist, ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),\
                                            convert_data=dict(bins=XBINS), test_xvals=TEST_XVALS),
                                  hist_samples=dict(gen_func=hist, ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),\
                                                    convert_data=dict(bins=XBINS, method='samples',\
                                                                                  size=NSAMPLES),\
                                                    atol_diff=1e-1, atol_diff2=1e-1,\
                                                    test_xvals=TEST_XVALS, do_samples=True))



	
Checks for new code
-------------------

There are a number of checks that will need to pass before a pull request adding new code will be accepted.  These should all be implemented in the travis automated testing, but it can also be useful to run them yourself before you make the pull request.


Running pylint
--------------

There is a .pylintrc file defining the style that we want.   You can run any changes against that by doing:

.. code-block:: bash

    pylint qp

Please correct any and all messages.   It a very few cases you can disable specific warnings in specific functions, for example by adding

.. code-block:: python

    # pylint: disable=arguments-differ

To the function in question.


Adding unit tests for your class
--------------------------------

If you have implemented the `make_test_data` classmethod, then up to four sets unit tests will be automatcially 
generated for your class.  These are built by the `PDFTestCase.auto_add_class` function in `qp/tests/test_auto.py`.
The actual functions are in `qp/test_funcs.py`; they are:

1.  pdf functionality tests, which runs a set of consistency checks to make sure that the pdf is well defined and to test
    that the relationships between `pdf()`, `cdf()`, `sf()`, `ppf()`, etc.. are consistent.

2.  persistence tests, which runs a loopback test that write the class to disk in various formats and reads it back
    and verifies that the result is identical to the original.

3.  conversion tests, which verifies that converting to the class works by comparing the `pdf()` values computed on a grid
    from an input ensemble in a different representation to values in your classes representation.

4.  plotting tets, which verifies that the plotting function doesn't crash.  Making sure the output is sensible is up to you. 


Running unit tests
------------------

You can use the `do_cover.sh` script to run the unit test and check their coverage.  We will require 100\% coverage, but it is ok to use `#pragma: no cover` statements to skip error blocks.

.. code-block:: python

    ./do_cover.sh


#### Running demo notebooks

There are some demo notebooks in `qp` you can verify that they all work by rendering them to html.

.. code-block:: bash

    ./render_nb.sh



