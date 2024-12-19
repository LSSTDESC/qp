.. _qp:


API Documentation for qp
************************

`qp` provides a `PDF` class object, that builds on the
`scipy.stats` distributions to provide various approximate forms.
The package also contains some `utils` and `metrics` for quantifying the quality of
these approximations.


Ensemble and Factory
====================

.. automodule:: qp.ensemble
    :members:
    :undoc-members:

    
.. automodule:: qp.factory
    :members:
    :undoc-members:
       

Distribution types
==================

Histogram based
---------------

.. autoclass :: qp.hist_gen
    :members: 
    :show-inheritance:
    :undoc-members:
    :exclude-members: test_data

		      
Interpolation of a fixed grid
-----------------------------
		      
.. autoclass :: qp.interp_gen
    :members:
    :show-inheritance:
    :undoc-members:
    :exclude-members: test_data

		      
Interpolation of a non-fixed grid
---------------------------------
		      
.. autoclass :: qp.interp_irregular_gen
    :members:
    :show-inheritance:
    :undoc-members:
    :exclude-members: test_data


Spline based
------------

.. autoclass :: qp.spline_gen
    :members:
    :show-inheritance:
    :undoc-members:       
    :exclude-members: test_data


Quantile based
--------------

.. autoclass :: qp.quant_gen
    :members:
    :show-inheritance:
    :undoc-members:
    :exclude-members: test_data

		      
Gaussian mixture model based
----------------------------

.. autoclass :: qp.mixmod_gen
    :members:
    :show-inheritance:
    :undoc-members:
    :exclude-members: test_data

		      
		      
`scipy` distributions
---------------------

.. automodule :: qp.scipy_pdfs
    :members:
    :undoc-members:


              
Quantification Metrics
======================
.. automodule:: qp.metrics.metrics
    :members:
    :undoc-members:

.. automodule:: qp.metrics.array_metrics
    :members:
    :undoc-members:

.. automodule:: qp.metrics.brier
.. autoclass:: Brier
    :members:

.. automodule:: qp.metrics.pit
.. autoclass:: PIT
    :members:

Utility functions
=================

`qp.conversion_funcs`
----------------------
       
.. automodule:: qp.conversion_funcs
    :members:
    :undoc-members:

       
`qp.utils`: PDF evaluation and construction utility functions
-------------------------------------------------------------

.. automodule:: qp.utils
    :members:
    :undoc-members:


      
Infrastructure and Core functionality
=====================================

`qp.pdf_gen`: `scipy.stats` interface
-------------------------------------

.. automodule:: qp.pdf_gen
    :members:
    :undoc-members:

       
`qp.dict_utils` tools for multi-level dictionary manipulation
-------------------------------------------------------------
    
.. automodule:: qp.dict_utils
    :members:
    :undoc-members:


`qp.plotting`: Tools for PDF plotting
-------------------------------------

.. automodule:: qp.plotting
    :members:
    :undoc-members:



