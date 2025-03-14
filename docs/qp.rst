.. _qp:

========================
API Documentation for qp
========================

`qp` provides a `PDF` class object, that builds on the
`scipy.stats` distributions to provide various approximate forms.
The package also contains some `utils` and `metrics` for quantifying the quality of
these approximations.


Ensemble
========

.. autoclass:: qp.Ensemble
    :members:
    :undoc-members:

    
Factory 
=======

.. automodule:: qp.core.factory
    :members:
    :undoc-members:
       

.. _parameterization-types:
Parameterization types
======================

Histogram based
---------------

.. autoclass :: qp.hist_gen
    :members: 
    :show-inheritance:
    :undoc-members:


.. automodule:: qp.parameterizations.hist.hist_utils
    :members:
		      
Interpolation of a fixed grid
-----------------------------
		      
.. autoclass :: qp.interp_gen
    :members:
    :show-inheritance:
    :undoc-members:


		      
Interpolation of a non-fixed grid
---------------------------------
		      
.. autoclass :: qp.interp_irregular_gen
    :members:
    :show-inheritance:
    :undoc-members:

Quantile based
--------------

.. autoclass :: qp.quant_gen
    :members:
    :show-inheritance:
    :undoc-members:

   
Gaussian mixture model based
----------------------------

.. autoclass :: qp.mixmod_gen
    :members:
    :show-inheritance:
    :undoc-members:

Spline based
------------

.. autoclass :: qp.spline_gen
    :members:
    :show-inheritance:
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

`qp.utils.array`: Array utility functions
-----------------------------------------

.. automodule:: qp.utils.array
    :members:
    :undoc-members:

`qp.utils.conversion`: Conversion utility functions
---------------------------------------------------
       
.. automodule:: qp.utils.conversion
    :members:
    :undoc-members:

`qp.utils.dictionary`: Multi-level dictionary manipulation
----------------------------------------------------------

.. automodule:: qp.utils.dictionary
    :members:
    :undoc-members:
       
`qp.utils.interpolation`: PDF evaluation and construction utility functions
----------------------------------------------------------------------------

.. automodule:: qp.utils.interpolation
    :members:
    :undoc-members:


      
Infrastructure
==============

`qp.pdf_gen`: `scipy.stats` interface
-------------------------------------

.. automodule:: qp.parameterizations.base
    :members:
    :undoc-members:



`qp.plotting`: Tools for PDF plotting
-------------------------------------

.. automodule:: qp.plotting
    :members:
    :undoc-members:



