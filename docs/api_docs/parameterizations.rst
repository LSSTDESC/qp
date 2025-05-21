.. _parameterization-types:
Parameterization types
======================

Histogram based
---------------

.. autoclass :: qp.hist_gen
    :members: 
    :show-inheritance:
    :undoc-members:


Utility functions
^^^^^^^^^^^^^^^^^

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

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.interp.interp_utils
    :members:

Quantile based
--------------

.. autoclass :: qp.quant_gen
    :members:
    :show-inheritance:
    :undoc-members:

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.quant.quant_utils
    :members:


.. autoclass :: qp.parameterizations.quant.abstract_pdf_constructor.AbstractQuantilePdfConstructor
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass :: qp.parameterizations.quant.cdf_spline_derivative.CdfSplineDerivative
    :members:
    :show-inheritance:
    :undoc-members:
   
.. autoclass :: qp.parameterizations.quant.dual_spline_average.DualSplineAverage
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass :: qp.parameterizations.quant.piecewise_constant.PiecewiseConstant
    :members:
    :show-inheritance:
    :undoc-members:

.. autoclass :: qp.parameterizations.quant.piecewise_linear.PiecewiseLinear
    :members:
    :show-inheritance:
    :undoc-members:

Gaussian mixture model based
----------------------------

.. autoclass :: qp.mixmod_gen
    :members:
    :show-inheritance:
    :undoc-members:

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.mixmod.mixmod_utils
    :members:

Spline based
------------

.. autoclass :: qp.spline_gen
    :members:
    :show-inheritance:
    :undoc-members:       

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.spline.spline_utils
    :members:


Packed Interpolation
--------------------

.. autoclass :: qp.packed_interp_gen
    :members:
    :show-inheritance:
    :undoc-members:

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.packed_interp.packing_utils
    :members:

Sparse Interpolation
--------------------

.. autoclass :: qp.sparse_gen
    :members:
    :show-inheritance:
    :undoc-members:

Utility functions
^^^^^^^^^^^^^^^^^

.. automodule:: qp.parameterizations.sparse_interp.sparse_rep
    :members:

.. automodule:: qp.parameterizations.sparse_interp.sparse_utils
    :members: