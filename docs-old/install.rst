.. _install:

Installing qp
=============

For now the only installation type that is support is installing from source.   Since ``qp`` is all python code, this is pretty easy.


Installing from source
-----------------------

.. code-block:: bash

    git clone https://github.com/LSSTDESC/qp.git
    cd qp
    pip install .

Installing for development
--------------------------

If you're installing with the intention of developing ``qp``, you'll need a few more steps.

1. Install MPI from source following `these instructions <https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi>`_. e.g.

.. code-block:: bash

    sudo apt-get install mpich

2. Install all dependencies

.. code-block:: bash

    git clone https://github.com/LSSTDESC/qp.git
    cd qp
    pip install -e .[dev]

Or for installing on a Mac:

.. code-block:: bash

    pip install '.[dev]'

3. Run ``pytest`` to confirm that you can build and run ``qp`` in your environment.
