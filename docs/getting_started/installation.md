# Installation

## Basic Installation

To install `qp`, you can run the following commands:

```bash
git clone https://github.com/LSSTDESC/qp.git
cd qp
pip install .
```

## Parallel Installation

To install `qp` with parallel functionality, make sure that your installations of `h5py` and `hdf5` are built with MPI support. If you are running it in a `conda` environment, you can do this by running the following installation commands:

```bash

conda install "h5py>=2.9=mpi_openmpi*"  # 2.9 is the first version built with mpi on this channel
conda install hdf5=*=*mpi_openmpi*

```

You will also need to install [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html), which can be done through `pip`:

```bash
pip install mpi4py
```

or `conda`:

```bash
conda install mpi4py
```

- how to install on NERSC if different
