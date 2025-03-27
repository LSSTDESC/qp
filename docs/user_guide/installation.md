# Installation

## Basic Installation

To install the basic version of `qp`, you can run the following commands:

```bash
git clone https://github.com/LSSTDESC/qp.git
cd qp
pip install .
```

## Parallel Installation

To install `qp` with parallel functionality, first make sure that your installations of h5py and HDF5 are built with MPI support. If you are running it in a conda environment, you can do this by running the following installation command:

```bash

conda install "h5py>=2.9=mpi_openmpi*"  # 2.9 is the first version built with mpi on this channel

```

If you run into errors with this, try adding the "conda-forge" channel:

```bash
conda install conda-forge::"h5py>=2.9=mpi_openmpi*"
```

This should install HDF5 and mpi4py as well. If not, you can install HDF5 via the following:

```bash
conda install "hdf5=*=*mpi_openmpi*"
```

You may also need to install [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html), which can be done through pip:

```bash
pip install mpi4py
```

or conda:

```bash
conda install mpi4py
```

Then you can use the same installation command as above to install `qp`:

```bash
pip install .
```

## Installation on NERSC
