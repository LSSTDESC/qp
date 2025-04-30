# Installation

## Basic Installation

To install the basic version of `qp`, you can run the following commands:

```bash
# Clone the repository
git clone https://github.com/LSSTDESC/qp.git
cd qp

# Install
pip install .
```

This installs a minimal version of `qp` that can only write HDF5 files and doesn't have plotting enabled. We recommend you install the full version of `qp` to get all of the available functionality by using the following command:

::::{tab-set}

:::{tab-item} bash

```bash
pip install .[full]
```

:::

:::{tab-item} zsh

```zsh
pip install '.[full]'
```

:::

::::

## Parallel Installation

To install `qp` with parallel functionality, first make sure that your installations of [h5py](https://docs.h5py.org/en/stable/quick.html#) and [HDF5](https://support.hdfgroup.org/documentation/index.html) are built with MPI support. If you are running it in a [conda](https://anaconda.org/anaconda/conda) environment, you can do this by running the following installation command:

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

```{tip}
If you're still having difficulties installing, try using the `environment.yml` file to set up your conda environment (as described in <project:../developer_docs/setup.md>), and then installing `qp` normally as described here.
```
