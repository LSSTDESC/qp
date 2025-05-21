# Developer Setup

## Developer Environment Setup

For the installation of `qp` for the purpose of development, we recommend that you use a separate [Anaconda](https://docs.anaconda.com/anaconda/install/) virtual environment with `qp` installed in "editable" mode with all dev optional dependencies added.

In this guide, we will name this developer environment `qp_dev` and we will assume that an Anaconda with a minimum Python version of 3.10 has been previously installed.

To install the developer environment:

::::{tab-set}

:::{tab-item} bash

```bash
# Clone the repo and enter it
git clone https://github.com/LSSTDESC/qp.git
cd qp

# Creating the environment from the YAML
conda env create -n qp_dev -f environment.yml

# Activate the environment
conda activate qp_dev

# Install qp in editable mode with dev dependencies
pip install -e .[dev]
```

:::

:::{tab-item} zsh

```bash
# Clone the repo and enter it
git clone https://github.com/LSSTDESC/qp.git
cd qp

# Creating the environment from the YAML
conda env create -n qp_dev -f environment.yml

# Activate the environment
conda activate qp_dev

# Install qp in editable mode with dev dependencies
pip install -e '.[dev]'
```

:::

::::

## Running tests

To run coverage tests, run the following on the command line from the base `qp` directory.

```bash

./do_cover.sh

```

The output is in the `cover/` folder, which will output HTML files that provide an overview of the coverage status. For a pull request, the goal is to have 100% coverage.

## Building Documentation

To build the documentation locally, start by making sure that you have the appropriate documentation packages installed:

::::{tab-set}

:::{tab-item} bash

```bash
pip install -e .[docs]

```

:::

:::{tab-item} zsh

```bash
pip install -e .[docs]

```

:::

::::

Once you have the appropriate packages, run the following lines of code to make the documentation:

```bash
cd docs/
make html

```

The HTML files will be generated in the `_build/` folder inside the `docs/` folder.

## Where to go from here

Now that you've got a development environment set up, take a look at <project:codestructure.md> to get a sense of how `qp` is structured. Make sure to read over the <project:contribution.md> before getting started, as this covers the intended workflow for developers to follow, in addition to expected naming conventions and so on.

If you are looking to work on a new parameterization type, take a look at <project:parameterizationcontribution.md> for detailed instructions and a code template to get you started. Or if you're looking for ideas of where to start, take a look at <project:techdebt.md> or the open [issues](https://github.com/LSSTDESC/qp/issues).
