# Developer Setup

## Developer Environment Setup

For the installation of `qp` for the purpose of development, we recommend that you use a separate [Anaconda](https://docs.anaconda.com/anaconda/install/) virtual environment with `qp` installed in "editable" mode with all dev optional dependencies added.

In this guide, we will name this developer environment `qp_dev` and we will assume that an Anaconda with a minimum python version of 3.9 has been previously installed.

To install the developer environment:

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

## Running tests

- run coverage tests using ./do_cover.sh
- what the output should look like

## Building Documentation

- how to make documentation locally

## Where to go from here

- quick introduction to how qp is structured
  - include a diagram of the code flow here
- introduction to the rest of the section (i.e. link to guidelines, mention development priorities a place to look for things to work on, etc)
