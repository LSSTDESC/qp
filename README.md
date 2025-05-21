# qp

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/LSSTDESC/qp)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/qp/python-package.yml)
![Read the Docs](https://img.shields.io/readthedocs/qp)

`qp` is a python library for the storage and manipulation of tables of probability distributions.

### Features

- Read and write tables of probability distributions to/from file
- Parameterize probability distributions inferred from real data
- Convert between different methods of parameterizing probability distributions
- Perform statistical methods on many distributions at a time

### Links

- [Read the Docs](http://qp.readthedocs.io/)
- [PyPI](https://pypi.org/project/qp-prob/)

## Installation

For a basic install of `qp`:

```bash

git clone https://github.com/LSSTDESC/qp.git
cd qp
pip install .

```

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

For more details see the [installation instructions](http://qp.readthedocs.io/user_guide/installation.html) on Read the Docs.

## Building the documentation locally

To build the documentation locally, start by making sure that you have the appropriate documentation packages installed:

```bash
pip install -e '.[docs]'

```

Once you have the appropriate packages, run the following lines of code to make the documentation:

```bash
cd docs/
make html

```

The HTML files will be generated in the `_build/` folder inside the `docs/` folder.

## People

See the [contributors page](https://github.com/LSSTDESC/qp/graphs/contributors) for an up-to-date list of the major contributors. Some of the main contributors are listed here:

- [Alex Malz](https://github.com/LSSTDESC/qp/issues/new?body=@aimalz)
- [Phil Marshall](https://github.com/LSSTDESC/qp/issues/new?body=@drphilmarshall)
- [Eric Charles](https://github.com/LSSTDESC/qp/issues/new?body=@eacharles)
- [Sam Schmidt](https://github.com/LSSTDESC/qp/issues/new?body=@sschmidt)

## Citation

If you end up using any of the code or ideas you find here in your academic research, please cite our paper: [A. I. Malz et al 2018 AJ 156 35](https://ui.adsabs.harvard.edu/abs/2018AJ....156...35M/abstract) ([ADS - BibTex](https://ui.adsabs.harvard.edu/abs/2018AJ....156...35M/exportcitation)).

## Contribution

If you are interested in this project, please [write us an issue](https://github.com/LSSTDESC/qp/issues/new). Before contributing to the `qp` project, take a look at the [Contribution Guidelines](http://qp.readthedocs.io/developer_docs/contribution.html).

## License

The code in this repo is available for re-use under the MIT license (see the [license](./LICENSE) file).
