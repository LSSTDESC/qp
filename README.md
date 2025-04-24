# qp

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/LSSTDESC/qp)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/qp/python-package.yml)
![Read the Docs](https://img.shields.io/readthedocs/qp)

`qp` is a library that allows you to store and manipulate tables of probability distribution data.

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

## Building the documentation

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

- [Alex Malz](https://github.com/LSSTDESC/qp/issues/new?body=@aimalz) (NYU)
- [Phil Marshall](https://github.com/LSSTDESC/qp/issues/new?body=@drphilmarshall) (SLAC)
- [Eric Charles](https://github.com/LSSTDESC/qp/issues/new?body=@eacharles) (SLAC)
- [Sam Schmidt](https://github.com/LSSTDESC/qp/issues/new?body=@sschmidt) (UC Davis)

## Citation

If you end up using any of the code or ideas you find here in your academic research, please cite us as `Malz et al, ApJ 156 1 35`.

## Contribution

If you are interested in this project, please do drop us a line via the hyperlinked contact names above, or by [writing us an issue](https://github.com/LSSTDESC/qp/issues/new). To get started contributing to the `qp` project, take a look at the [Contribution Guidelines](http://qp.readthedocs.io/developer_docs/contribution.html).

## License

The code in this repo is available for re-use under the MIT license (see the [license](./LICENSE) file).
