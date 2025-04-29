# Contribution Guidelines

## Conventions to follow

Generally, the code should follow the guidelines given in the [LSST DM Developer Guide](https://developer.lsst.io/index.html). This section lays out some project-specific guidelines.

### Typing Recommendations:

It is recommended to use type hints for the arguments and outputs of functions to improve the ability to develop and understand code. For some tips on how to get started see this [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

- Anything that requires a parameterization class can be typed using the base parameterization class, `Pdf_gen`
- Use the type `ArrayLike` from <inv:#numpy.typing> for a parameter or output that can be anything from a float to a `numpy.ndarray`.

### Naming and placement

- Parameterization classes get their own folder within the `parameterizations/` folder. The folder should be called `[name]/`, where `name` is the name of the parameterization.
  - The class file should be called `[name].py`
  - Supporting files should go in this folder in addition to the class file. The generic supporting file name is `[name]_utils.py`
  - The class itself should be called `[name]_gen`

## Tests

When creating new test files, they should be in the same location within the `tests/` folder as the file that is being tested. The test file should have the name `test_[filename].py`, where `filename` is the name of the file being tested. For example, a test for the `qp/core/ensemble.py` module is called `test_ensemble.py` and located in the `tests/core/` folder.

Test data should be stored in the `test_data/` folder. Any helper functions for tests that are not tests themselves should be placed in the `helpers/` folder.

We recommend using temporary files and directories when writing files during tests, either through [pytest's fixtures](https://docs.pytest.org/en/stable/how-to/tmp_path.html) or through [tempfile](https://docs.python.org/3/library/tempfile.html). This means that if tests fail, then cleanup still functions appropriately and the developer will not be left with an excess of files to clean up.

## Documentation

All documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/index.html). The source files live in the `docs/` folder, and the output is created in the `_build/` folder inside the `docs/` folder. Documentation files are written in Markdown, and any images or other assets are in the `assets/` folder. When new documentation packages are added, make sure to add them to the [`requirements.txt`](https://github.com/LSSTDESC/qp/blob/main/docs/requirements.txt) file, where they will be used when [Read the Docs](https://about.readthedocs.com/) builds the documentation.

### Writing Documentation Pages

When writing new documentation pages, make sure to add them to the relevant `toctree`, either in [`index.rst`](https://github.com/LSSTDESC/qp/blob/main/docs/index.rst) or in the relevant `index.md` file if it is a sub-page.

Tutorial Jupyter notebooks can be placed directly in the `nb/` directory, and then linked to on the `nb/index.md` page. Notebooks will be automatically evaluated and turned into MarkDown via the [myst-nb](https://myst-nb.readthedocs.io/en/v0.13.2/index.html) extension.

### Docstrings

Ideally, all functions and classes should have docstrings, which should be written using [NumPy documentation format](https://numpydoc.readthedocs.io/en/latest/format.html). An example of how to format docstrings for a class using this format is given below:

```{literalinclude} ./docstring_formatting_example.py

```

## Contribution workflow

The contribution workflow described here is pulled from the [RAIL contribution workflow](https://rail-hub.readthedocs.io/en/latest/source/contributing.html).

### Issue

When you identify something that should be done, [make an issue](https://github.com/LSSTDESC/qp/issues/new/choose) for it.

### Branch

Install the code following the [developer installation](setup.md#developer-environment-setup) instructions.
If your branch is addressing a specific issue, the branch name should be `issue/[issue_number]/[title]`, where the `[title]` is a short description of the issue, with `_` separating words.

While developing in a branch, don’t forget to pull from main regularly (at least daily) to make sure your work is compatible with other recent changes.

Make sure that if the issue solves one of the items listed in <project:techdebt.md>, you remove that item from the documentation page. And make sure to update the documentation with any changes that alter `qp`'s functionality, or add examples and additional pages as necessary to explain any additions.

When you’re ready to merge your branch into the main branch, create a pull request (PR) in the `qp` repository. GitHub has instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

Several continuous integration checks will be performed for new pull requests. If any of these automatic processes find issues with the code, you should address them in the branch before sending for review. These include tests (does the code function correctly), [Pylint](https://docs.pylint.org/) (code style), and coverage (how much code is exercised in unit tests).

Once you are satisfied with your PR, request that other team members review and approve it. You could send the request to someone whom you’ve worked with on the topic, or one of the core maintainers of `qp`.

### Merge

Once the changes in your PR have been approved, these are your next steps:

- merge the change by selecting “Squash and merge” on the approved pull request
- enter `closes #[#]` in the comment field to close the resolved issue
- delete your branch using the button on the merged pull request.

### Reviewing a PR

To review a pull request, it’s a good idea to start by pulling the changes and running the tests locally (see <project:setup.md#running-tests> for instructions).

Check the code for complete and accurate docstrings, sufficient comments, and ensure any instances of `#pragma: no cover` (excluding the code from unit test coverage accounting) are extremely well-justified.

Feel free to mark the PR with “Request changes” for necessary changes. e.g. writing an exception for an edge case that will break the code, updating names to adhere to the naming conventions, etc.

It is also considered good practice to make suggestions for optional improvements, such as adding a one-line comment before a clever block of code or including a demonstration of new functionality in the example notebooks.

## Version Release and Deployment Procedures

### Publishing Package on PyPI

There is a Github Action that will publish the package to [PyPI](https://pypi.org/project/qp-prob/) after a new release is created.

### Making the Documentation on "Read The Docs"

Read the Docs is linked to the [github repo](https://github.com/LSSTDESC/qp), and will rebuild the docs when there are any changes to `main`.

### Informing Developers of Downstream Packages

`qp` is a core package of the LSST DESC RAIL ecosystem. Consequently, the developers of the following packages should be informed about new versions:

- [`qp_flexzboost`](https://github.com/LSSTDESC/qp_flexzboost)
- [`rail`](https://github.com/LSSTDESC/rail)
- [`rail_base`](https://github.com/LSSTDESC/rail_base)
- [`rail_bpz`](https://github.com/LSSTDESC/rail_bpz)
- [`rail_cmnn`](https://github.com/LSSTDESC/rail_cmnn)
- [`rail_delight`](https://github.com/LSSTDESC/rail_delight)
- [`rail_dnf`](https://github.com/LSSTDESC/rail_dnf)
- [`rail_dsps`](https://github.com/LSSTDESC/rail_dsps)
- [`rail_fsps`](https://github.com/LSSTDESC/rail_fsps)
- [`rail_gpz_v1`](https://github.com/LSSTDESC/rail_gpz_v1)
- [`rail_lephare`](https://github.com/LSSTDESC/lephare)
- [`rail_pzflow`](https://github.com/LSSTDESC/rail_pzflow)
- [`rail_sklearn`](https://github.com/LSSTDESC/rail_sklearn)
- [`rail_som`](https://github.com/LSSTDESC/rail_som)
- [`rail_tpz`](https://github.com/LSSTDESC/rail_tpz)
