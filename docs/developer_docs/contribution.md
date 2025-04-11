# Contribution Guidelines

## Conventions to follow

Generally, the code should follow the guidelines given in the [LSST DM Developer Guide](https://developer.lsst.io/index.html). This section lays out some project-specific guidelines.

### Typing Recommendations:

It is recommended to use type hints for the arguments and outputs of functions to improve the ability to develop and understand code. For some tips on how to get started see this [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

### Naming

- name_pdf for parameterizations

### Placement

- parameterizations go in the parameterizations folder
- analytic parameterizations go in a separate folder
- if you have one that needs supporting functions, create a folder and place the class file and the supporting files in that folder
- conversion functions go in the conversion_funcs file

## Tests

When creating new test files, they should be in the same location within the `tests/` folder as the file that is being tested. The test file should have the name `test_[filename].py`, where `filename` is the name of the file being tested. For example, a test for the `src/tables_io/conv/conv_table.py` module is called `test_conv_table.py` and located in the `tests/conv/` folder.

- what is expected in terms of code coverage, what tests need to be written, types of tests, etc
- tests should exist purely in test folder
- test data stored in tests/test_data
  - what test data should look like
- output of tests should be written to temporary path for ease of clean up

## Documentation

All documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/index.html). The source files live in the `docs/` folder, and the output is created in the `_build/` folder inside the `docs/` folder. Documentation files are written in Markdown, and any images or other assets are in the `assets/` folder. When new documentation packages are added, make sure to add them to the [`requirements.txt`](https://github.com/LSSTDESC/qp/blob/main/docs/requirements.txt) file, where they will be used when [Read the Docs](https://about.readthedocs.com/) builds the documentation.

### Writing Documentation Pages

When writing new documentation pages, make sure to add them to the relevant `toctree` in [`index.rst`](https://github.com/LSSTDESC/qp/blob/main/docs/index.rst).

Tutorial Jupyter notebooks can be placed directly in the `nb/` directory, and then linked to on the `nb/index.md` page. Notebooks will be automatically evaluated and turned into MarkDown via the [myst-nb](https://myst-nb.readthedocs.io/en/v0.13.2/index.html) extension.

- every function should have docstrings describing what the function does
- what is expected in terms of documentation upkeep (i.e. if you change something and the documentation is now out of date)

## Contribution workflow

The contribution workflow described here is pulled from the [RAIL contribution workflow](https://rail-hub.readthedocs.io/en/latest/source/contributing.html).

### Issue

When you identify something that should be done, [make an issue](https://github.com/LSSTDESC/qp/issues/new/choose) for it.

### Branch

Install the code following the [developer installation](setup.md#developer-environment-setup) instructions.
If your branch is addressing a specific issue, the branch name should be `issue/[issue_number]/[title]`, where the `[title]` is a short description of the issue, with `_` separating words.

While developing in a branch, don’t forget to pull from main regularly (at least daily) to make sure your work is compatible with other recent changes.

Make sure that if the issue solves one of the items listed in <project:techdebt.md>, you remove that item from the documentation page.

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

Read the Docs is linked to the [github repo](https://github.com/LSSTDESC/tables_io), and will rebuild the docs when there are any changes to `main`. If there are any issues with this process, contact the project maintainer, Eric Charles.

### Informing Developers of Downstream Packages

`qp` is a core package of the LSST DESC RAIL ecosystem. Consequently, the developers of the following packages should be informed about new versions:

- [`rail`](https://github.com/LSSTDESC/rail)
