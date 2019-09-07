**IMPORTANT:** The `pvfit-m` repo now redirects here. Its contents are found in
[this subpackage](pvfit/measurement/spectral_correction/README.md).

# pvfit

PVfit: Photovoltaic (PV) Device Performance Measurement and Modeling

[![Build Status](https://dev.azure.com/markcampanelli/markcampanelli/_apis/build/status/markcampanelli.pvfit?branchName=master)](https://dev.azure.com/markcampanelli/markcampanelli/_build/latest?definitionId=1&branchName=master)
[![Coverage](https://img.shields.io/azure-devops/coverage/markcampanelli/markcampanelli/1.svg?logo=azuredevops)](https://dev.azure.com/markcampanelli/markcampanelli/_build/latest?definitionId=1&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/pvfit/badge/?version=latest)](https://pvfit.readthedocs.io/en/latest/?badge=latest)

## Up and Running in 5 Minutes

`pvfit` requires [Python 3.6+](https://www.python.org/),
[numpy](https://www.numpy.org/), and [scipy](https://www.scipy.org/). It is
tested on recent versions of Ubuntu, macOS, and Windows.

### Download and Install Package (non-editable mode)

This package will not be available on [PyPI](https://pypi.org/) until the application programming interface (API) is
deemed stable and sufficiently tested and documented. Meanwhile, install the latest code directly from the GitHub repo
using `pip`—
```terminal
pip install git+https://github.com/markcampanelli/pvfit
```
NOTE: You may want to install your own optimized versions of [`numpy`](https://www.numpy.org/) and
[`scipy`](https://www.scipy.org/) (e.g., [conda](https://docs.conda.io/en/latest/)), otherwise this setup will grab the
default versions from [PyPI](https://pypi.org/).

Verify your installation—
```terminal
python -c "from pkg_resources import get_distribution; import pvfit; print(get_distribution('pvfit').version)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

Stay up to date with code changes using—
```terminal
pip install --upgrade git+https://github.com/markcampanelli/pvfit
```

## So What Can PVfit Do for Me?

PVfit is currently restricted to direct-current (DC) PV performance measurement and modeling. See the README's for
individual subpackages to get started with specific functionalities.

TODO: List subpackages with links to READMEs.

TODO: Describe association with Intelligent Measurement Systems LLC and [https://pvfit.app](https://pvfit.app).

## Developer Notes

### Download and Install Package with Developer and Testing Dependencies

Clone this repo using your preferred git method, and go to the repo's root directory.

Install `pvfit` in editable (development) mode, including the `sphinx`, `pytest` packages, with `pip`—
```terminal
pip install -e .[dev,test]
```

Verify your installation—
```terminal
python -c "from pkg_resources import get_distribution; import pvfit; print(get_distribution('pvfit').version)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

### Run Tests with Coverage Locally

From the [root](.) directory—
```terminal
pytest --doctest-modules --cov=pvfit --cov-report=html
```
and the root of the generated coverage report is at `htmlcov/index.html`. 

### Build Documentation Locally

From the [docs](docs) directory—
```terminal
sphinx-apidoc -f -o . ../pvfit ../*_test.py
```
then—
```terminal
build html
```
and the root of the generated documentation is at `docs/_build/html/pvfit.html`. 

### Dependencies

Currently, [`numpy`](https://www.numpy.org/) and [`scipy`](https://www.scipy.org/) are the only runtime dependencies
(minium versions TBD). In order to ensure a straightforward, consistent, and well-tested API, the decision has been made
to avoid any dependecy on [`pandas`](https://pandas.pydata.org/). However, a design goal is for straightforward
integration with consumers that use `pandas`, e.g., integrating computations with
[Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) and
[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) objects. To avoid
bloat, we also avoid dependency on plotting libraries such as [`matplotlib`](https://matplotlib.org/). Any new
dependencies or version ranges should be appropriately recorded in [setup.py](setup.py).

### Coding Requirements and Style

- [Type hints](https://docs.python.org/3/library/typing.html) should be used throughout
- [`flake8`](http://flake8.pycqa.org/en/latest/) formatting with a 120-character line limit for source code files
- A 75-character line limit for all docstrings, following the
[numpydoc docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html)
- An 80-character line limit for example code in `examples` directories
- There is no character line limit for data in Python code, such as in
[data.py](pvfit/measurement/spectral_correction/data.py).
- Unit testing is a must, with naming scheme `module_test.py` to test `module.py` in the same directory
