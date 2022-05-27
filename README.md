**NOTE:** The `pvfit-m` repo now redirects here, with its contents in
[this subpackage](pvfit/measurement/spectral_correction).

# pvfit

**PVfit: Photovoltaic (PV) Device Performance Measurement and Modeling**

**IMPORTANT:** This code is pre-release, and so the code organiztion and Application Programming Interface (API) should
be expected to change without warning.

![CI](https://github.com/markcampanelli/pvfit/actions/workflows/ci.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pvfit/badge/?version=latest)](https://pvfit.readthedocs.io/en/latest/?badge=latest)


## So What Can PVfit Do for Me?

PVfit is currently restricted to direct-current (DC) PV performance measurement and modeling. Following the standardized
technical approach of most accredited PV calibration laboratories for measuring I-V curves using PV reference devices,
PVfit makes considerable use of the effective irradiance ratio (F = Isc / Isc0 = M * Isc,ref / Isc0,ref) to quantify the
*effective* irradiance on a PV device, in contrast to the common use of MET-station data
([poster](https://pvpmc.sandia.gov/download/7302/)). See [this paper](https://doi.org/10.1002/ese3.190) for a more
detailed introduction and/or email [Mark Campanelli](mailto:mark.campanelli@gmail.com) to be added to the
[PVfit Slack channel](https://pvfit.slack.com), where you can chat realtime about your quesitons. This open-source code
supports and complements a closed-source model calibration service (e.g., single-diode model parameter fitting from I-V
curve data) available at [https://pvfit.app](https://pvfit.app) and via a REST API.

See the README's for individual subpackages to get started with specific functionalities—

- [Measurement](pvfit/measurement)
  - [Spectral Mismatch Correction Factor M](pvfit/measurement/spectral_correction)
  - Short-Circuit Current Calibration Using Absolute Spectral Response (FUTURE)
- [Modeling](pvfit/modeling)
  - [Single Diode](pvfit/modeling/single_diode)
      - [Equation](pvfit/modeling/single_diode/equation.py)
      - [Model](pvfit/modeling/single_diode/model.py)

## Up and Running in 5 Minutes

`pvfit` minimally requires [python>=3.8,<3.11](https://www.python.org/) with [numpy](https://numpy.org/) and
[scipy](https://www.scipy.org/). It is tested on recent versions of Ubuntu, macOS, and Windows.

### Download, Install, and Verify Package (non-editable mode)

This package will not be available on [PyPI](https://pypi.org/) until the application programming interface (API) is
deemed stable and sufficiently tested and documented. Meanwhile, install the latest code directly from the GitHub repo
using `pip`—
```terminal
pip install git+https://github.com/markcampanelli/pvfit#egg=pvfit[demo]
```
NOTES:
- You may want to install your own optimized versions of [`numpy`](https://www.numpy.org/) and
[`scipy`](https://www.scipy.org/) (e.g., using [conda](https://docs.conda.io/en/latest/)), otherwise this setup will
grab the default versions from [PyPI](https://pypi.org/).
- The `demo` option adds the [matplotlib](https://matplotlib.org/), [pandas](https://pandas.pydata.org/), and
[requests](https://2.python-requests.org/en/master/) packages in order to run all the provided demonstrations in the
`demos` directories.

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
pip install --upgrade git+https://github.com/markcampanelli/pvfit#egg=pvfit[demo]
```

## About the Maintainer

The maintainer of this code is [Mark Campanelli](https://www.linkedin.com/in/markcampanelli/), the proprietor of
[Intelligent Measurement Systems LLC (IMS)](https://intelligentmeasurementsystems.com), in Bozeman, MT, USA. Your
[suggestions/bug reports](https://github.com/markcampanelli/pvfit/issues),
[questions/discussions](https://github.com/markcampanelli/pvfit/discussions), and
[contributions](https://github.com/markcampanelli/pvfit/pulls) are welcome.

## Developer Notes

### Download, Install, and Verify Package with Developer and Testing Dependencies (editable mode)

Clone this repo using your preferred git method, and go to the repo's root directory.

Install `pvfit` in editable (development) mode, including the `pytest` and `sphinx` packages, with `pip`—
```terminal
pip install -e .[demo,dev,docs,test]
```
This also installs the libraries needed to develop the code demonstrations.

Verify your installation—
```terminal
python -c "from pkg_resources import get_distribution; import pvfit; print(get_distribution('pvfit').version)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

### Run Tests with Coverage Locally

From the repo's `pvfit` subdirectory—
```terminal
pytest --doctest-modules --cov=pvfit --cov-report=html
```
and the root of the generated coverage report (uncommitted) is at `pvfit/htmlcov/index.html`. 

### Build Documentation Locally

From the [docs](docs) directory—
```terminal
sphinx-apidoc -f -o . ../pvfit ../*_test.py
```
then—
```terminal
make html
```
and the root of the generated documentation (uncommitted) is at `docs/_build/html/pvfit.html`. 

### Dependencies

Currently, [`numpy`](https://www.numpy.org/) and [`scipy`](https://www.scipy.org/) are the only runtime dependencies. In
order to ensure a straightforward, consistent, and well-tested API, the decision has been made to avoid any dependecy on
[`pandas`](https://pandas.pydata.org/). However, a design goal is for straightforward integration with consumers that
use `pandas`, e.g., integrating computations with
[Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) and
[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) objects. To avoid
bloat, we also avoid dependency on plotting libraries such as [`matplotlib`](https://matplotlib.org/). Any new
dependencies or version ranges should be appropriately recorded in [setup.cfg](setup.cfg).

### Coding Requirements and Style

- [Type hints](https://docs.python.org/3/library/typing.html) should be used throughout.
- [`flake8`](http://flake8.pycqa.org/en/latest/) formatting with a 120-character line limit for source code files.
- A 75-character line limit for all docstrings, following the
[numpydoc docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html).
- An 80-character line limit for demonstration code in `demos` directories.
- There is no character line limit for data in Python code, such as in
[data.py](pvfit/measurement/spectral_correction/data.py).
- Unit testing is a must, with naming scheme `module_test.py` to test `module.py` in the same directory.
