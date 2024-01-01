# pvfit

**PVfit: Photovoltaic (PV) Device Performance Measurement and Modeling**

**IMPORTANT:** This code is pre-release, and so the code organiztion and Application
Programming Interface (API) should be expected to change without warning.

**NOTICE:** We are in the process of open-sourcing the single-diode equation (SDE) and 
single-diode model (SDM) fitting algorithms (!), and thus moving the related code here.
The SDE move is reasonably complete, but the SDM move is still under way, and all the
SDM is temporarily unavailable on the `master` branch.

![CI](https://github.com/markcampanelli/pvfit/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pvfit/badge/?version=latest)](https://pvfit.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## So What Can PVfit Do for Me?

PVfit is currently focused on direct-current (DC) PV module performance measurement and
modeling. Following the standardized technical approach of most accredited PV
calibration laboratories for measuring I-V curves using PV reference devices, PVfit
makes considerable use of the effective irradiance ratio
(F = Isc / Isc0 = M * Isc,ref / Isc0,ref) to quantify the *effective* irradiance on a PV
device, in contrast to the common use of MET-station data
([poster](https://pvpmc.sandia.gov/download/7302/)). See
[this paper](https://doi.org/10.1002/ese3.190) for a more detailed introduction. Email
[Mark Campanelli](mailto:mark.campanelli@gmail.com) to be added to the
[PVfit Slack channel](https://pvfit.slack.com), where you can chat realtime about your
quesitons/applications. This open-source code complements a model calibration service
(e.g., single-diode model parameter fitting from I-V curve data) available at
[https://pvfit.app](https://pvfit.app) and via a REST API.

See the `demos/getting_started.py` in individual subpackages to get started with
specific functionalities—

- [Measurement](pvfit/measurement)
  - [Spectral Mismatch Correction](pvfit/measurement/spectral_correction)
    - [Quantum Efficiency/Spectral Response and Spectrum Types](pvfit/measurement/spectral_correction/types.py)
    - [Spectral Mismatch Correction Computations](pvfit/measurement/spectral_correction/computation.py)
    - Short-Circuit Current Calibration Using Absolute Spectral Response (FUTURE)
  - [Current-Voltage (I-V) Data/Curves](pvfit/measurement/iv)
    - [I-V Data/Curve Types](pvfit/measurement/iv/types.py)
    - [I-V Data/Curve Computations](pvfit/measurement/iv/computation.py)
- [Modeling](pvfit/modeling)
  - [Direct Current (DC)](pvfit/modeling/simulation/dc)
    - [Single Diode](pvfit/modeling/simulation/dc/single_diode)
      - [Equation (single operating condition)](pvfit/modeling/simulation/dc/single_diode/equation)
        - [Parameter Fitting](pvfit/modeling/simulation/dc/single_diode/equation/inference.py)
        - [Simulation](pvfit/modeling/simulation/dc/single_diode/equation/simulation.py)
      - [Model (variable operating conditions)](pvfit/modeling/simulation/dc/single_diode/model) (WARNING: Currently missing due to refactor.)
        - [Parameter Fitting to IEC 61853-1 Performance Matrices](pvfit/modeling/simulation/dc/single_diode/model/inference_matrxi.py)
        - [Parameter Fitting to Module Specification Datasheets](pvfit/modeling/simulation/dc/single_diode/model/inference_spec_sheet.py)
        - [Parameter Fitting to I-V Curve Collections](pvfit/modeling/simulation/dc/single_diode/model/inference_iv_curves.py)
        - [Simulation](pvfit/modeling/simulation/dc/single_diode/model/simulation.py)

The above subpackages are reasonably well tested. Certain other subpackages are
considered more experimental.

## Up and Running in 5 Minutes

`pvfit` minimally requires [python>=3.9,<3.13](https://www.python.org/) with
[numpy](https://numpy.org/) and [scipy](https://www.scipy.org/). It is tested with
CPython on recent versions of Ubuntu, macOS, and Windows. We suggest using a suitable
Python virtual environment that provides [pip](https://pypi.org/project/pip/).

### Download, Install, and Verify Package (non-editable mode)

This package will not be available on [PyPI](https://pypi.org/) until the application
programming interface (API) is deemed stable and sufficiently tested and documented.
Meanwhile, install the latest code directly from the GitHub repo using a sufficiently
recent version of `pip` and `setuptools`—
```terminal
python -m pip install --upgrade pip setuptools
python -m pip install git+https://github.com/markcampanelli/pvfit#egg=pvfit[demo]
```
NOTES:
- You may want to install your own optimized versions of
[`numpy`](https://www.numpy.org/) and [`scipy`](https://www.scipy.org/) (e.g., using
[conda](https://docs.conda.io/en/latest/)), otherwise this setup will grab the default
versions from [PyPI](https://pypi.org/).
- The `demo` option adds the [matplotlib](https://matplotlib.org/) package in order to
run all the provided demonstrations in the `demos` directories.

Verify your installation—
```terminal
python -c "from pvfit import __version__; print(__version__)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

Likewise, stay up to date with the latest code changes using—
```terminal
python -m pip install --upgrade git+https://github.com/markcampanelli/pvfit#egg=pvfit[demo]
```

You should now be able to explore PVfit's functionality with the `getting_started.py`
modules in the various `demos` directories of the various subpackages.

## Developer Notes

### Download, Install, and Verify Package with Developer and Testing Dependencies (editable mode)

Clone this repo using your preferred git method, and go to the repo's root directory.

Install `pvfit` with all extras in editable (development) mode with `pip`—
```terminal
python -m pip install --upgrade pip setuptools
python -m pip install -e .[demo,dev,docs,test]
```
This also installs the libraries needed to test, develop the code demonstrations, and
build documentation and source and wheel distributions.

Verify your installation—
```terminal
python -c "from pvfit import __version__; print(__version__)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

Next, make sure that the tests are passing.

### Test with Coverage

From the root directory—
```terminal
python -m pytest --doctest-modules --cov=pvfit --cov-report=html:htmlcov tests
```
The root of the generated coverage report is at `artifacts/test/htmlcov/index.html` (not
committed). 

### Build Documentation

From the [docs](docs) subdirectory—
```terminal
sphinx-apidoc -f -o . ../pvfit ../*_test.py
```
then—
```terminal
make html
```
The root of the generated documentation is at `docs/_build/html/pvfit.html` (not
committed). 

### Distribute, inc. with Nuitka

PEP-517-compliant [build](https://pypa-build.readthedocs.io/en/latest/) is used to
generate distributions using [setuptools](https://setuptools.pypa.io/en/latest/) as the
build backend (specified in `pyproject.toml`). From the repo root, execute--
```terminal
python -m build
```
Pure-Python `*.whl` and `*.tar.gz` files are placed in the `dist` directory (not
committed).

Alternatively, [nuitka](https://nuitka.net/index.html) can be used to transpile the
Python source code into faster-executing, compiled C code with the same Python
interface. With an appropriate setup for Nuitka (compilers, etc.), swap the
`[build-system]` table in the `pyproject.toml`, then--
```terminal
python -m build
```
A platfrom-specific `*.whl` file is placed in the `dist` directory (not committed). The
included Python extension module has the same interface. Users may wish to remove
tests and demos before generating such wheel files.

Finally, the distribution manifests (cf. `MANIFEST.in`) are checked using--
```terminal
python -m check_manifest
```

### Dependencies

Currently, [`numpy`](https://www.numpy.org/) and [`scipy`](https://www.scipy.org/) are
the only runtime dependencies. In order to ensure a straightforward, consistent, and
well-tested API, the decision has been made to avoid any dependecy on
[`pandas`](https://pandas.pydata.org/). However, a design goal is for straightforward
integration with consumers that use `pandas`, e.g., integrating computations with
[Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)
and
[DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
objects. To avoid bloat, we also avoid dependency on plotting libraries such as
[`matplotlib`](https://matplotlib.org/). Any new dependencies or version ranges should
be appropriately recorded in [pyproject.toml](pyproject.toml).

### Coding Requirements and Style

- Unit testing is a must, with a "collocation" scheme, i.e., `module_test.py` to test
`module.py` in the same directory. 100% code coverage is the goal.
- [Type hints](https://docs.python.org/3/library/typing.html) should be used
throughout (WIP).
- [`pylint`](https://pylint.readthedocs.io/en/latest/?badge=latest) is used for linting,
with `black`'s default 88-character line limit (configured in 
[pyproject.toml](pyproject.toml)). Check before committing code using--
```terminal
python -m pylint .
```
Skip troublesome lines (sparingly) with the suffix `# pylint: disable=<code>`.
- [`black`](https://black.readthedocs.io/en/stable/index.html) is used to autoformat
code. Autoformat before committing
code, using--
```terminal
python -m black .
```

## About the Author and Maintainer

The author and maintainer of this code is
[Mark Campanelli](https://www.linkedin.com/in/markcampanelli/), the proprietor of
[Intelligent Measurement Systems LLC (IMS)](https://intelligentmeasurementsystems.com),
in Bozeman, MT, USA. Your
[suggestions/bug reports](https://github.com/markcampanelli/pvfit/issues),
[questions/discussions](https://github.com/markcampanelli/pvfit/discussions), and
[contributions](https://github.com/markcampanelli/pvfit/pulls) are welcome.
