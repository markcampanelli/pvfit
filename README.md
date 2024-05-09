# pvfit

**PVfit: Photovoltaic (PV) Device Performance Measurement and Modeling**

**IMPORTANT:** This code is pre-release, and so the code organiztion and Application
Programming Interface (API) should be expected to change with minimal warning.

**NOTICE:** We are in the process of open-sourcing the single-diode equation (SDE) and 
single-diode model (SDM) fitting algorithms (🎉), and thus moving the related code here.
The move is reasonably complete, but the code for SDM fitting using full I-V curves has
not yet been ported. Likewise, the documentation badly needs updating, so for now we
refer users to the `demos/getting_started.py` modules in the various subpackages.

![CI](https://github.com/markcampanelli/pvfit/actions/workflows/ci.yml/badge.svg)
<!-- [![Documentation Status](https://readthedocs.org/projects/pvfit/badge/?version=latest)](https://pvfit.readthedocs.io/en/latest/?badge=latest) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## So What Can PVfit Do for Me?

PVfit is currently focused on direct-current (DC) PV module performance measurement and
modeling. Following the standardized technical approach of most accredited PV
calibration laboratories for measuring current-voltage (I-V) curves using reference
devices, PVfit formulates it's DC performance models in terms of the effective
irradiance ratio (e.g., F = Isc / Isc0 = M * Isc,ref / Isc0,ref) to quantify the
*effective* irradiance on a PV device. This has benefits for both model calibration and
performane simulation. PVfit provides extensions for working with common
irradiance-based MET-station data, and PVfit also supports inference of
effective-irradiance ratio and cell temperature directly from I-V measurements, see
([poster](https://pvpmc.sandia.gov/download/3924/?tmstv=1715255668)). See
[this paper](https://doi.org/10.1002/ese3.190) for a more detailed introduction. Email
[Mark Campanelli](mailto:mark.campanelli@gmail.com) for support, etc. See the
`demos/getting_started.py` in individual subpackages to get started with specific
functionalities—

- [Measurement](pvfit/measurement)
  - [Current-Voltage (I-V) Data/Curves](pvfit/measurement/iv)
    - [I-V Data/Curve Types](pvfit/measurement/iv/types.py)
    - [I-V Data/Curve Computations](pvfit/measurement/iv/computation.py)
  - [Spectral Mismatch Correction](pvfit/measurement/spectral_correction)
    - [Quantum Efficiency/Spectral Response and Spectrum Types](pvfit/measurement/spectral_correction/types.py)
    - [Spectral Mismatch Correction Computations](pvfit/measurement/spectral_correction/computation.py)
    - Short-Circuit Current Calibration Using Absolute Spectral Response (FUTURE)
- [Modeling](pvfit/modeling)
  - [Direct Current (DC)](pvfit/modeling/dc)
    - [Single Diode](pvfit/modeling/dc/single_diode)
      - [Equation (single operating condition)](pvfit/modeling/dc/single_diode/equation)
        - [Simple Formulation](pvfit/modeling/dc/single_diode/equation/simple)
          - [Parameter Fitting](pvfit/modeling/dc/single_diode/equation/simple/inference_iv_curve.py)
          - [Simulation](pvfit/modeling/dc/single_diode/equation/simple/simulation.py)
      - [Model (variable operating conditions)](pvfit/modeling/dc/single_diode/model)
        - [Simple Formulation](pvfit/modeling/dc/single_diode/model/simple)
          - [Parameter Fitting to IEC 61853-1 Performance Matrices](pvfit/modeling/dc/single_diode/model/simple/inference_matrix.py)
          - [Parameter Fitting to Module Specification Datasheets](pvfit/modeling/dc/single_diode/model/simple/inference_spec_sheet.py)
          - [Inference of Operating Conditions from I-V Data](pvfit/modeling/dc/single_diode/model/simple/inference_oc.py)
          - [Auxiliary Equations (for simulation via simple SDE)](pvfit/modeling/dc/single_diode/model/simple/auxiliary_equations.py)
        - [Photoconductive-Shunt Formulation](pvfit/modeling/dc/single_diode/model/photoconductive_shunt)
          - [Parameter Fitting to IEC 61853-1 Performance Matrices](pvfit/modeling/dc/single_diode/model/photoconductive_shunt/inference_matrix.py)
          - [Auxiliary Equations (for simulation via simple SDE)](pvfit/modeling/dc/single_diode/model/photoconductive_shunt/auxiliary_equations.py)
  - [Alternating Current (AC)](pvfit/modeling/ac)
    - [Sandia Inverter Performance Model](pvfit/modeling/ac/sipm) - Very experimental code here
      - [Parameter Fitting](pvfit/modeling/ac/sipm/inference.py)
      - [Simulation](pvfit/modeling/ac/sipm/simulation.py)

We still need to improve test coverage for certain subpackages, esp.the simple SDM.

## Up and Running in 5 Minutes

`pvfit` minimally requires [python>=3.10,<3.13](https://www.python.org/) with
[numpy](https://numpy.org/) and [scipy](https://www.scipy.org/). It is tested with
CPython on recent versions of Ubuntu, macOS, and Windows. We suggest using a suitable
Python virtual environment that provides [pip](https://pypi.org/project/pip/).

### Download, Install, and Verify Package (non-editable mode)

This package is available at [PyPI](https://pypi.org/), but it is still pre-v1. With
sufficiently recent versions of `pip` and `setuptools`, install `pvfit` with the extra
packages needed for the demos using—
```terminal
python -m pip install --upgrade pip setuptools
python -m pip install pvfit[demo]
```

Verify your installation—
```terminal
python -c "from pvfit import __version__; print(__version__)"
```
which should print something similar to—
```terminal
0.0.1
```

You should now be able to explore PVfit's functionality with the `getting_started.py`
modules in the various `demos` directories of the various subpackages.

NOTES:
- You may want to install your own optimized versions of
[`numpy`](https://www.numpy.org/) and [`scipy`](https://www.scipy.org/) (e.g., using
[conda](https://docs.conda.io/en/latest/)), otherwise this setup will grab the default
versions from [PyPI](https://pypi.org/).
- The `demo` option adds the [matplotlib](https://matplotlib.org/),
[pandas](https://pandas.pydata.org/), and
[pvlib-python](https://pvlib-python.readthedocs.io/) packages in order to run all the
provided demonstrations in the `demos` directories.
- You can also run `pvfit` on the bleeding edge. If you have `git` installed, then
install from the `main` branch using—
```terminal 
python -m pip install --upgrade "pvfit[demo] @ git+https://github.com/markcampanelli/pvfit"
```

## Developer Notes

### Download, Install, and Verify Package with Developer and Testing Dependencies (editable mode)

Clone the repo at https://github.com/markcampanelli/pvfit using your preferred git
method, and go to the repo's root directory.

Install `pvfit` with all extras in editable (development) mode with `pip`—
```terminal
python -m pip install --upgrade pip setuptools
python -m pip install -e .[demo,dev,docs,test]
python -m pip install --progress-bar off "ivcurves @ git+https://github.com/cwhanse/ivcurves@7ae47284b23cfff167932b8cccae53c10ebf9bf9"
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
well-tested API, the decision has been made to avoid any dependecy of the core code on
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
