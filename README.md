# pvfit-m

Computes the spectral mismatch correction factor (M), e.g., for photovoltaic (PV) applications.

## Up and Running in 5 Minutes

This package requires [Python 3.6+](https://www.python.org/), [numpy](https://www.numpy.org/), and
[scipy](https://www.scipy.org/).

### Download and Install Package (non-editable mode)

This package will not be available on [PyPI](https://pypi.org/) until the application programming interface (API) is
deemed stable and sufficiently tested and documented. Meanwhile, install the latest code directly from the GitHub repo
using `pip`—
```terminal
pip install git+https://github.com/markcampanelli/pvfit-m
```
NOTE: You may want to install your own optimized versions of [`numpy`](https://www.numpy.org/) and
[`scipy`](https://www.scipy.org/), otherwise this setup will grab the default versions from [PyPI](https://pypi.org/).

Verify your installation—
```terminal
python -c "import pvfit_m; print(pvfit_m.__version__)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

Stay up to date with code changes using—
```terminal
pip install --upgrade git+https://github.com/markcampanelli/pvfit-m
```

### Load Example Data and Compute a Spectral Mismatch Correction Factor (M)

The `pvfit-m` package comes with some example data, with special thanks to
[Behrang Hamadani at NIST](https://www.nist.gov/people/behrang-hamadani) :). Get started using this example data by
reading through and executing the script [examples/getting_started.py](examples/getting_started.py) from within the
`examples` directory—
```terminal
python getting_started.py
```
which should ultimately print out—
```terminal
M = 0.9982571553509605
```

## How M Is Computed

_M_ is computed with the function, [`pvfit_m.api.m()`](pvfit_m/api.py), using a piecewise linear interpolation
between data points for each curve to fully define each function on its finite interval domain of definition in the
first quadrant. The integration of the product of two piecewise linear functions is done using
[`pvfit_m.api.inner_product()`](pvfit_m/api.py) over the common interval domain of definition of the two curves. This is effeciently accomplished by a closed-form summation formula involving the piecewise-quadratic anti-derivative on each sub-interval of the combined partition of the common domain interval. It is currently assumed that one/both of the
curves is sufficiently close to zero at each end of their common interval domain of definition in order to produce an
accurate computation. Vectorized computations are possible on mutil-curve data arrays, so long as the wavelength
domain of each multi-curve is a one-dimensional constant. The last dimension of the multi-curve's data array must always
index wavelength to allow proper [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Future Plans

- Add bad computation detectors/warnings
- Publish on [PyPI](https://pypi.org/) (this repo may first be subsumed as a sub-package by a larger PV-related open
source Python repo)
- Implement different algorithms as requested/contributed by the community, such as interpolation of a set of
temperature-dependent quantum efficiency curves

## Developer Notes

### Download and Install Package with Developer and Testing Dependencies

Clone this repo using your preferred git method, and go to the repo's root directory.

Install `pvfit-m` in editable (development) mode, including the `sphinx`, `pytest` packages, with `pip`—
```terminal
pip install -e .[dev,test]
```
NOTE: Documentation generation using `sphinx` is not yet implemented.

Verify your installation—
```terminal
python -c "import pvfit_m; print(pvfit_m.__version__)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

### Run Tests

From the root directory—
```terminal
pytest
```

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
- [`flake8`](http://flake8.pycqa.org/en/latest/) formatting with a 120 character line limit for source code files
- An 80 character line limit for example code in the [examples](examples) directory
- There is no character line limit for data in Python code, such as in [data.py](pvfit_m/data.py).
- Unit testing is a must (coverage in CI to be added)
