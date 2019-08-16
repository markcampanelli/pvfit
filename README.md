# pvfit-m
Computes the spectral mismatch correction factor (M), e.g., for photovoltaic (PV) applications.

## Up and Running in 5 Minutes
This package requires [Python 3.6+](https://www.python.org/) and [numpy](https://www.numpy.org/).

### Download and Install Package (non-editable mode)
This package will not be available on [PyPI](https://pypi.org/) until the application programming interface (API) is
deemed stable and sufficiently tested and documented. Meanwhile, install the latest code directly from the GitHub repo using `pip`—
```terminal
pip install git+https://github.com/markcampanelli/pvfit-m
```
NOTE: You may want to install your own optimized version of [`numpy`](https://www.numpy.org/) first, otherwise this will grab the latest version from [here](https://pypi.org/project/numpy/).

Verify your installation—
```terminal
python -c "import pvfit_m; print(pvfit_m.__version__)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

### Load Example Data and Compute a Spectral Mismatch Correction Factor (M)

The `pvfit-m` package comes with some example data, with special thanks to [Behrang Hamadani at NIST](https://www.nist.gov/people/behrang-hamadani) :). Get started
using this example data by reading through and executing the script
[examples/getting_started.py](examples/getting_started.py) from within the `examples` directory—
```terminal
python getting_started.py
```
which should ultimately print out—
```terminal
M = 0.9982571553509605
```

## How M Is Computed
_M_ is computed with the function, `pvfit_m.api.compute_m()`, using a piecewise linear interpolation between data points
for each curve to fully define each function on its finite interval domain of definition in the first quadrant. The
integration of the product of two piecewise linear functions is done using `pvfit_m.api.inner_product()` over the common
interval domain of definition of the two curves. This is effeciently accomplished by a closed-form summation formula
involving the piecewise-quadratic anti-derivative on each sub-interval of the combined partition of the common domain
interval. It is currently assumed that one/both of the curves is sufficiently close to zero at each end of their common
interval domain of definition in order to produce an accurate computation.

## Future Plans

- Enable vectorized computations of _M_, e.g., for time-series applications such as
[captest](https://github.com/pvcaptest/pvcaptest/)
- Add bad computation detectors/warnings
- Publish on [PyPI](https://pypi.org/) (this repo may first be subsumed as a sub-package by a larger PV-related open
source Python repo)
- Implement different algorithms as requested/contributed by the community

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

Currently, [`numpy`](https://www.numpy.org/) is the only runtime dependency (minium version TBD). In order to ensure a
well-tested, consistent, and straightforward API, a decision has been made to avoid any dependecy on
[`pandas`](https://pandas.pydata.org/). However, a design goal is for straightforward integration with consumers that
use `pandas`, e.g., time-series computations. To avoid bloat, we also avoid dependency on plotting libraries  such as `matplotlib`. We anticipate the likely addition
of [`scipy`](https://www.scipy.org/) as a dependency as features are added. Any new dependencies or version ranges
should be appropriately recorded in [setup.py](setup.py).

### Coding Requirements and Style

- [Type hints](https://docs.python.org/3/library/typing.html) should be used throughout
- [`flake8`](http://flake8.pycqa.org/en/latest/) formatting with a 120 character line limit for source code files
- Unit testing is a must (coverage in CI to be added)
- An 80 character line limit for example code in the [examples](examples) directory
