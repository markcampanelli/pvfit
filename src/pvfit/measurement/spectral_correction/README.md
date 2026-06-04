# pvfit.measurement.spectral_correction

A package for spectral mismatch correction factors (M) for PV devices.

## Up and Running in 5 Minutes

First, make sure that you have followed the package setup instructions
[here](../../README.md#Up-and-Running-in-5-Minutes).

### Load Example Data and Compute a Spectral Mismatch Correction Factor (M)

The `pvfit.measurement.spectral_correction` package comes with some example data, with special thanks to
[Behrang Hamadani at NIST](https://www.nist.gov/people/behrang-hamadani) :). Get started using this example data by
reading through and executing the script [getting_started.py](demos/getting_started.py) from within the
[demos](demos) directory—
```terminal
python getting_started.py
```
which should ultimately print out—
```terminal
M = 0.9982571553509605
```

## How M Is Computed

_M_ is computed with the function, [pvfit.measurement.spectral_correction.computation.M](computation.py), using a
piecewise linear interpolation between data points for each curve to fully define each function on its finite interval
domain of definition in the first quadrant. The integration of the product of two piecewise linear functions is done
using [pvfit.measurement.spectral_correction.computation.inner_product](computation.py) over the common interval domain
of definition of the two curves. This is effeciently accomplished by a closed-form summation formula involving the
piecewise-quadratic anti-derivative on each sub-interval of the combined partition of the common domain interval. It is
currently assumed that one/both of the curves is sufficiently close to zero at each end of their common interval domain
of definition in order to produce an accurate computation. Vectorized computations are possible on mutil-curve data
arrays, so long as the wavelength domain of each multi-curve is a one-dimensional constant. The last dimension of the
multi-curve's data array must always index wavelength to allow proper
[broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Future Plans

- Add detectors/warnings for (potentially) bad computations
- Implement different algorithms as requested/contributed by the community, such as interpolation of a set of
temperature-dependent quantum efficiency curves
