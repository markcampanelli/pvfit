# pvfit-m
Computes the spectral mismatch correction factor (M), e.g., for photovoltaic (PV) applications.

## Up and Running in 5 Minutes
This package requires [Python 3.6+](https://www.python.org/) and [numpy](https://www.numpy.org/).

### Download package
This package will not be available on [PyPI](https://pypi.org/) until the application programming interface (API) is deemed stable and sufficiently tested and documented. Meanwhile, install the latest code directly from the GitHub repo using `pip`—
```terminal
pip install git+https://github.com/markcampanelli/pvfit-m
```
or, for editable (development) mode, including the `pytest` package—
```terminal
pip install -e git+https://github.com/markcampanelli/pvfit-m#egg=pvfit-m[dev,test]
```
Verify your installation—
```terminal
python -c "import pvfit_m; print(pvfit_m.__version__)"
```
which should print something similar to—
```terminal
0.1.dev9+gadf7f38.d20190812
```

### Load Example Data and Compute a Spectral Mismatch Correction Factor (M)

The `pvfit-m` package comes with some example data, with special thanks to Behrang Hamadani at NIST :). Execute the commands in the following Python script—
```python
# Python 3.6-7
import numpy

# This imports `pvfit_m.api` and `pvfit_m.data`. 
import pvfit_m

print(f"pvfit_m version {pvfit_m.__version__}")

# pvfit_m has several data classes that wrap underlying numpy.ndarray data represting the various curves appearing in the four integrals in the formula for M.
# See, for example, equation (5) in https://onlinelibrary.wiley.com/doi/full/10.1002/ese3.190

# pvfit_m.data has already created several useful example data objects.
# For example, assuming one has loaded wavelength and spectral responsivity data as 1D numpy arrays for the NIST test device (a x-Si PV cell)...
# lambda_nm = numpy.array([...])
lambda_nm = pvfit_m.data.lambda_nm_td_NIST
# sr_A_per_W = numpy.array([...])
sr_A_per_W = pvfit_m.data.sr_A_per_W_td_NIST
# ...one then creates a SpectralResponsivity object—
sr = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm, sr_A_per_W=sr_A_per_W)
# This gets spectral responsivity [A/W] at each wavelength [nm] from the underlying numpy.ndarray and show them stacked together as rows.
print(f"sr = {numpy.vstack((sr.lambda_nm, sr.sr_A_per_W))}")

# Instead of re-creating all the necessary data objects for computing M, we use ones already made for demonstration purposes.

# Load spectral responsivity of test device (here, a Si PV cell at 25degC) as a SpectralResponsivity object from wavelength [nm] and spectral responsivity [A/W] data (each an underlying numpy.ndarray).
sr_td = pvfit_m.data.sr_td_NIST
# Load spectral irradiance illuminating test device (here, a Xenon solar simulator) as a SpectralIrradiance object containing wavelength [nm] and spectral irradiance [W/m2/nm] data (each an underlying numpy.ndarray).
si_td = pvfit_m.data.si_sim_NIST
# Load spectral responsivity of reference device (here, a Si PV cell at at 25degC) as a SpectralResponsivity object containing wavelength [nm] and spectral responsivity [A/W] data (each an underlying numpy.ndarray).
sr_rd = pvfit_m.data.sr_rd_NIST
# Load spectral irradiance illuminating the reference device (here, a Xenon solar simulator) as a SpectralIrradiance object containing wavelength [nm] and spectral irradiance [W/m2/nm] data (each an underlying numpy.ndarray).
si_rd = pvfit_m.data.si_sim_NIST
# Load reference spectral irradiance (here, ASTM G173 Global Tilt) as a SpectralIrradiance object containing wavelength [nm] and spectral irradiance [W/m2/nm] data (each an underlying numpy.ndarray).
si_0 = pvfit_m.data.si_G173_global_tilt

# Compute spectral mismatch correction factor M.
# This assumes a constant scaling between the spectral responsivity and spectral response of each device. These two scalings are assumed to cancel out between the numerator and denominator in the formula for M. Likewise, the spectral irradiance curves need only be relative (not absolute) curves.
print(f"M = {pvfit_m.api.compute_m(sr_td=sr_td, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd, si_0=si_0)}")
```
which should ultimately print—
```terminal
M = 0.9982571553509605
```

## How M Is Computed
`pvfit_m.api.compute_m()` uses a piecewise linear interpolation between data points for each curve to fully define each function on its finite interval domain of definition in the first quadrant. The integration of the product of the two piecewise linear functions is done over the common interval domain of definition of the two curves which can be effeciently accomplished by a closed-form summation formula involving the piecewise-quadratic anti-derivative on each sub-interval of the combined partition of the common domain interval. It is currently assumed that one/both of the curves is sufficiently close to zero at each end of their common interval domain of definition in order to produce an accurate computation.