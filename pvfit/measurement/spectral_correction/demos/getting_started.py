"""
PVfit: Getting-started demo for spectral corrections.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import pvfit.measurement.spectral_correction.computation as computation
import pvfit.measurement.spectral_correction.demos.data as data

# By convention, variable names for numeric values include the units.

"""
pvfit.measurement.spectral_correction.computation has several data classes
that wrap underlying numpy.ndarray data represting the various curves
appearing in the four integrals in the formula for M. See, for example,
equation (5) in https://doi.org/10.1002/ese3.190.

pvfit.measurement.spectral_correction.data has already created several
useful example data objects. For example, assuming one has loaded
wavelength and spectral responsivity data as 1D numpy arrays for the NIST
test device (a x-Si PV cell)...
"""

# lambda_nm = numpy.array([...])
lambda_nm = data.lambda_nm_TD_NIST
# S_A_per_W = numpy.array([...])
S_A_per_W = data.S_A_per_W_TD_NIST

"""
...one then creates a SpectralResponsivity object—
"""
S = computation.SpectralResponsivity(lambda_nm=lambda_nm, S_A_per_W=S_A_per_W)
"""
This gets spectral responsivity [A/W] at each wavelength [nm] from the
underlying numpy.ndarray and shows the domain and range—
"""
print(f"S.lambda_nm = {S.lambda_nm}")
print(f"S.S_A_per_W = {S.S_A_per_W}")

"""
Instead of re-creating all the necessary data objects for computing M, we
use ones already made for demonstration purposes.

Load spectral responsivity of test device at operating condition and at
reference condition (here, a Si PV cell at 25degC) as a
SpectralResponsivity object containing wavelength [nm] and spectral
responsivity [A/W] data (each an underlying numpy.ndarray).
"""
S_TD_OC = data.S_TD_NIST
S_TD_RC = data.S_TD_NIST

"""
Load spectral irradiance illuminating test device at operating condition
(here, a Xenon solar simulator) as a SpectralIrradiance object containing
wavelength [nm] and spectral irradiance [W/m2/nm] data (each an underlying
numpy.ndarray).
"""
E_TD_OC = data.E_sim_NIST

"""
Load spectral responsivity of reference device at operating condition and
at reference condition (here, a Si PV cell at at 25degC) as a
SpectralResponsivity object containing wavelength [nm] and spectral
responsivity [A/W] data (each an underlying numpy.ndarray).
"""
S_RD_OC = data.S_RD_NIST
S_RD_RC = data.S_RD_NIST

"""
Load spectral irradiance illuminating reference device at operating
condition (here, a Xenon solar simulator) as a SpectralIrradiance object
containing wavelength [nm] and spectral irradiance [W/m2/nm] data (each an
underlying numpy.ndarray).
"""
E_RD_OC = data.E_sim_NIST

"""
Load spectral irradiance illuminating both test device and reference
device at reference condition (here, ASTM G173 Global Tilt) as a
SpectralIrradiance object containing wavelength [nm] and spectral
irradiance [W/m2/nm] data (each an underlying numpy.ndarray).
"""
E_TD_RC = data.E_G173_global_tilt
E_RD_RC = data.E_G173_global_tilt

"""
Compute spectral mismatch correction factor M. See the function docstring
for details.
"""
M = computation.M(
    S_TD_OC=S_TD_OC,
    E_TD_OC=E_TD_OC,
    S_TD_RC=S_TD_RC,
    E_TD_RC=E_TD_RC,
    S_RD_OC=S_RD_OC,
    E_RD_OC=E_RD_OC,
    S_RD_RC=S_RD_RC,
    E_RD_RC=E_RD_RC,
)
print("M = {}".format(M))
