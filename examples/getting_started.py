# Python 3.6-7
import numpy

# This imports `pvfit_m.api` and `pvfit_m.data`.
import pvfit_m

print(f"pvfit_m version {pvfit_m.__version__}")

"""
pvfit_m has several data classes that wrap underlying numpy.ndarray data
represting the various curves appearing in the four integrals in the
formula for M.

See, for example, equation (5) in
https://onlinelibrary.wiley.com/doi/full/10.1002/ese3.190
"""

"""
pvfit_m.data has already created several useful example data objects.
For example, assuming one has loaded wavelength and spectral responsivity
data as 1D numpy arrays for the NIST test device (a x-Si PV cell)...
"""
# lambda_nm = numpy.array([...])
lambda_nm = pvfit_m.data.lambda_nm_td_NIST
# sr_A_per_W = numpy.array([...])
sr_A_per_W = pvfit_m.data.sr_A_per_W_td_NIST
# ...one then creates a SpectralResponsivity object—
sr = pvfit_m.api.SpectralResponsivity(lambda_nm=lambda_nm,
                                      sr_A_per_W=sr_A_per_W)
# This gets spectral responsivity [A/W] at each wavelength [nm] from the
# underlying numpy.ndarray and show them stacked together as rows.
print(f"sr = {numpy.vstack((sr.lambda_nm, sr.sr_A_per_W))}")

# Instead of re-creating all the necessary data objects for computing M,
# we use ones already made for demonstration purposes.

"""
Load spectral responsivity of test device (here, a Si PV cell at 25degC) as a
SpectralResponsivity object from wavelength [nm] and spectral responsivity
[A/W] data (each an underlying numpy.ndarray).
"""
sr_td = pvfit_m.data.sr_td_NIST

"""
Load spectral irradiance illuminating test device (here, a Xenon solar
simulator) as a SpectralIrradiance object containing wavelength [nm] and
spectral irradiance [W/m2/nm] data (each an underlying numpy.ndarray).
"""
si_td = pvfit_m.data.si_sim_NIST

"""
Load spectral responsivity of reference device (here, a Si PV cell at at
25degC) as a SpectralResponsivity object containing wavelength [nm] and
spectral responsivity [A/W] data (each an underlying numpy.ndarray).
"""
sr_rd = pvfit_m.data.sr_rd_NIST

"""
Load spectral irradiance illuminating the reference device (here, a Xenon solar
simulator) as a SpectralIrradiance object containing wavelength [nm] and
spectral irradiance [W/m2/nm] data (each an underlying numpy.ndarray).
"""
si_rd = pvfit_m.data.si_sim_NIST


"""
Load reference spectral irradiance (here, ASTM G173 Global Tilt) as a
SpectralIrradiance object containing wavelength [nm] and spectral irradiance
[W/m2/nm] data (each an underlying numpy.ndarray).
"""
si_0 = pvfit_m.data.si_G173_global_tilt

"""
Compute spectral mismatch correction factor M.
This assumes a constant scaling between the spectral responsivity and
spectral response of each device. These two scalings are assumed to cancel
out between the numerator and denominator in the formula for M. Likewise,
the spectral irradiance curves need only be relative (not absolute) curves.
"""
m = pvfit_m.api.m(sr_td=sr_td, si_td=si_td, sr_rd=sr_rd, si_rd=si_rd,
                  si_0=si_0)
print('M = {}'.format(m))
