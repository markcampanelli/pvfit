import inspect

import scipy.constants
from scipy.optimize import newton
from scipy.optimize.optimize import _minimize_scalar_bounded

# Some useful physical constants (CODATA 2018: https://physics.nist.gov/cuu/Constants/).

# elementary charge, 1.602176634e-19 C.
q_C = scipy.constants.value('elementary charge')

# Boltzmann constant, 1.380649e-23 J/K.
k_B_J_per_K = scipy.constants.value('Boltzmann constant')

# Boltzmann constant, 8.617333262e-05 eV/K.
k_B_eV_per_K = scipy.constants.value('Boltzmann constant in eV/K')

# Speed of light, 299792458.0 m/s
c_m_per_s = scipy.constants.c

# Plank's constant, 6.62607015e-34 J s
h_J_s = scipy.constants.h

# Reference values, including Standard Test Conditions (STC).

# Temperature.
# Absolute zero in degrees Celsius.
T_abs_zero_degC = scipy.constants.convert_temperature(0, 'Kelvin', 'Celsius')
# STC temperature in degrees Celsius.
T_stc_degC = 25.
# STC temperature in Kelvin.
T_stc_K = scipy.constants.convert_temperature(T_stc_degC, 'Celsius', 'Kelvin')

# Total irradiance.
# Hemispherical irradiance at STC (includes specified sun orientation, plane orientation, spectrum, etc.).
G_hemi_stc_W_per_m2 = 1000.

# Materials.
materials = \
    {
     'CIGS': {  # Copper Indium Gallium Selenide (CIGS).
              # Band gap at STC, from De Soto et al. 2006.
              'E_g_stc_eV': 1.15,
             },
     'CIS':  {  # Copper Indium diSelenide (CIS).
              # Band gap at STC, from De Soto et al. 2006.
              'E_g_stc_eV': 1.010,
             },
     'CdTe': {  # Cadmium Telluride (CdTe).
              # Band gap at STC, from De Soto et al. 2006.
              'E_g_stc_eV': 1.475,
             },
     'GaAs': {  # Gallium Arsenide (GaAs).
              # Band gap at 300 K, Kittel, C., Intro. to Solid State Physics, 6th ed. 1986, p 185.
              'E_g_stc_eV': 1.43,
             },
     'x-Si': {  # Mono-/multi-crystalline Silicon (x-Si).
              # Band gap at STC, from De Soto et al. 2006.
              'E_g_stc_eV': 1.121,
             }
    }

# Solvers

# Default options taken from current version of scipy.optimize.newton.
newton_options_default = {'maxiter': inspect.signature(newton).parameters['maxiter'].default,
                          'tol': inspect.signature(newton).parameters['tol'].default}

# Default options taken from current version of scipy.optimize.minimize_scalar, for 'bounded' mode.
minimize_scalar_bounded_options_default = {
    'maxiter': inspect.signature(_minimize_scalar_bounded).parameters['maxiter'].default,
    'xatol': inspect.signature(_minimize_scalar_bounded).parameters['xatol'].default}
