import scipy.constants

# Some useful physical constants (CODATA 2018:
# https://physics.nist.gov/cuu/Constants/).

# elementary charge, 1.602176634e-19 C.
q_C = scipy.constants.value("elementary charge")

# Boltzmann constant, 1.380649e-23 J/K.
k_B_J_per_K = scipy.constants.value("Boltzmann constant")

# Boltzmann constant, 8.617333262e-05 eV/K.
k_B_eV_per_K = scipy.constants.value("Boltzmann constant in eV/K")

# Speed of light, 299792458.0 m/s
c_m_per_s = scipy.constants.c

# Plank's constant, 6.62607015e-34 J s
h_J_s = scipy.constants.h

# Reference values, including Standard Test Condition (STC).

# Temperature.
# Absolute zero in degrees Celsius.
T_degC_abs_zero = scipy.constants.convert_temperature(0, "Kelvin", "Celsius")
# STC temperature in degrees Celsius.
T_degC_stc = 25.0
# STC temperature in Kelvin.
T_K_stc = scipy.constants.convert_temperature(T_degC_stc, "Celsius", "Kelvin")

# Total irradiance.
# Hemispherical irradiance at STC (includes specified sun orientation,
# plane orientation, spectrum, etc.).
G_hemi_W_per_m2_stc = 1000.0

# Materials.
materials = {
    "CIGS": {  # Copper Indium Gallium Selenide (CIGS).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.15,
    },
    "CIS": {  # Copper Indium diSelenide (CIS).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.010,
    },
    "CdTe": {  # Cadmium Telluride (CdTe).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.475,
    },
    "GaAs": {  # Gallium Arsenide (GaAs).
        # Band gap at 300 K, Kittel, C., Intro. to Solid State
        # Physics, 6th ed. 1986, p 185.
        "E_g_eV_stc": 1.43,
    },
    "mono-Si": {  # Mono-crystalline Silicon (mono-Si).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.121,
    },
    "multi-Si": {  # Multi-crystalline Silicon (multi-Si).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.121,
    },
    "poly-Si": {  # Poly-crystalline Silicon (poly-Si).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.121,
    },
    "x-Si": {  # Crystalline Silicon (x-Si).
        # Band gap at STC, from De Soto et al. 2006.
        "E_g_eV_stc": 1.121,
    },
}

# Assertions

# Absolute tolerance for numpy.testing.assert_allclose when testing
# current sum is sufficnently near zero.
I_sum_A_atol = 1e-12
