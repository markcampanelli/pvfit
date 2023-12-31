"""
PVfit: Common items for DC modeling.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
import scipy.constants


from pvfit.common import k_B_J_per_K, q_C
from pvfit.types import FloatArray, FloatBroadcastable, IntBroadcastable

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
MATERIALS = {
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

# Limits on ideality factor on first diode.
N_1_IC_MIN = 1.0
N_1_IC_MAX = 2.0


def get_scaled_thermal_voltage(
    N_s: IntBroadcastable, T_degC: FloatBroadcastable
) -> FloatArray:
    """
    Compute thermal voltage [V], scaled by the number of cells in series in each
    parallel string.
    """
    return numpy.array(
        N_s
        * (
            k_B_J_per_K
            * scipy.constants.convert_temperature(T_degC, "Celsius", "Kelvin")
            / q_C
        )
    )
