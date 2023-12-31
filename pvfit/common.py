"""
PVfit: Common items.

Copyright 2023 Intelligent Measurement Systems LLC
"""

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

# Codes for scipy.odr.ODR.run() results.
ODR_SUCCESS_CODES = ("1", "2", "3")  # In ones place.
ODR_NUMERICAL_ERROR_CODE = "6"  # In ten-thousands place.
ODR_NOT_FULL_RANK_ERROR_CODE = "2"  # In tens place.
