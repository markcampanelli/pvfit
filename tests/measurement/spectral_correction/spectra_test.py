"""
PVfit testing: Spectra, esp. standard ones.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from pvfit.measurement.spectral_correction.spectra import (
    E_G173_hemispherical_tilted,
    E_60904_3_hemispherical_tilted,
)

# TODO Check E_G173_hemispherical_tilted.G_total_W_per_m2 == 1000.37 (& formally define 1000.37?)

# TODO Check E_60904_3_hemispherical_tilted.G_total_W_per_m2 == E_hemispherical_tilted_W_per_m2_stc

E_60904_3_hemispherical_tilted.E_total_W_per_m2
E_60904_3_hemispherical_tilted.E_tail_W_per_m2

# TODO Check E_60904_3_hemispherical_tilted on main interval == 997.47 (& formally define 997.47?)

E_60904_3_hemispherical_tilted.get_E_total_subinterval_W_per_m2(
    lambda_min_nm=E_60904_3_hemispherical_tilted.lambda_nm[0],
    lambda_max_nm=E_60904_3_hemispherical_tilted.lambda_nm[-1],
)
