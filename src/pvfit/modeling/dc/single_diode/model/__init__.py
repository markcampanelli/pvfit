"""
PVfit: Single-diode model (SDM).

Currently, SDM simulation is designed to support various auxiliary equations for the SDM
that compute model parameters for the underlying SDE, typically a function of the
operating conditions (i.e., effective-irradiance ratio, F, and cell temperature, T).

Thus, one can plug in custom auxiliary equations computations, as long as they are
interface compatible. PVfit provides some common auxiliary equations.

Copyright 2023 Intelligent Measurement Systems LLC
"""
