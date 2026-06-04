"""
PVfit: Types for single-diode equation (SDE).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import dataclass

import numpy
import odrpack

from pvfit.common import T_degC_abs_zero
from pvfit.types import FloatBroadcastable, IntBroadcastable


@dataclass
class ModelParameters:
    """
    Model parameters that should broadcastable against each other in SDE.

    All parameters are at the device level, where the device consists of N_s PV cells in
    series in each of N_p strings in parallel.
    """

    N_s: IntBroadcastable
    T_degC: FloatBroadcastable
    I_ph_A: FloatBroadcastable
    I_rs_A: FloatBroadcastable
    n: FloatBroadcastable
    R_s_Ohm: FloatBroadcastable
    G_p_S: FloatBroadcastable

    def __post_init__(self) -> None:
        """Validate model parameters."""

        if numpy.any(numpy.logical_not(numpy.isfinite(self.N_s))):
            raise ValueError(
                "number of cells in series in each parallel string is not well "
                f"defined: {self.N_s}"
            )

        if numpy.any(self.N_s <= 0):
            raise ValueError(
                "number of cells in series in each parallel string is less than or "
                f"equal to zero: {self.N_s}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.T_degC))):
            raise ValueError(f"device temperature is not well defined: {self.T_degC}")

        if numpy.any(self.T_degC <= T_degC_abs_zero):
            raise ValueError(
                "device temperature is less than or equal to absolute zero: "
                f"{self.T_degC}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.I_ph_A))):
            raise ValueError(f"photocurrent is not well defined: {self.I_ph_A}")

        if numpy.any(self.I_ph_A < 0):
            raise ValueError(f"photocurrent is less than zero: {self.I_ph_A}")

        if numpy.any(numpy.logical_not(numpy.isfinite(self.I_rs_A))):
            raise ValueError(
                f"reverse-saturation current is not well defined: {self.I_rs_A}"
            )

        if numpy.any(self.I_rs_A <= 0):
            raise ValueError(
                "reverse-saturation current is less than or equal to zero: "
                f"{self.I_rs_A}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.n))):
            raise ValueError(f"diode ideality factor is not well defined: {self.n}")

        if numpy.any(self.n <= 0):
            raise ValueError(
                f"diode ideality factor is less than or equal to zero: {self.n}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.R_s_Ohm))):
            raise ValueError(f"series resistance is not well defined: {self.R_s_Ohm}")

        if numpy.any(self.R_s_Ohm < 0):
            raise ValueError(f"series resistance is less than zero: {self.R_s_Ohm}")

        if numpy.any(numpy.logical_not(numpy.isfinite(self.G_p_S))):
            raise ValueError(f"parallel conductance is not well defined: {self.G_p_S}")

        if numpy.any(self.G_p_S < 0):
            raise ValueError(f"parallel conductance is less than zero: {self.G_p_S}")


@dataclass
class ModelParametersUnfittable:
    """Unfittable model parameters."""

    N_s: int
    T_degC: float

    def __post_init__(self) -> None:
        """Validate unfittable model parameters."""

        if not numpy.isfinite(self.N_s):
            raise ValueError(
                "number of cells in series in each parallel string is not well "
                f"defined: {self.N_s}"
            )

        if self.N_s <= 0:
            raise ValueError(
                "number of cells in series in each parallel string is less than or "
                f"equal to zero: {self.N_s}"
            )

        if not numpy.isfinite(self.T_degC):
            raise ValueError(f"device temperature is not well defined: {self.T_degC}")

        if self.T_degC <= T_degC_abs_zero:
            raise ValueError(
                "device temperature is less than or equal to absolute zero: "
                f"{self.T_degC}"
            )


@dataclass
class ModelParametersFittable:
    """Fittable model parameters."""

    I_ph_A: float
    I_rs_A: float
    n: float
    R_s_Ohm: float
    G_p_S: float

    def __post_init__(self) -> None:
        """Validate fittable model parameters."""
        if not numpy.isfinite(self.I_ph_A):
            raise ValueError(f"photocurrent is not well defined: {self.I_ph_A}")

        if self.I_ph_A < 0:
            raise ValueError(f"photocurrent is less than zero: {self.I_ph_A}")

        if not numpy.isfinite(self.I_rs_A):
            raise ValueError(
                f"reverse-saturation current is not well defined: {self.I_rs_A}"
            )

        if self.I_rs_A <= 0:
            raise ValueError(
                "reverse-saturation current is less than or equal to zero: "
                f"{self.I_rs_A}"
            )

        if not numpy.isfinite(self.n):
            raise ValueError(f"diode ideality factor is not well defined: {self.n}")

        if self.n <= 0:
            raise ValueError(
                f"diode ideality factor is less than or equal to zero: {self.n}"
            )

        if not numpy.isfinite(self.R_s_Ohm):
            raise ValueError(f"series resistance is not well defined: {self.R_s_Ohm}")

        if self.R_s_Ohm < 0:
            raise ValueError(f"series resistance is less than zero: {self.R_s_Ohm}")

        if not numpy.isfinite(self.G_p_S):
            raise ValueError(f"parallel conductance is not well defined: {self.G_p_S}")

        if self.G_p_S < 0:
            raise ValueError(f"parallel conductance is less than zero: {self.G_p_S}")


@dataclass
class ModelParametersFittableFixed:
    """Fittable model parameters to be fixed for scalar model-parameter fits."""

    I_ph_A: bool = False
    I_rs_A: bool = False
    n: bool = False
    R_s_Ohm: bool = False
    G_p_S: bool = False


@dataclass
class FitResultODR:
    """Fit result that used odrpack.odr_fit."""

    model_parameters_ic: ModelParameters
    model_parameters: ModelParameters
    solver_result: odrpack.OdrResult
