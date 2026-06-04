"""
PVfit: Types for single-diode model (SDM) using simple auxiliary equations.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import dataclass

import numpy
import odrpack
import scipy.optimize

from pvfit.common import T_degC_abs_zero
from pvfit.types import FloatBroadcastable, IntBroadcastable


@dataclass
class ModelParameters:
    """
    Model parameters that should broadcastable with each other in SDM.

    All parameters are at the device level, where the device consists of N_s PV cells in
    series in each of N_p strings in parallel.
    """

    N_s: IntBroadcastable
    T_degC_0: FloatBroadcastable
    I_sc_A_0: FloatBroadcastable
    I_rs_A_0: FloatBroadcastable
    n_0: FloatBroadcastable
    R_s_Ohm_0: FloatBroadcastable
    G_p_S_0: FloatBroadcastable
    E_g_eV_0: FloatBroadcastable

    def __post_init__(self) -> None:
        """Validate fittable model parameters."""

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

        if numpy.any(numpy.logical_not(numpy.isfinite(self.T_degC_0))):
            raise ValueError(
                "device temperature at reference conditions is not well defined: "
                f"{self.T_degC_0}"
            )

        if numpy.any(self.T_degC_0 <= T_degC_abs_zero):
            raise ValueError(
                "device temperature at reference conditions is less than or equal to "
                f"absolute zero: {self.T_degC_0}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.I_sc_A_0))):
            raise ValueError(
                "short-circuit current at reference conditions is not well defined: "
                f"{self.I_sc_A_0}"
            )

        if numpy.any(self.I_sc_A_0 < 0):
            raise ValueError(
                "short-circuit current at reference conditions is less than zero: "
                f"{self.I_sc_A_0}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.I_rs_A_0))):
            raise ValueError(
                "reverse-saturation current at reference conditions is not well "
                f"defined: {self.I_rs_A_0}"
            )

        if numpy.any(self.I_rs_A_0 <= 0):
            raise ValueError(
                "reverse-saturation current at reference conditions is less than or "
                f"equal to zero: {self.I_rs_A_0}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.n_0))):
            raise ValueError(
                "diode ideality factor at reference conditions is not well defined: "
                f"{self.n_0}"
            )

        if numpy.any(self.n_0 <= 0):
            raise ValueError(
                "diode ideality factor at reference conditions is less than or equal "
                f"to zero: {self.n_0}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.R_s_Ohm_0))):
            raise ValueError(
                "series resistance at reference conditions is not well defined: "
                f"{self.R_s_Ohm_0}"
            )

        if numpy.any(self.R_s_Ohm_0 < 0):
            raise ValueError(
                "series resistance at reference conditions is less than zero: "
                f"{self.R_s_Ohm_0}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.G_p_S_0))):
            raise ValueError(
                "parallel conductance at reference conditions is not well defined: "
                f"{self.G_p_S_0}"
            )

        if numpy.any(self.G_p_S_0 < 0):
            raise ValueError(
                "parallel conductance at reference conditions is less than zero: "
                f"{self.G_p_S_0}"
            )

        if numpy.any(numpy.logical_not(numpy.isfinite(self.E_g_eV_0))):
            raise ValueError(
                "material bandgap at reference conditions is not well defined: "
                f"{self.E_g_eV_0}"
            )

        if numpy.any(self.E_g_eV_0 <= 0):
            raise ValueError(
                "material bandgap at reference conditions is less than or equal to "
                f"zero: {self.E_g_eV_0}"
            )


@dataclass
class ModelParametersUnfittable:
    """Unfittable model parameters."""

    N_s: int
    T_degC_0: float
    I_sc_A_0: float

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

        if not numpy.isfinite(self.T_degC_0):
            raise ValueError(
                "device temperature at reference conditions is not well defined: "
                f"{self.T_degC_0}"
            )

        if self.T_degC_0 <= T_degC_abs_zero:
            raise ValueError(
                "device temperature at reference conditions is less than or equal to "
                f"absolute zero: {self.T_degC_0}"
            )

        if not numpy.isfinite(self.I_sc_A_0):
            raise ValueError(
                "short-circuit current at reference conditions is not well defined: "
                f"{self.I_sc_A_0}"
            )

        if self.I_sc_A_0 < 0:
            raise ValueError(
                "short-circuit current at reference conditions is less than zero: "
                f"{self.I_sc_A_0}"
            )


@dataclass
class ModelParametersFittable:
    """Fittable model parameters."""

    I_rs_A_0: float
    n_0: float
    R_s_Ohm_0: float
    G_p_S_0: float
    E_g_eV_0: float

    def __post_init__(self) -> None:
        """Validate fittable model parameters."""

        if not numpy.isfinite(self.I_rs_A_0):
            raise ValueError(
                "reverse saturation current at reference conditions is not well "
                f"defined: {self.I_rs_A_0}"
            )

        if self.I_rs_A_0 <= 0:
            raise ValueError(
                "reverse saturation current at reference conditions is less than or "
                f"equal to zero: {self.I_rs_A_0}"
            )

        if not numpy.isfinite(self.n_0):
            raise ValueError(
                "diode ideality factor at reference conditions is not well defined: "
                f"{self.n_0}"
            )

        if self.n_0 <= 0:
            raise ValueError(
                "diode ideality factor at reference conditions is less than or equal "
                f"to zero: {self.n_0}"
            )

        if not numpy.isfinite(self.R_s_Ohm_0):
            raise ValueError(
                "series resistance at reference conditions is not well defined: "
                f"{self.R_s_Ohm_0}"
            )

        if self.R_s_Ohm_0 < 0:
            raise ValueError(
                "series resistance at reference conditions is less than zero: "
                f"{self.R_s_Ohm_0}"
            )

        if not numpy.isfinite(self.G_p_S_0):
            raise ValueError(
                "parallel conductance at reference conditions is not well defined: "
                f"{self.G_p_S_0}"
            )

        if self.G_p_S_0 < 0:
            raise ValueError(
                "parallel conductance at reference conditions is less than zero: "
                f"{self.G_p_S_0}"
            )

        if not numpy.isfinite(self.E_g_eV_0):
            raise ValueError(
                "material bandgap at reference conditions is not well defined: "
                f"{self.E_g_eV_0}"
            )

        if self.E_g_eV_0 <= 0:
            raise ValueError(
                "material bandgap at reference conditions is less than or equal to "
                f"zero: {self.E_g_eV_0}"
            )


@dataclass
class ModelParametersFittableFixed:
    """Fittable model parameters to be fixed for scalar model-parameter fits."""

    I_rs_A_0: bool = False
    n_0: bool = False
    R_s_Ohm_0: bool = False
    G_p_S_0: bool = False
    E_g_eV_0: bool = False


@dataclass
class OperatingConditions:
    """Operating-conditions parameters."""

    F: float
    T_degC: float

    def __post_init__(self) -> None:
        """Validate fittable model parameters."""

        if not numpy.isfinite(self.F):
            raise ValueError(
                "effective-irradiance ratio at operating conditions is not well "
                f"defined: {self.F}"
            )

        if self.F < 0.0:
            raise ValueError(
                "effective-irradiance ratio at operating conditions is less than zero: "
                f"{self.F}"
            )

        if not numpy.isfinite(self.T_degC):
            raise ValueError(
                "device temperature at operating conditions is not well defined: "
                f"{self.T_degC}"
            )

        if self.T_degC <= T_degC_abs_zero:
            raise ValueError(
                "device temperature at operating conditions is less than or equal to "
                f"absolute zero: {self.T_degC}"
            )


@dataclass
class OperatingConditionsFixed:
    """
    Operating-conditions parameters to be fixed for scalar operating-conditions
    parameter fits.
    """

    F: bool = False
    T_degC: bool = False


@dataclass
class ModelParametersFitResultLeastSquares:
    """Model-parameters fit result that used scipy.optimize.least_squares."""

    model_parameters_ic: ModelParameters
    model_parameters: ModelParameters
    solver_result: scipy.optimize.OptimizeResult


@dataclass
class ModelParametersFitResultODR:
    """Model-parameters fit result that used odrpack.odr_fit."""

    model_parameters_ic: ModelParameters
    model_parameters: ModelParameters
    solver_result: odrpack.OdrResult


@dataclass
class OperatingConditionsFitResultODR:
    """Operating-conditions fit result that used odrpack.odr_fit."""

    operating_conditions_ic: OperatingConditions
    operating_conditions: OperatingConditions
    solver_result: odrpack.OdrResult
