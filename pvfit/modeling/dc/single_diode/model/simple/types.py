"""
PVfit: Types for single-diode model (SDM) using simple auxiliary equations.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import TypedDict

import numpy
import scipy.odr
import scipy.optimize

from pvfit.common import T_degC_abs_zero
from pvfit.types import FloatBroadcastable, IntBroadcastable


class ModelParameters(TypedDict):
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


class ModelParametersUnfittable(TypedDict):
    """Unfittable model parameters."""

    N_s: int
    T_degC_0: float


def validate_model_parameters_unfittable(
    *,
    model_parameters_unfittable: ModelParametersUnfittable,
) -> None:
    """Validate unfittable model parameters."""
    if model_parameters_unfittable["N_s"] <= 0:
        raise ValueError(
            "provided value for number of cells in series in each parallel string, "
            f"N_s = {model_parameters_unfittable['N_s']}, is less than or equal to "
            "zero"
        )

    if model_parameters_unfittable["T_degC_0"] <= T_degC_abs_zero:
        raise ValueError(
            "provided value for refrence device temperature, T_degC_0 = "
            f"{model_parameters_unfittable['T_degC_0']}, is less than or equal to "
            "absolute zero."
        )


class ModelParametersFittable(TypedDict):
    """Fittable model parameters."""

    I_sc_A_0: float
    I_rs_A_0: float
    n_0: float
    R_s_Ohm_0: float
    G_p_S_0: float
    E_g_eV_0: float


def validate_model_parameters_fittable(
    *,
    model_parameters_fittable: ModelParametersFittable,
) -> None:
    """Validate fittable model parameters."""
    if model_parameters_fittable["I_sc_A_0"] < 0:
        raise ValueError(
            "short-circuit current at reference conditions is less than zero: "
            f"{model_parameters_fittable}"
        )

    if model_parameters_fittable["I_rs_A_0"] <= 0:
        raise ValueError(
            "reverse saturation current at reference conditions is less than or equal "
            f"to zero: {model_parameters_fittable}"
        )

    if model_parameters_fittable["n_0"] <= 0:
        raise ValueError(
            "diode ideality factor at reference conditions is less than or equal to "
            f"zero: {model_parameters_fittable}"
        )

    if model_parameters_fittable["R_s_Ohm_0"] < 0:
        raise ValueError(
            "series resistance at reference conditions is less than zero: "
            f"{model_parameters_fittable}"
        )

    if model_parameters_fittable["G_p_S_0"] < 0:
        raise ValueError(
            "parallel conductance at reference conditions is less than zero: "
            f"{model_parameters_fittable}"
        )

    if model_parameters_fittable["E_g_eV_0"] <= 0:
        raise ValueError(
            "material band gap at reference conditions is less than or equal to zero: "
            f"{model_parameters_fittable}"
        )

    # Raise for anything else non-finite. For example, inf or nan.
    if numpy.any(
        numpy.logical_not(
            numpy.isfinite(numpy.array(list(model_parameters_fittable.values())))
        )
    ):
        raise ValueError(
            f"at least one model parameter is not finite: {model_parameters_fittable}"
        )


class ModelParametersFittableProvided(TypedDict, total=False):
    """
    Optionally provided fittable model parameters, e.g., for initial conditions (IC).
    """

    I_sc_A_0: float
    I_rs_A_0: float
    n_0: float
    R_s_Ohm_0: float
    G_p_S_0: float
    E_g_eV_0: float


class ModelParametersFittableFixed(TypedDict):
    """Fittable model parameters to be fixed for parameter fits."""

    I_sc_A_0: bool
    I_rs_A_0: bool
    n_0: bool
    R_s_Ohm_0: bool
    G_p_S_0: bool
    E_g_eV_0: bool


def get_model_parameters_fittable_fixed_default() -> ModelParametersFittableFixed:
    """Get default ModelParametersFittableFixed (no parameter fixing)."""

    return ModelParametersFittableFixed(
        I_sc_A_0=False,
        I_rs_A_0=False,
        n_0=False,
        R_s_Ohm_0=False,
        G_p_S_0=False,
        E_g_eV_0=False,
    )


class ModelParametersFittableFixedProvided(TypedDict, total=False):
    """Optionally provided fittable model parameters to be fixed for parameter fits."""

    I_sc_A_0: bool
    I_rs_A_0: bool
    n_0: bool
    R_s_Ohm_0: bool
    G_p_S_0: bool
    E_g_eV_0: bool


class FitResultODR(TypedDict):
    """Fit result that used scipy.odr.ODR."""

    model_parameters_ic: ModelParameters
    model_parameters: ModelParameters
    odr_output: scipy.odr.Output


class FitResultLeastSquares(TypedDict):
    """Fit result that used scipy.optimize.least_squares."""

    model_parameters_ic: ModelParameters
    model_parameters: ModelParameters
    optimize_result: scipy.optimize.OptimizeResult
