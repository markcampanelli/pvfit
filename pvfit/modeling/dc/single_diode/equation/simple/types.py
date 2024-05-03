"""
PVfit: Types for single-diode equation (SDE).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import TypedDict

import numpy
import scipy.odr

from pvfit.common import T_degC_abs_zero
from pvfit.types import FloatBroadcastable, IntBroadcastable


class ModelParameters(TypedDict):
    """
    Model parameters that should broadcastable with each other in SDE.

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


class ModelParametersUnfittable(TypedDict):
    """Unfittable model parameters."""

    N_s: int
    T_degC: float


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

    if model_parameters_unfittable["T_degC"] <= T_degC_abs_zero:
        raise ValueError(
            "provided value for device temperature, T_degC = "
            f"{model_parameters_unfittable['T_degC']}, is less than or equal to "
            "absolute zero."
        )


class ModelParametersFittable(TypedDict):
    """Fittable model parameters."""

    I_ph_A: float
    I_rs_A: float
    n: float
    R_s_Ohm: float
    G_p_S: float


def validate_model_parameters_fittable(
    *,
    model_parameters_fittable: ModelParametersFittable,
) -> None:
    """Validate fittable model parameters."""
    if model_parameters_fittable["I_ph_A"] < 0:
        raise ValueError(f"photocurrent is less than zero: {model_parameters_fittable}")

    if model_parameters_fittable["I_rs_A"] <= 0:
        raise ValueError(
            "reverse saturation current is less than or equal to zero: "
            f"{model_parameters_fittable}"
        )

    if model_parameters_fittable["n"] <= 0:
        raise ValueError(
            "diode ideality factor is less than or equal to zero: "
            f"{model_parameters_fittable}"
        )

    if model_parameters_fittable["R_s_Ohm"] < 0:
        raise ValueError(
            f"series resistance is less than zero: {model_parameters_fittable}"
        )

    if model_parameters_fittable["G_p_S"] < 0:
        raise ValueError(
            f"parallel conductance is less than zero: {model_parameters_fittable}"
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

    I_ph_A: float
    I_rs_A: float
    n: float
    R_s_Ohm: float
    G_p_S: float


class ModelParametersFittableFixed(TypedDict):
    """Fittable model parameters to be fixed for parameter fits."""

    I_ph_A: bool
    I_rs_A: bool
    n: bool
    R_s_Ohm: bool
    G_p_S: bool


def get_model_parameters_fittable_fixed_default() -> ModelParametersFittableFixed:
    """Get default FittableModelParametersFixed (no parameter fixing)."""

    return ModelParametersFittableFixed(
        I_ph_A=False,
        I_rs_A=False,
        n=False,
        R_s_Ohm=False,
        G_p_S=False,
    )


class ModelParametersFittableFixedProvided(TypedDict, total=False):
    """Optionally provided fittable model parameters to be fixed for parameter fits."""

    I_ph_A: bool
    I_rs_A: bool
    n: bool
    R_s_Ohm: bool
    G_p_S: bool


class FitResultODR(TypedDict):
    """Fit result that used scipy.odr.ODR."""

    model_parameters_ic: ModelParameters
    model_parameters: ModelParameters
    odr_output: scipy.odr.Output
