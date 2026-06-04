"""
Infer (F, T) operating conditions (OC) using calibrated simple single-diode model (SDM)
and an I-V measurement.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from dataclasses import asdict
from typing import Optional

import numpy
import odrpack
from scipy.constants import convert_temperature

from pvfit.common import T_degC_abs_zero, k_B_J_per_K, k_B_eV_per_K, q_C
from pvfit.measurement.iv.types import IVData
from pvfit.modeling.dc.single_diode.model.simple.initial_conditions import (
    determine_operating_conditions_ic,
)
from pvfit.modeling.dc.single_diode.model.simple.types import (
    OperatingConditionsFitResultODR,
    ModelParameters,
    OperatingConditions,
    OperatingConditionsFixed,
)
from pvfit.types import OdrOptions


def fit(
    *,
    iv_data: IVData,
    model_parameters: ModelParameters,
    operating_conditions_ic: Optional[OperatingConditions] = None,
    operating_conditions_fixed: Optional[OperatingConditionsFixed] = None,
    odr_options: Optional[OdrOptions] = None,
) -> OperatingConditions:
    """
    Fit the effective irradiance ratio and temperature of a PV device with a calibrated
    SDM from I-V curve data at a single operating condition.

    # FIXME Docstring descriptions to match other fit functions.
    Parameters
    ----------
    Observables at operating condition (device-level):
        iv_curve
            Should contain minimally two sufficiently separated points.
    Model parameters at reference condition (device-level):
        Non-fitted device parameters:
            N_s integer number of cells in series in each parallel string
            T_degC_0 temperature
        Previously calibrated model parameters (device-level):
            I_sc_A_0 short-circuit current
            I_rs_1_A_0 diode reverse-saturation current
            n_1_0 diode ideality factor
            R_s_Ohm_0 series resistance
            G_p_S_0 parallel (shunt) conductance
            E_g_eV_0 material bandgap
        Initial conditions (ICs) for remaining fit parameters at OC, default value (None) means compute from I-V data:
            oc_params_ic dictionary of float (device-level)
                F effective irradiance ratio
                T_degC temperature

    Returns
    -------
    dictionary with the following
        oc_parameters_ic
            model parameters at ICs, provided or estimated from I-V curve observables
        oc_parameters
            model parameters from fit algorithm starting at oc_parameters_ic
        solver_result
            odrpack.OdrResult object, with solver result (for a transformed problem)
    """
    # TODO Add check for at least two distinguished (I, V) points in iv_data.

    T_K_0 = convert_temperature(model_parameters.T_degC_0, "Celsius", "Kelvin")

    if operating_conditions_ic is None:
        operating_conditions_ic = determine_operating_conditions_ic(iv_data=iv_data)

    if operating_conditions_fixed is None:
        operating_conditions_fixed = OperatingConditionsFixed()

    def f(x: numpy.ndarray, beta: numpy.ndarray) -> numpy.ndarray:
        """
        Implicit system of SDM-derived equations over which model parameters are
        optimized. Note closure over some fixed variables.
        """
        F = beta[0]
        T_K = convert_temperature(beta[1], "Celsius", "Kelvin")

        V_diode_V = x[0] + x[1] * model_parameters.R_s_Ohm_0
        I_rs_A = (
            model_parameters.I_rs_A_0
            * (T_K / T_K_0) ** 3
            * numpy.exp(
                model_parameters.E_g_eV_0
                / (model_parameters.n_0 * k_B_eV_per_K)
                * (1 / T_K_0 - 1 / T_K)
            )
        )

        I_sc_A = F * model_parameters.I_sc_A_0

        I_ph_A = (
            I_rs_A
            * numpy.expm1(
                (q_C * I_sc_A * model_parameters.R_s_Ohm_0)
                / (model_parameters.N_s * model_parameters.n_0 * k_B_J_per_K * T_K)
            )
            + model_parameters.G_p_S_0 * I_sc_A * model_parameters.R_s_Ohm_0
            + I_sc_A
        )

        return (
            I_ph_A
            - I_rs_A
            * numpy.expm1(
                (q_C * V_diode_V)
                / (model_parameters.N_s * model_parameters.n_0 * k_B_J_per_K * T_K)
            )
            - model_parameters.G_p_S_0 * V_diode_V
            - x[1]
        )

    xdata = numpy.vstack((iv_data.V_V, iv_data.I_A))
    ydata = numpy.zeros(xdata.shape[-1])
    beta0 = numpy.array(
        [
            operating_conditions_ic.F,
            operating_conditions_ic.T_degC,
        ]
    )
    bounds = (numpy.array((0.0, T_degC_abs_zero)), numpy.full_like(beta0, numpy.inf))
    fix_beta = [
        operating_conditions_fixed[key] for key in operating_conditions_ic.keys()
    ]

    if odr_options is None:
        odr_options = OdrOptions()

    solver_result = odrpack.odr_fit(
        f,
        xdata,
        ydata,
        beta0,
        bounds=bounds,
        task="implicit-ODR",
        fix_beta=fix_beta,
        **asdict(odr_options),
    )
    operating_conditions = OperatingConditions(
        F=solver_result.beta[0],
        T_degC=solver_result.beta[1],
    )

    return OperatingConditionsFitResultODR(
        operating_conditions_ic=operating_conditions_ic,
        operating_conditions=operating_conditions,
        solver_result=solver_result,
    )
