"""
PVfit: Auxiliary equations for single-diode model (SDM) with photoconductive shunt.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy
from scipy.constants import convert_temperature

from pvfit.common import k_B_eV_per_K
from pvfit.measurement.iv.types import FTData
from pvfit.modeling.dc.common import get_scaled_thermal_voltage
import pvfit.modeling.dc.single_diode.equation.simple.types as sde_types
from pvfit.modeling.dc.single_diode.model.simple.types import ModelParameters


def compute_sde_model_parameters(
    *,
    ft_data: FTData,
    model_parameters: ModelParameters,
) -> sde_types.ModelParameters:
    """
    Computes auxiliary equations for SDM with photoconductive shunt at the specified
    effective irradiance ratio and cell temperature and model parameters, producing
    model parameters for simple SDE.

    Parameters
    ----------
    ft_data
        Operating conditions
    model_parameters
        Model parameters for SDM with photoconductive shunt

    Returns
    -------
    model_parameters
        Model parameters for simple SDE
    """
    # Temperatures must be in Kelvin.
    T_K = convert_temperature(ft_data.T_degC, "Celsius", "Kelvin")
    T_K_0 = convert_temperature(model_parameters["T_degC_0"], "Celsius", "Kelvin")

    # Compute variables at specified operating conditions.

    # Compute diode ideality factor (constant).
    n = model_parameters["n_0"]

    # Compute band gap (constant).
    E_g_eV = model_parameters["E_g_eV_0"]

    # Compute reverse-saturation current at T_degC (this is independent of
    # F, I_sc_0_A, R_s_0_Ohm, and G_p_0_S).
    I_rs_A = (
        model_parameters["I_rs_A_0"]
        * (T_K / T_K_0) ** 3
        * numpy.exp(E_g_eV / (n * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
    )

    # Compute series resistance (constant).
    R_s_Ohm = model_parameters["R_s_Ohm_0"]

    # Compute parallel conductance (photoconductive shunt).
    G_p_S = ft_data.F * model_parameters["G_p_S_0"]

    # Compute photo-generated current at F and T_degC (V=0 with I=Isc for this).
    I_sc_A = model_parameters["I_sc_A_0"] * ft_data.F
    V_diode_sc_V = I_sc_A * R_s_Ohm
    I_ph_A = (
        I_sc_A
        + I_rs_A
        * numpy.expm1(
            V_diode_sc_V
            / (
                n
                * get_scaled_thermal_voltage(
                    N_s=model_parameters["N_s"], T_degC=ft_data.T_degC
                )
            )
        )
        + G_p_S * V_diode_sc_V
    )

    # Ensure model parameters are broadcastable.
    N_s, T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S = numpy.broadcast_arrays(
        model_parameters["N_s"], ft_data.T_degC, I_ph_A, I_rs_A, n, R_s_Ohm, G_p_S
    )

    return sde_types.ModelParameters(
        N_s=N_s,
        T_degC=T_degC,
        I_ph_A=I_ph_A,
        I_rs_A=I_rs_A,
        n=n,
        R_s_Ohm=R_s_Ohm,
        G_p_S=G_p_S,
    )
