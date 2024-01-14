"""
Calibrate single-diode model (SDM) using orthogonal distance regression (ODR) and only
information from module spec sheet.
"""

import numpy
from scipy.constants import convert_temperature
from scipy.optimize import least_squares

from pvfit.common import k_B_J_per_K, k_B_eV_per_K, q_C
from pvfit.measurement.iv.types import FTData
from pvfit.modeling.dc.common import MATERIALS, T_degC_stc
import pvfit.modeling.dc.single_diode.equation.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_ae
from pvfit.modeling.dc.single_diode.model.simple.inference_ic import (
    estimate_model_parameters_fittable_ic,
)
from pvfit.modeling.dc.single_diode.model.simple.types import ModelParameters

delta_T_degC = 0.1


def fun(
    x: numpy.ndarray,
    *,
    I_sc_A_0: float,
    I_mp_A_0: float,
    V_mp_V_0: float,
    V_oc_V_0: float,
    dI_sc_dT_A_per_K_0: float,
    dP_mp_dT_W_per_K_0: float,
    dV_oc_dT_V_per_K_0: float,
    T_degC_0: float,
    N_s: int,
):
    """Computes all the simultaneous constraints produced by SDM at RC."""

    # Everything measured at RC, where F=1 and T=T_degC_0.
    I_rs_A_0 = numpy.exp(x[0])
    n_0 = x[1]
    R_s_Ohm_0 = x[2]
    G_p_S_0 = x[3]
    E_g_eV_0 = x[4]

    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")
    scaled_thermal_voltage_V_0 = N_s * n_0 * k_B_J_per_K * T_K_0 / q_C

    # Derivative of reverse-saturation current w.r.t. temperature at RC.
    dI_rs_1_dT_A_per_K_0 = (I_rs_A_0 / T_K_0) * (
        3 + E_g_eV_0 / (n_0 * k_B_eV_per_K * T_K_0)
    )

    # Photocurrent at RC.
    I_ph_A_0 = (
        I_rs_A_0 * numpy.expm1((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        + G_p_S_0 * I_sc_A_0 * R_s_Ohm_0
        + I_sc_A_0
    )

    # Derivative of photocurrent w.r.t. T at RC.
    dI_ph_dT_A_per_K_0 = (
        dI_rs_1_dT_A_per_K_0
        * numpy.expm1((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        + ((I_rs_A_0 * R_s_Ohm_0) / (scaled_thermal_voltage_V_0 * T_K_0))
        * numpy.exp((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        * (T_K_0 * dI_sc_dT_A_per_K_0 - I_sc_A_0)
        + dI_sc_dT_A_per_K_0 * (G_p_S_0 * R_s_Ohm_0 + 1)
    )

    # Maximum-power power point.
    V_diode_mp_V_0 = V_mp_V_0 + I_mp_A_0 * R_s_Ohm_0
    y_0 = (
        I_ph_A_0
        - I_rs_A_0 * numpy.expm1(V_diode_mp_V_0 / scaled_thermal_voltage_V_0)
        - G_p_S_0 * V_diode_mp_V_0
    ) - I_mp_A_0

    # Maximum attained at maximum-power point.
    y_1 = (V_mp_V_0 - I_mp_A_0 * R_s_Ohm_0) * (
        (I_rs_A_0 / scaled_thermal_voltage_V_0)
        * numpy.exp(V_diode_mp_V_0 / scaled_thermal_voltage_V_0)
        + G_p_S_0
    ) - I_mp_A_0

    # Open-circuit voltage point.
    y_2 = (
        I_ph_A_0
        - I_rs_A_0 * numpy.expm1(V_oc_V_0 / scaled_thermal_voltage_V_0)
        - V_oc_V_0 * G_p_S_0
    )

    # Maximum-power power temperature coefficient.
    maximum_powers = sde_sim.P_mp(
        model_parameters=sdm_ae.compute_sde_model_parameters(
            ft_data=FTData(
                F=1,
                T_degC=numpy.array([T_degC_0 - delta_T_degC, T_degC_0 + delta_T_degC]),
            ),
            model_parameters=ModelParameters(
                N_s=N_s,
                T_degC_0=T_degC_0,
                I_sc_A_0=I_sc_A_0,
                I_rs_A_0=I_rs_A_0,
                n_0=n_0,
                R_s_Ohm_0=R_s_Ohm_0,
                G_p_S_0=G_p_S_0,
                E_g_eV_0=E_g_eV_0,
            ),
        )
    )[0]
    y_3 = (
        (maximum_powers[1] - maximum_powers[0]) / (2 * delta_T_degC)
    ) - dP_mp_dT_W_per_K_0

    # Open-circuit voltage temperature coefficient.
    y_4 = (
        (
            dI_ph_dT_A_per_K_0
            - dI_rs_1_dT_A_per_K_0 * numpy.expm1(V_oc_V_0 / scaled_thermal_voltage_V_0)
            + ((I_rs_A_0 * V_oc_V_0) / (scaled_thermal_voltage_V_0 * T_K_0))
            * numpy.exp(V_oc_V_0 / scaled_thermal_voltage_V_0)
        )
        / (
            (I_rs_A_0 / scaled_thermal_voltage_V_0)
            * numpy.exp(V_oc_V_0 / scaled_thermal_voltage_V_0)
            + G_p_S_0
        )
    ) - dV_oc_dT_V_per_K_0

    output = [y_0, y_1, y_2, y_3, y_4]

    # FUTURE Optionally add one more OC at which to fit just three points, not
    # coefficients. Then scan through IEC 61853-1 matrix for best additional point to
    # add.

    return numpy.array(output)


def fit(
    *,
    I_sc_A_0: float,
    I_mp_A_0: float,
    V_mp_V_0: float,
    V_oc_V_0: float,
    dI_sc_dT_A_per_degC_0: float,
    dP_mp_dT_W_per_degC_0: float,
    dV_oc_dT_V_per_degC_0: float,
    N_s: int,
    T_degC_0: float = T_degC_stc,
    material: str = "x-Si",
) -> dict:
    """
    Use nonlinear least squares (NLLS) to fit the implicit 6-parameter single-diode
    model (SDM) equivalent-circuit model given module specification (spec) sheet data at
    reference condtions.

    Inputs: FIXME
    Observables at operating conditions (device-level):
        V_V terminal voltage
        I_A terminal current
        F effective irradiance
        T_degC temperature
            Assumes that V_V, I_A, F, and T_degC arguments are rank-1 float64 numpy
            arrays with the same size, sufficiently distributed over irradiance and
            temperature, and with at least three well-distributed points, e.g., near
            short-circuit current, maximum power, and open-circuit voltage, at/near a
            single operating condition (OC), which is usualy the reference condition
            (RC).
    Model parameters at reference condition (device-level):
        Non-fitted (always fixed) device parmeters:
            N_s number of cells in series in each parallel string
            T_degC_0 temperature
        Initial conditions (ICs) for remaining fit parameters, default value (None)
            means compute from I-V data:
                model_params_ic dictionary of float:
                    I_sc_A_0 short-circuit current
                    I_rs_1_A_0 diode reverse-saturation current
                    n_1_0 diode ideality factor
                    R_s_Ohm_0 series resistance
                    G_p_S_0 parallel (shunt) conductance
                    E_g_eV_0 material band gap
    Supporting information:
        material (optional) PV material system, helps to determine proper ICs from data

    Outputs (device-level):
        dict containing:
            iv_params_data_0 I-V curve parameters at RC estimated from I-V curve data
            model_params_ic model parameters at ICs, provided or estimated from I-V
                curve observables
            I_sum_A_ic residuals for current sum at diode's anode node for model
                calculated using model_params_ic
            iv_params_ic_0 I-V curve parameters at RC calculated using model_params_ic
            model_params_fit model parameters from fit algorithm starting at
                model_params_ic
            I_sum_A_fit residuals for current sum at diode's anode node for model
                calculated using model_params_fit
            iv_params_fit_0 I-V curve parameters at RC calculated using model_params_fit
            sol full solution object for fit returned by the solver (for a transformed
                problem)
    """
    # Prepare initial conditions.
    V_V_vec = numpy.array([0.0, V_mp_V_0, V_oc_V_0])
    I_A_vec = numpy.array([I_sc_A_0, I_mp_A_0, 0.0])

    # TODO Define a fit_prep in this module that calls underlying equation.fit_prep.
    fit_prep_result = fit_prep(
        V_V=V_V_vec,
        I_A=I_A_vec,
        T_degC=T_degC_0,
        N_s=N_s,
        model_params_ic=None,
        model_params_fixed=None,
    )

    # Initial condition for E_g_eV_0.
    if material in MATERIALS:
        fit_prep_result["model_params_ic"]["E_g_eV"] = MATERIALS[material]["E_g_eV_stc"]
    else:
        raise ValueError(
            "unrecognized material for determining initial condition for material band "
            "gap at reference condition"
        )

    x_0 = numpy.array(
        [
            numpy.log(fit_prep_result["model_params_ic"]["I_rs_1_A"]),
            fit_prep_result["model_params_ic"]["n_1"],
            fit_prep_result["model_params_ic"]["R_s_Ohm"],
            fit_prep_result["model_params_ic"]["G_p_S"],
            fit_prep_result["model_params_ic"]["E_g_eV"],
        ]
    )

    kwargs = {
        "I_sc_A_0": I_sc_A_0,
        "I_mp_A_0": I_mp_A_0,
        "V_mp_V_0": V_mp_V_0,
        "V_oc_V_0": V_oc_V_0,
        "dI_sc_dT_A_per_K_0": dI_sc_dT_A_per_degC_0,
        "dP_mp_dT_W_per_K_0": dP_mp_dT_W_per_degC_0,
        "dV_oc_dT_V_per_K_0": dV_oc_dT_V_per_degC_0,
        "N_s": N_s,
        "T_degC_0": T_degC_0,
    }

    # NLLS
    output = least_squares(
        fun,
        x_0,
        jac="3-point",
        bounds=(
            [-numpy.inf, 0, 0, 0, 0],
            numpy.inf,
        ),
        method="trf",
        x_scale="jac",
        verbose=1,
        max_nfev=1000 * (len(kwargs) - 2),
        kwargs=kwargs,
    )

    print(f"Final fun(x): {list(output.fun)}")

    # Transform back fit values.
    I_rs_1_A_0 = numpy.exp(output.x[0])
    n_1_0 = output.x[1]
    R_s_Ohm_0 = output.x[2]
    G_p_S_0 = output.x[3]
    E_g_eV_0 = output.x[4]

    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")
    dI_rs_1_dT_A_per_K_0 = (I_rs_1_A_0 / T_K_0) * (
        3 + E_g_eV_0 / (n_1_0 * k_B_eV_per_K * T_K_0)
    )
    scaled_thermal_voltage_V_0 = N_s * n_1_0 * k_B_J_per_K * T_K_0 / q_C
    dI_ph_dT_A_per_K_0 = (
        dI_rs_1_dT_A_per_K_0
        * numpy.expm1((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        + ((I_rs_1_A_0 * R_s_Ohm_0) / (scaled_thermal_voltage_V_0 * T_K_0))
        * numpy.exp((I_sc_A_0 * R_s_Ohm_0) / scaled_thermal_voltage_V_0)
        * (T_K_0 * dI_sc_dT_A_per_degC_0 - I_sc_A_0)
        + dI_sc_dT_A_per_degC_0 * (G_p_S_0 * R_s_Ohm_0 + 1)
    )
    V_diode_sc_V_0 = I_sc_A_0 * R_s_Ohm_0
    dI_sc_dT_A_per_degC_0 = (
        dI_ph_dT_A_per_K_0
        - dI_rs_1_dT_A_per_K_0
        * numpy.expm1(V_diode_sc_V_0 / scaled_thermal_voltage_V_0)
        + (I_rs_1_A_0 * V_diode_sc_V_0)
        / (scaled_thermal_voltage_V_0 * T_K_0)
        * numpy.exp(V_diode_sc_V_0 / scaled_thermal_voltage_V_0)
    ) / (
        (I_rs_1_A_0 * R_s_Ohm_0 / scaled_thermal_voltage_V_0)
        * numpy.exp(V_diode_sc_V_0 / scaled_thermal_voltage_V_0)
        + G_p_S_0 * R_s_Ohm_0
        + 1
    )
    maximum_powers = sde_sim.P_mp(
        **auxiliary_equations(
            F=1,
            T_degC=numpy.array([T_degC_0 - delta_T_degC, T_degC_0 + delta_T_degC]),
            N_s=N_s,
            T_degC_0=T_degC_0,
            I_sc_A_0=I_sc_A_0,
            I_rs_1_A_0=I_rs_1_A_0,
            n_1_0=n_1_0,
            R_s_Ohm_0=R_s_Ohm_0,
            G_p_S_0=G_p_S_0,
            E_g_eV_0=E_g_eV_0,
        )
    )["P_mp_W"]
    dP_mp_dT_W_per_degC_0 = (maximum_powers[1] - maximum_powers[0]) / (2 * delta_T_degC)
    dV_oc_dT_V_per_degC_0 = (
        dI_ph_dT_A_per_K_0
        - dI_rs_1_dT_A_per_K_0 * numpy.expm1(V_oc_V_0 / scaled_thermal_voltage_V_0)
        + (I_rs_1_A_0 * V_oc_V_0)
        / (scaled_thermal_voltage_V_0 * T_K_0)
        * numpy.exp(V_oc_V_0 / scaled_thermal_voltage_V_0)
    ) / (
        (I_rs_1_A_0 / scaled_thermal_voltage_V_0)
        * numpy.exp(V_oc_V_0 / scaled_thermal_voltage_V_0)
        + G_p_S_0
    )

    model_params_fit = {
        "I_sc_A_0": I_sc_A_0,
        "I_rs_1_A_0": I_rs_1_A_0,
        "n_1_0": n_1_0,
        "R_s_Ohm_0": R_s_Ohm_0,
        "G_p_S_0": G_p_S_0,
        "E_g_eV_0": E_g_eV_0,
        "N_s": N_s,
        "T_degC_0": T_degC_0,
    }

    iv_params_fit_0 = sde_sim.iv_curve_parameters(
        F=1, T_degC=T_degC_0, **model_params_fit
    )
    iv_params_fit_0["dI_sc_dT_A_per_degC_0"] = dI_sc_dT_A_per_degC_0
    iv_params_fit_0["dP_mp_dT_W_per_degC_0"] = dP_mp_dT_W_per_degC_0
    iv_params_fit_0["dV_oc_dT_V_per_degC_0"] = dV_oc_dT_V_per_degC_0

    return {
        **fit_prep_result,
        "model_params_fit": model_params_fit,
        "I_sum_A_fit": sde_sim.I_sum_diode_anode_at_I_V(
            V_V=V_V_vec,
            I_A=I_A_vec,
            F=1,
            T_degC=T_degC_0,
            **model_params_fit,
        )["I_sum_A"],
        "iv_params_fit_0": iv_params_fit_0,
        "output": output,
    }
