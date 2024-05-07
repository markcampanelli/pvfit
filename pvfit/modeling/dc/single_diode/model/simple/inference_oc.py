"""
Infer (F, T) operating conditions (OC) using calibrated simple single-diode model (SDM)
and an I-V measurement.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from typing import Optional

import numpy
from scipy.constants import convert_temperature
import scipy.odr
import scipy.optimize

from pvfit.common import T_degC_abs_zero, k_B_J_per_K, k_B_eV_per_K, q_C
from pvfit.measurement.iv.types import FTData, IVData
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
from pvfit.modeling.dc.single_diode.model.simple import types
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_simple_ae
from pvfit.types import OdrOptions


def fit(
    *,
    iv_data: IVData,
    model_parameters: types.ModelParameters,
    oc_parameters_ic: Optional[dict] = None,
    oc_parameters_fixed: Optional[dict] = None,
    odr_options: Optional[OdrOptions] = None,
):
    """
    Fit the effective irradiance ratio and temperature of a PV device with a calibrated
    SDM from I-V curve data at a single operating condition.

    Inputs:
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
            E_g_eV_0 material band gap
        Initial conditions (ICs) for remaining fit parameters at OC, default value (None) means compute from I-V data:
            oc_params_ic dictionary of float (device-level)
                F effective irradiance ratio
                T_degC temperature

    Outputs (device-level):
        dict containing:
            oc_parameters_ic model parameters at ICs, provided or estimated from I-V curve observables
            oc_parameters model parameters from fit algorithm starting at oc_params_ic
            odr_outpus output for fit returned by the ODR solver
    """
    # TODO Add check for at least two distinguished (I,V) points in iv_data.

    # Check for provided ICs, and provide default values for missing items.
    if oc_parameters_ic is None:
        oc_parameters_ic = {}

    oc_parameters_ic_default = {"F": None, "T_degC": None}
    oc_parameters_ic_default.update(oc_parameters_ic)
    oc_parameters_ic = oc_parameters_ic_default

    # Check for provided fixed parameters, and provide default values for missing items.
    if oc_parameters_fixed is None:
        oc_parameters_fixed = {}

    oc_params_fixed_default = {"F": False, "T_degC": False}
    oc_params_fixed_default.update(oc_parameters_fixed)
    oc_parameters_fixed = oc_params_fixed_default

    # Check for provided odr parameters, and provide default values for missing items.
    if odr_options is None:
        odr_options = OdrOptions()

    odr_options_default = OdrOptions(maxit=1000)
    odr_options_default.update(odr_options)
    odr_options = odr_options_default

    # F IC.
    if oc_parameters_ic["F"] is None:
        # To estimate I_sc_A, get I_A at index of V_V that closest in absolute value to 0.
        I_sc_A_est = iv_data.I_A[numpy.argmin(numpy.abs(iv_data.V_V))]
        oc_parameters_ic["F"] = I_sc_A_est / model_parameters["I_sc_A_0"]
        if oc_parameters_ic["F"] <= 0 or not numpy.isfinite(oc_parameters_ic["F"]):
            raise ValueError(
                "Computed initial condition for effective irradiance ratio, F = "
                f"{oc_parameters_ic['F']}, is not strictly posiitve and finite."
            )
    elif oc_parameters_ic["F"] <= 0 or not numpy.isfinite(oc_parameters_ic["F"]):
        raise ValueError(
            "Provided initial condition for effective irradiance ratio, F = "
            f"{oc_parameters_ic['F']}, is not strictly positive and finite."
        )

    # T_degC IC.
    if oc_parameters_ic["T_degC"] is None:
        oc_parameters_ic["T_degC"] = scipy.optimize.least_squares(
            lambda x: sde_sim.I_sum_diode_anode_at_I_V(
                iv_data=IVData(I_A=iv_data.I_A, V_V=iv_data.V_V),
                model_parameters=sdm_simple_ae.compute_sde_model_parameters(
                    ft_data=FTData(
                        F=oc_parameters_ic["F"],
                        T_degC=x,
                    ),
                    model_parameters=model_parameters,
                ),
            )["I_sum_diode_anode_A"],
            model_parameters["T_degC_0"],
        ).x.item()

        if oc_parameters_ic["T_degC"] <= T_degC_abs_zero or not numpy.isfinite(
            oc_parameters_ic["T_degC"]
        ):
            raise ValueError(
                "Computed initial condition for device temperature, T_degC = "
                f"{oc_parameters_ic['T_degC']}, is not greater than absolute zero and "
                "finite."
            )
    elif oc_parameters_ic["T_degC"] <= T_degC_abs_zero or not numpy.isfinite(
        oc_parameters_ic["T_degC"]
    ):
        raise ValueError(
            "Provided initial condition for device temperature, T_degC = "
            f"{oc_parameters_ic['T_degC']}, is not greater than absolute zero and finite."
        )

    data = scipy.odr.Data(numpy.vstack((iv_data.V_V, iv_data.I_A)), 1)

    T_K_0 = convert_temperature(model_parameters["T_degC_0"], "Celsius", "Kelvin")

    # Inline these functions here for transformed model.
    def fun(beta, x):
        F = beta[0]
        T_K = convert_temperature(beta[1], "Celsius", "Kelvin")

        V_diode_V = x[0] + x[1] * model_parameters["R_s_Ohm_0"]
        I_rs_1_A = (
            model_parameters["I_rs_A_0"]
            * (T_K / T_K_0) ** 3
            * numpy.exp(
                model_parameters["E_g_eV_0"]
                / (model_parameters["n_0"] * k_B_eV_per_K)
                * (1 / T_K_0 - 1 / T_K)
            )
        )

        I_sc_A = F * model_parameters["I_sc_A_0"]

        I_ph_A = (
            I_rs_1_A
            * numpy.expm1(
                (q_C * I_sc_A * model_parameters["R_s_Ohm_0"])
                / (
                    model_parameters["N_s"]
                    * model_parameters["n_0"]
                    * k_B_J_per_K
                    * T_K
                )
            )
            + model_parameters["G_p_S_0"] * I_sc_A * model_parameters["R_s_Ohm_0"]
            + I_sc_A
        )

        return (
            I_ph_A
            - I_rs_1_A
            * numpy.expm1(
                (q_C * V_diode_V)
                / (
                    model_parameters["N_s"]
                    * model_parameters["n_0"]
                    * k_B_J_per_K
                    * T_K
                )
            )
            - model_parameters["G_p_S_0"] * V_diode_V
            - x[1]
        )

    model = scipy.odr.Model(fun, implicit=True)

    beta0 = numpy.array(
        [
            oc_parameters_ic["F"],
            oc_parameters_ic["T_degC"],
        ]
    )

    ifixb = [
        int(oc_parameters_fixed[param] is False) for param in oc_parameters_ic.keys()
    ]

    # Compute fit.
    output = scipy.odr.ODR(data, model, beta0=beta0, ifixb=ifixb, **odr_options).run()

    # Do not allow negative F. (TBD if this actually works.)
    if output.beta[0] < 0:
        ifixb[0] = 0
        beta0[0] = 0
        output = scipy.odr.ODR(
            data, model, beta0=beta0, ifixb=ifixb, **odr_options
        ).run()

    oc_params_fit = {"F": output.beta[0], "T_degC": output.beta[1]}

    return {
        "oc_parameters_ic": oc_parameters_ic,
        "oc_parameters": oc_params_fit,
        "odr_output": output,
    }
