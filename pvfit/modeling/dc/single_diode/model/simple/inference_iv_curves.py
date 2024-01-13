"""Calibrate single-diode model (SDM) using orthogonal distance regression (ODR)."""

from typing import Optional

import numpy
from scipy.constants import convert_temperature
from scipy.odr import Data, Model, ODR
from scipy.optimize import least_squares
from scipy.stats import linregress

from pvfit.common.constants import (
    T_degC_abs_zero,
    T_degC_stc,
    k_B_J_per_K,
    k_B_eV_per_K,
    materials,
    q_C,
)
from pvfit.modeling.inference.dc.single_diode.equation import fit as sde_fit
from pvfit.modeling.inference.dc.single_diode.model_fit_prep import fit_prep
import pvfit.modeling.dc.single_diode.simulation.model as sdm


def fit(
    *,
    V_V: numpy.ndarray,
    I_A: numpy.ndarray,
    F: numpy.ndarray,
    T_degC: numpy.ndarray,
    N_s: int,
    material: str = "x-Si",
    model_params_ic: Optional[dict] = None,
    model_params_fixed: Optional[dict] = None,
    T_degC_0: float = T_degC_stc,
) -> dict:
    """
    Use orthogonal distance regression (ODR) to fit the implicit 6-parameter
    single-diode model (SDM) equivalent-circuit model given current-voltage (I-V) curve
    data taken over a range of irradiance and temperature (F-T) operating conditions.

    Inputs:
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
        Parameter fixing booleans (True to fix to user-provided IC or automatically
            calculated IC):
                model_params_fixed dictionary of bool:
                    I_sc_A_0 fix short-circuit current
                    I_rs_1_A_0 fix diode reverse-saturation current
                    n_1_0 fix diode ideality factor
                    R_s_Ohm_0 fix series resistance
                    G_p_S_0 fix parallel (shunt) conductance
                    E_g_eV_0 fix material band gap
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
            model_params_fixed model parameters fixed during fit
            sol full solution object for fit returned by the solver (for a transformed
                problem)
    """
    # Prepare for fit.
    fit_prep_result = fit_prep(
        V_V=V_V,
        I_A=I_A,
        F=F,
        T_degC=T_degC,
        N_s=N_s,
        material=material,
        model_params_ic=model_params_ic,
        model_params_fixed=model_params_fixed,
        T_degC_0=T_degC_0,
    )

    # Optimization: Thermal voltage at RC scaled by the number of cells in series in
    # each parallel string.
    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")
    # scaled_thermal_voltage_V_0 = N_s * k_B_J_per_K * T_K_0 / q_C
    # E_g_scale_eV = k_B_eV_per_K * T_K_0

    # # Scale observables for fit.
    # V_scale_V = fit_prep_result["iv_params_ic_0"]["V_oc_V"]
    # I_scale_A = fit_prep_result["iv_params_ic_0"]["I_sc_A"]
    # V = V_V / V_scale_V
    # I = I_A / I_scale_A
    # H = convert_temperature(T_degC, "Celsius", "Kelvin") / T_K_0
    # data = Data(numpy.vstack((V, I, F, H)), 1)

    # # Inline these functions here for transformed model.
    # def fun(beta, x):
    #     # V: x[0]
    #     # I: x[1]
    #     # F: x[2]
    #     # H: x[3]

    #     # Diode voltage and modified ideality factor.
    #     I_sc = x[2] * beta[0]
    #     V_diode = x[0] + x[1] * beta[3]
    #     n_mod = beta[2] * scaled_thermal_voltage_V_0 * x[3]

    #     # Auxiliary equations.
    #     I_rs = (
    #         numpy.exp(beta[1])
    #         * x[3] ** 3
    #         * numpy.exp(beta[5] / beta[2] * (1 - 1 / x[3]))
    #     )
    #     I_ph = I_rs * numpy.expm1(I_sc * beta[3] / n_mod) + I_sc * (
    #         beta[4] * beta[3] + 1
    #     )

    #     return I_ph - I_rs * numpy.expm1(V_diode / n_mod) - beta[4] * V_diode - x[1]

    # Construct IC vector for fit. Includes implicit unit-valued conversion factor in
    # band gap with units V/eV.
    # beta0 = numpy.array(
    #     [
    #         fit_prep_result["model_params_ic"]["I_sc_A_0"] / I_scale_A,
    #         numpy.log(fit_prep_result["model_params_ic"]["I_rs_1_A_0"] / I_scale_A),
    #         fit_prep_result["model_params_ic"]["n_1_0"],
    #         fit_prep_result["model_params_ic"]["R_s_Ohm_0"] * I_scale_A / V_scale_V,
    #         fit_prep_result["model_params_ic"]["G_p_S_0"] * V_scale_V / I_scale_A,
    #         fit_prep_result["model_params_ic"]["E_g_eV_0"] / E_g_scale_eV,
    #     ]
    # )

    data = Data(numpy.vstack((V_V, I_A, F, T_degC)), 1)

    # Inline these functions here for transformed model.
    def fun(beta, x):
        # Diode voltage and modified ideality factor.
        I_sc_A_0 = beta[0]
        I_rs_1_A_0 = beta[1]
        n_1_0 = beta[2]
        R_s_Ohm_0 = beta[3]
        G_p_S = beta[4]
        E_g_eV_0 = beta[5]

        T_K = convert_temperature(x[3], "Celsius", "Kelvin")
        I_sc_A = x[2] * I_sc_A_0
        V_diode_V = x[0] + x[1] * R_s_Ohm_0
        I_rs_1_A = (
            numpy.exp(I_rs_1_A_0)
            * (T_K / T_K_0) ** 3
            * numpy.exp(E_g_eV_0 / (n_1_0 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
        )
        I_ph_A = (
            I_rs_1_A
            * numpy.expm1(
                (q_C * I_sc_A * R_s_Ohm_0) / (N_s * n_1_0 * k_B_J_per_K * T_K)
            )
            + G_p_S * I_sc_A * R_s_Ohm_0
            + I_sc_A
        )

        return (
            I_ph_A
            - I_rs_1_A
            * numpy.expm1((q_C * V_diode_V) / (N_s * n_1_0 * k_B_J_per_K * T_K))
            - G_p_S * V_diode_V
            - x[1]
        )

    model = Model(fun, implicit=True)

    beta0 = numpy.array(
        [
            fit_prep_result["model_params_ic"]["I_sc_A_0"],
            numpy.log(fit_prep_result["model_params_ic"]["I_rs_1_A_0"]),
            fit_prep_result["model_params_ic"]["n_1_0"],
            fit_prep_result["model_params_ic"]["R_s_Ohm_0"],
            fit_prep_result["model_params_ic"]["G_p_S_0"],
            fit_prep_result["model_params_ic"]["E_g_eV_0"],
        ]
    )

    params = ("I_sc_A_0", "I_rs_1_A_0", "n_1_0", "R_s_Ohm_0", "G_p_S_0", "E_g_eV_0")
    ifixb = [
        int(fit_prep_result["model_params_fixed"][param] is False) for param in params
    ]

    # Compute fit. CAUTION: This solution is for a transformed problem!
    output = ODR(data, model, beta0=beta0, ifixb=ifixb, maxit=1000).run()

    # Do not allow negative G_p_S_0.
    if output.beta[4] < 0:
        ifixb[4] = 0
        beta0[4] = 0
        output = ODR(data, model, beta0=beta0, ifixb=ifixb, maxit=1000).run()

    # Transform back fit values.
    model_params_fit = {
        "I_sc_A_0": output.beta[0],  # * I_scale_A,
        "I_rs_1_A_0": numpy.exp(output.beta[1]),  # * I_scale_A,
        "n_1_0": output.beta[2],
        "R_s_Ohm_0": output.beta[3],  # * V_scale_V / I_scale_A,
        "G_p_S_0": output.beta[4],  # * I_scale_A / V_scale_V,
        "E_g_eV_0": output.beta[5],  # * E_g_scale_eV,
        "N_s": N_s,
        "T_degC_0": T_degC_0,
    }

    return {
        **fit_prep_result,
        "model_params_fit": model_params_fit,
        "I_sum_A_fit": sdm.current_sum_at_diode_node(
            V_V=V_V, I_A=I_A, F=F, T_degC=T_degC, **model_params_fit
        )["I_sum_A"],
        "iv_params_fit_0": sdm.iv_params(F=1, T_degC=T_degC_0, **model_params_fit),
        "output": output,
    }


def fit_F_T(
    *,
    V_V: numpy.ndarray,
    I_A: numpy.ndarray,
    N_s: int,
    I_sc_A_0: float,
    I_rs_1_A_0: float,
    n_1_0: float,
    R_s_Ohm_0: float,
    G_p_S_0: float,
    E_g_eV_0: float,
    T_degC_0: float,
    oc_params_fixed: Optional[dict] = None,
    oc_params_ic: Optional[dict] = None,
):
    """
    Fit the effective irradiance ratio and temperature of a PV device with a calibrated SDM from I-V curve data at a
    single operating condition.

    Inputs:
    Observables at operating condition (device-level):
        V_V terminal voltage
        I_A terminal current
            Assumes that V_V, I_A are rank-1 float64 numpy arrays with the same size, with minimally a point near Isc
            and a point near Voc.
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
            oc_params_ic model parameters at ICs, provided or estimated from I-V curve observables
            I_sum_A_ic residuals for current sum at diode's anode node for model calculated using oc_params_ic
            iv_params_ic I-V curve parameters calculated at OC using oc_params_ic
            oc_params_fit model parameters from fit algorithm starting at oc_params_ic
            I_sum_A_fit residuals for current sum at diode's anode node for model calculated using oc_params_fit
            iv_params_fit I-V curve parameters calculated at OC using oc_params_fit
            oc_params_fixed model parameters that were fixed during fit
            sol full solution object for fit returned by the solver
    """
    # TODO Add parameter fixing.
    model_params = {
        "N_s": N_s,
        "T_degC_0": T_degC_0,
        "I_sc_A_0": I_sc_A_0,
        "I_rs_1_A_0": I_rs_1_A_0,
        "n_1_0": n_1_0,
        "R_s_Ohm_0": R_s_Ohm_0,
        "G_p_S_0": G_p_S_0,
        "E_g_eV_0": E_g_eV_0,
    }

    # Check for provided ICs, and provide a default value for missing items.
    if oc_params_ic is None:
        oc_params_ic = {}
    oc_params_ic_default = {"F": None, "T_degC": None}
    oc_params_ic_default.update(oc_params_ic)
    oc_params_ic = oc_params_ic_default

    # Check for provided fixed params, and provide a default value for missing items.
    if oc_params_fixed is None:
        oc_params_fixed = {}
    oc_params_fixed_default = {"F": False, "T_degC": False}
    oc_params_fixed_default.update(oc_params_fixed)
    oc_params_fixed = oc_params_fixed_default

    # F IC.
    if oc_params_ic["F"] is None:
        # To estimate I_sc_A, get I_A at index of V_V that closest in absolute value to 0.
        I_sc_A_est = I_A[numpy.argmin(numpy.abs(V_V))]
        oc_params_ic["F"] = I_sc_A_est / I_sc_A_0
        if oc_params_ic["F"] <= 0 or not numpy.isfinite(oc_params_ic["F"]):
            raise ValueError(
                f"Computed initial condition for effective irradiance ratio, F = {oc_params_ic['F']}, is \
not strictly posiitve and finite."
            )
    elif oc_params_ic["F"] <= 0 or not numpy.isfinite(oc_params_ic["F"]):
        raise ValueError(
            f"Provided initial condition for effective irradiance ratio, F = {oc_params_ic['F']}, is not \
strictly positive and finite."
        )

    # T_degC IC.
    if oc_params_ic["T_degC"] is None:
        oc_params_ic["T_degC"] = least_squares(
            lambda x: sdm.current_sum_at_diode_node(
                V_V=V_V, I_A=I_A, F=oc_params_ic["F"], T_degC=x, **model_params
            )["I_sum_A"],
            T_degC_0,
        ).x.item()
        if oc_params_ic["T_degC"] <= T_degC_abs_zero or not numpy.isfinite(
            oc_params_ic["T_degC"]
        ):
            raise ValueError(
                f"Computed initial condition for device temperature, T_degC = {oc_params_ic['T_degC']}, \
is not greater than absolute zero and finite."
            )
    elif oc_params_ic["T_degC"] <= T_degC_abs_zero or not numpy.isfinite(
        oc_params_ic["T_degC"]
    ):
        raise ValueError(
            f"Provided initial condition for device temperature, T_degC = {oc_params_ic['T_degC']}, is \
not greater than absolute zero and finite."
        )

    data = Data(numpy.vstack((V_V, I_A)), 1)

    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")

    # Inline these functions here for transformed model.
    def fun(beta, x):
        # Diode voltage and modified ideality factor.
        F = beta[0]
        T_K = convert_temperature(beta[1], "Celsius", "Kelvin")

        V_diode_V = x[0] + x[1] * R_s_Ohm_0
        I_rs_1_A = (
            I_rs_1_A_0
            * (T_K / T_K_0) ** 3
            * numpy.exp(E_g_eV_0 / (n_1_0 * k_B_eV_per_K) * (1 / T_K_0 - 1 / T_K))
        )

        I_sc_A = F * I_sc_A_0

        I_ph_A = (
            I_rs_1_A
            * numpy.expm1(
                (q_C * I_sc_A * R_s_Ohm_0) / (N_s * n_1_0 * k_B_J_per_K * T_K)
            )
            + G_p_S_0 * I_sc_A * R_s_Ohm_0
            + I_sc_A
        )

        return (
            I_ph_A
            - I_rs_1_A
            * numpy.expm1((q_C * V_diode_V) / (N_s * n_1_0 * k_B_J_per_K * T_K))
            - G_p_S_0 * V_diode_V
            - x[1]
        )

    model = Model(fun, implicit=True)

    beta0 = numpy.array(
        [
            oc_params_ic["F"],
            oc_params_ic["T_degC"],
        ]
    )

    ifixb = [int(oc_params_fixed[param] is False) for param in oc_params_ic.keys()]

    # Compute fit.
    output = ODR(data, model, beta0=beta0, ifixb=ifixb, maxit=1000).run()

    # Do not allow negative F. (TBD if this actually works.)
    if output.beta[0] < 0:
        ifixb[0] = 0
        beta0[0] = 0
        output = ODR(data, model, beta0=beta0, ifixb=ifixb, maxit=1000).run()

    oc_params_fit = {"F": output.beta[0], "T_degC": output.beta[1]}

    # # Construct IC vector for fit.
    # x0 = numpy.array([oc_params_ic["F"], oc_params_ic["T_degC"]])

    # # Compute fit.
    # sol = least_squares(
    #     lambda x: sdm.current_sum_at_diode_node(
    #         V_V=V_V, I_A=I_A, F=x[0], T_degC=x[1], **model_params
    #     )["I_sum_A"],
    #     x0,
    #     method="dogbox",
    #     jac="3-point",
    #     bounds=(
    #         [numpy.finfo(float).tiny, numpy.nextafter(T_degC_abs_zero, 1)],
    #         numpy.inf,
    #     ),
    #     max_nfev=1000 * x0.size,
    # )

    # oc_params_fit = {"F": sol.x[0], "T_degC": sol.x[1]}

    # oc_params_fixed = {"F": False, "T_degC": False}

    return {
        "oc_params_ic": oc_params_ic,
        "I_sum_A_ic": sdm.current_sum_at_diode_node(
            V_V=V_V, I_A=I_A, **oc_params_ic, **model_params
        )["I_sum_A"],
        "iv_params_ic": sdm.iv_params(**oc_params_ic, **model_params),
        "oc_params_fit": oc_params_fit,
        "I_sum_A_fit": sdm.current_sum_at_diode_node(
            V_V=V_V, I_A=I_A, **oc_params_fit, **model_params
        )["I_sum_A"],
        "iv_params_fit": sdm.iv_params(**oc_params_fit, **model_params),
        "oc_params_fixed": oc_params_fixed,
        "output": output,
    }


def fit_F_T_R_s_G_p(
    *,
    V_V,
    I_A,
    N_s,
    T_degC_0,
    I_sc_A_0,
    I_rs_1_A_0,
    n_1_0,
    E_g_eV_0,
    oc_params_ic=None,
    model_params_ic=None,
):
    """
    Use least squares to fit the effective irradiance ratio, device temperature, series resistance at reference
    condition (RC), and parallel conductance at RC of a PV device with a calibrated SDM from I-V curve data at an
    unknown operating condition (OC).

    Inputs (device-level):
    Observables at OC:
        V_V terminal voltage
        I_A terminal current
            Assumes that V_V, I_A are rank-1 float64 numpy arrays with the same size, with minimally a point near Isc, a
            point near Voc, and two distinct points in between.
    Model parameters at RC:
        Non-fitted device parameters:
            N_s integer number of cells in series in each parallel string
            T_degC_0 temperature
        Previously calibrated model parameters (device-level):
            I_sc_A_0 short-circuit current
            I_rs_1_A_0 diode reverse-saturation current
            n_1_0 diode ideality factor
            E_g_eV_0 material band gap
        Initial conditions (ICs) for remaining fit parameters, default value (None) means compute from I-V data:
            oc_params_ic dictionary (device-level) at OC
                F effective irradiance ratio
                T_degC temperature
            model_params_ic dictionary (device-level) at RC
                R_s_Ohm_0 series resistance
                G_p_S_0 parallel (shunt) conductance

    Outputs (device-level):
        dict containing:
            oc_params_ic model parameters at ICs, provided or estimated from I-V curve observables
            model_params_ic model parameters at ICs, provided or estimated from I-V curve observables
            I_sum_A_ic residuals for current sum at diode's anode node for model calculated using oc_params_ic and
                model_params_ic
            iv_params_ic I-V curve parameters calculated at OC using oc_params_oc and and model_params_ic
            model_params_fit model parameters from fit algorithm starting at oc_params_oc and and model_params_ic
            I_sum_A_fit residuals for current sum at diode's anode node for model calculated using oc_params_fit and
                model_params_fit
            iv_params_fit I-V curve parameters calculated at OC using oc_params_fit and model_params_fit
            oc_params_fixed operating condition parameters fixed during fit
            model_params_fixed model parameters fixed during fit
            sol full solution object for fit returned by the solver
    """
    # TODO Add parameter fixing.
    model_params = {
        "N_s": N_s,
        "T_degC_0": T_degC_0,
        "I_sc_A_0": I_sc_A_0,
        "I_rs_1_A_0": I_rs_1_A_0,
        "n_1_0": n_1_0,
        "E_g_eV_0": E_g_eV_0,
    }

    # Compute ICs for F and T_degC by assuming ideal series resistance and parallel conductance.
    # TODO Improve this by using non-degraded values, if available?
    fit_F_T_result = fit_F_T(
        V_V=V_V,
        I_A=I_A,
        **model_params,
        R_s_Ohm_0=0,
        G_p_S_0=0,
        oc_params_ic=oc_params_ic,
    )
    oc_params_ic = fit_F_T_result["oc_params_fit"]

    # Check for provided ICs, and provide a default value for missing items.
    if model_params_ic is None:
        model_params_ic = {}
    model_params_ic_default = {"R_s_Ohm_0": None, "G_p_S_0": None}
    model_params_ic_default.update(model_params_ic)
    model_params_ic_default.update(model_params)
    model_params_ic = model_params_ic_default

    # Compute any needed ICs for R_s_Ohm_0 and G_p_S_0 using SDE now that T_deg is surely estimated.
    sde_fit_result = sde_fit(
        V_V=V_V, I_A=I_A, N_s=N_s, T_degC=fit_F_T_result["oc_params_fit"]["T_degC"]
    )

    if model_params_ic["R_s_Ohm_0"] is None:
        model_params_ic["R_s_Ohm_0"] = sde_fit_result["model_params_fit"]["R_s_Ohm"]
        if model_params_ic["R_s_Ohm_0"] < 0 or not numpy.isfinite(
            model_params_ic["R_s_Ohm_0"]
        ):
            raise ValueError(
                f"Computed initial condition for series resistance at reference condition, R_s_Ohm_0 = \
{model_params_ic['R_s_Ohm_0']}, is not non-negative and finite."
            )
    elif model_params_ic["R_s_Ohm_0"] < 0 or not numpy.isfinite(
        model_params_ic["R_s_Ohm_0"]
    ):
        raise ValueError(
            f"Provided initial condition for series resistance at reference condition, R_s_Ohm_0 = \
{model_params_ic['R_s_Ohm_0']}, is not non-negative and finite."
        )

    if model_params_ic["G_p_S_0"] is None:
        model_params_ic["G_p_S_0"] = sde_fit_result["model_params_fit"]["G_p_S"]
        if model_params_ic["G_p_S_0"] < 0 or not numpy.isfinite(
            model_params_ic["G_p_S_0"]
        ):
            raise ValueError(
                f"Computed initial condition for parallel conductance at reference condition, G_p_S_0 = \
{model_params_ic['G_p_S_0']}, is not non-negative and finite."
            )
    elif model_params_ic["G_p_S_0"] < 0 or not numpy.isfinite(
        model_params_ic["G_p_S_0"]
    ):
        raise ValueError(
            f"Provided initial condition for parallel conductance at reference condition, G_p_S_0 = \
{model_params_ic['G_p_S_0']}, is not non-negative and finite."
        )

    # Construct IC vector for fit.
    x0 = numpy.array(
        [
            oc_params_ic["F"],
            oc_params_ic["T_degC"],
            model_params_ic["R_s_Ohm_0"],
            model_params_ic["G_p_S_0"],
        ]
    )

    # Compute fit.
    sol = least_squares(
        lambda x: sdm.current_sum_at_diode_node(
            V_V=V_V,
            I_A=I_A,
            F=x[0],
            T_degC=x[1],
            **model_params,
            R_s_Ohm_0=x[2],
            G_p_S_0=x[3],
        )["I_sum_A"],
        x0,
        method="dogbox",
        jac="3-point",
        bounds=(
            [numpy.finfo(float).tiny, numpy.nextafter(T_degC_abs_zero, 1), 0.0, 0.0],
            numpy.inf,
        ),
        max_nfev=1000 * x0.size,
    )

    oc_params_fit = {"F": sol.x[0], "T_degC": sol.x[1]}

    oc_params_fixed = {"F": False, "T_degC": False}

    model_params_fit = {
        "N_s": N_s,
        "T_degC_0": T_degC_0,
        "I_sc_A_0": I_sc_A_0,
        "I_rs_1_A_0": I_rs_1_A_0,
        "n_1_0": n_1_0,
        "R_s_Ohm_0": sol.x[2],
        "G_p_S_0": sol.x[3],
        "E_g_eV_0": E_g_eV_0,
    }

    model_params_fixed = {
        "N_s": True,
        "T_degC_0": True,
        "I_sc_A_0": True,
        "I_rs_1_A_0": True,
        "n_1_0": True,
        "R_s_Ohm_0": False,
        "G_p_S_0": False,
        "E_g_eV_0": True,
    }

    return {
        "oc_params_ic": oc_params_ic,
        "model_params_ic": model_params_ic,
        "I_sum_A_ic": sdm.current_sum_at_diode_node(
            V_V=V_V, I_A=I_A, **oc_params_ic, **model_params_ic
        )["I_sum_A"],
        "iv_params_ic": sdm.iv_params(**oc_params_ic, **model_params_ic),
        "oc_params_fit": oc_params_fit,
        "model_params_fit": model_params_fit,
        "I_sum_A_fit": sdm.current_sum_at_diode_node(
            V_V=V_V, I_A=I_A, **oc_params_fit, **model_params_fit
        )["I_sum_A"],
        "iv_params_fit": sdm.iv_params(**oc_params_fit, **model_params_fit),
        "oc_params_fixed": oc_params_fixed,
        "model_params_fixed": model_params_fixed,
        "sol": sol,
    }


def fit_I_sc_V_oc(
    *,
    V_oc_V,
    I_sc_A,
    N_s,
    T_degC_0,
    I_sc_A_0=None,
    E_g_eV_0=None,
    model_params_ic=None,
    model_params_fixed=None,
    material=None,
):
    """
    Fit zero-series-resistance Isc-Voc curve at reference temperature.

    Algorithm fits only I_rs_A_0, n_1_0, and G_p_S_0 while fixing R_s_Ohm_A at zero.

    Algorithm cannot know which value in I_sc_A is the reference-condition value, if any, so user should pass I_sc_A_0
    if known. Otherwise, update model_params_fit['I_sc_A_0'] in result before using in subsequent functions. Likewise
    for the material band gap at reference condition, E_g_eV_0, which can be specified (optionally) using material.

    TODO
    """
    # Checks on parameters that must be provided.
    if not isinstance(N_s, int) or N_s < 0:
        raise ValueError(
            f"Provided value for number of cells in each parallel string, N_s = {N_s}, is not a strictly \
positive integer."
        )
    if T_degC_0 <= T_degC_abs_zero:
        raise ValueError(
            f"Provided value for reference temperature, T_degC = {T_degC_0}, is not greater than absolute zero."
        )

    # Check for provided fit parameters to fix, and provide a default value for missing items.
    if model_params_fixed is None:
        model_params_fixed = {}
    model_params_fixed_default = {"I_rs_1_A_0": False, "n_1_0": False, "G_p_S_0": False}
    model_params_fixed_default.update(model_params_fixed)
    model_params_fixed = model_params_fixed_default
    # These are always fixed.
    model_params_fixed.update(
        {
            "N_s": True,
            "T_degC_0": True,
            "I_sc_A_0": True,
            "R_s_Ohm_0": True,
            "E_g_eV_0": True,
        }
    )

    # Check for provided ICs, and provide a default value for missing items.
    if model_params_ic is None:
        model_params_ic = {}
    model_params_ic_default = {"I_rs_1_A_0": None, "n_1_0": None, "G_p_S_0": None}
    model_params_ic_default.update(model_params_ic)
    model_params_ic = model_params_ic_default
    # These are included as part of the fit and to ease subsequent function calls.
    model_params_ic.update(
        {
            "N_s": N_s,
            "T_degC_0": T_degC_0,
            "I_sc_A_0": I_sc_A_0,
            "R_s_Ohm_0": 0.0,
            "E_g_eV_0": E_g_eV_0,
        }
    )

    # Value for I_sc_A_0.
    if model_params_ic["I_sc_A_0"] is not None and model_params_ic["I_sc_A_0"] <= 0:
        raise ValueError(
            f"Provided value for short-circuit current at reference condition, I_sc_A_0 = \
{model_params_ic['I_sc_A_0']}, is not strictly positive."
        )

    # Value for E_g_eV_0.
    if model_params_ic["E_g_eV_0"] is None:
        if material in materials:
            # This material is recognized.
            model_params_ic["E_g_eV_0"] = materials[material]["E_g_eV_stc"]
    elif model_params_ic["E_g_eV_0"] <= 0:
        raise ValueError(
            f"Provided value for material band gap at reference condition, E_g_eV_0 = \
{model_params_ic['E_g_eV_0']}, is not strictly positive."
        )

    # Optimization.
    T_K_0 = convert_temperature(T_degC_0, "Celsius", "Kelvin")
    scaled_thermal_voltage_V_0 = N_s * k_B_J_per_K * T_K_0 / q_C

    # I_rs_1_A_0 IC and n_1_0 IC, which assumes that G_p_S_0 is zero.
    if model_params_ic["I_rs_1_A_0"] is None and model_params_ic["n_1_0"] is None:
        slope, intercept, _, _, _ = linregress(V_oc_V, y=numpy.log(I_sc_A))
        model_params_ic["I_rs_1_A_0"] = numpy.exp(intercept)
        if model_params_ic["I_rs_1_A_0"] <= 0 or not numpy.isfinite(
            model_params_ic["I_rs_1_A_0"]
        ):
            raise ValueError(
                f"Calculated initial condition for reverse-saturation current at reference condition, \
I_rs_1_A_0 = {model_params_ic['I_rs_1_A_0']}, is not strictly positive and finite."
            )
        model_params_ic["n_1_0"] = 1 / (slope * scaled_thermal_voltage_V_0)
        if model_params_ic["n_1_0"] <= 0 or not numpy.isfinite(
            model_params_ic["n_1_0"]
        ):
            raise ValueError(
                f"Calculated initial condition for diode ideality factor at reference condition, n_1_0 = \
{model_params_ic['n_1_0']}, is not strictly positive and finite."
            )
    elif model_params_ic["I_rs_1_A_0"] is None:
        _, intercept, _, _, _ = linregress(V_oc_V, y=numpy.log(I_sc_A))
        model_params_ic["I_rs_1_A_0"] = numpy.exp(intercept)
        if model_params_ic["I_rs_1_A_0"] <= 0 or not numpy.isfinite(
            model_params_ic["I_rs_1_A_0"]
        ):
            raise ValueError(
                f"Calculated initial condition for reverse-saturation current at reference condition, \
I_rs_1_A_0 = {model_params_ic['I_rs_1_A_0']}, is not strictly positive and finite."
            )
        model_params_ic["n_1_0"] = 1 / (slope * scaled_thermal_voltage_V_0)
    elif model_params_ic["n_1_0"] is None:
        slope, _, _, _, _ = linregress(V_oc_V, y=numpy.log(I_sc_A))
        model_params_ic["n_1_0"] = 1 / (slope * scaled_thermal_voltage_V_0)
        if model_params_ic["n_1_0"] <= 0 or not numpy.isfinite(
            model_params_ic["n_1_0"]
        ):
            raise ValueError(
                f"Calculated initial condition for diode ideality factor at reference condition, n_1_0 = \
{model_params_ic['n_1_0']}, is not strictly positive and finite."
            )
    else:
        if model_params_ic["I_rs_1_A_0"] <= 0 or not numpy.isfinite(
            model_params_ic["I_rs_1_A_0"]
        ):
            raise ValueError(
                f"Provided initial condition for reverse-saturation current at reference condition, \
I_rs_1_A_0 = {model_params_ic['I_rs_1_A_0']}, is not strictly positive and finite."
            )
        if model_params_ic["n_1_0"] <= 0 or not numpy.isfinite(
            model_params_ic["n_1_0"]
        ):
            raise ValueError(
                f"Provided initial condition for diode ideality factor, n_1_0 = \
{model_params_ic['n_1_0']}, is not strictly positive and finite."
            )

    # G_p_S_0 IC.
    if model_params_ic["G_p_S_0"] is None:
        # TODO By construction above, does this usually give zero?
        model_params_ic["G_p_S_0"] = max(
            0.0,
            numpy.mean(
                (
                    I_sc_A
                    - model_params_ic["I_rs_1_A_0"]
                    * numpy.expm1(V_oc_V / model_params_ic["n_1_0"])
                )
                / V_oc_V
            ),
        )
    elif model_params_ic["G_p_S_0"] < 0 or not numpy.isfinite(
        model_params_ic["G_p_S_0"]
    ):
        raise ValueError(
            f"Provided initial condition for parallel (shunt) conductance at reference condition, \
G_p_S_0 = {model_params_ic['G_p_S_0']}, is not strictly positive and finite."
        )

    # Construct IC vector for fit.
    x0 = numpy.array(
        [
            numpy.log(model_params_ic["I_rs_1_A_0"]),
            scaled_thermal_voltage_V_0 * model_params_ic["n_1_0"],
            model_params_ic["G_p_S_0"],
        ]
    )

    # Inline these functions here for transformed model, with closures over data.
    def fun(x):
        return I_sc_A - numpy.exp(x[0]) * numpy.expm1(V_oc_V / x[1]) - x[2] * V_oc_V

    jac = "3-point"
    # def jac(x):
    #     # Reused expressions.
    #     exp_x1 = numpy.exp(x[1])
    #     H_cubed = H**3
    #     V_diode = V + I * x[3]
    #     neg_inv_n = -1 / x[2]
    #     n_mod = x[2] * H
    #     factor_1 = (H - 1) / n_mod
    #     factor_2 = x[5] * factor_1
    #     factor_3 = numpy.exp(factor_2)
    #     factor_4 = F * x[3] / n_mod
    #     factor_5 = x[0] * factor_4
    #     factor_6 = numpy.exp(factor_5)
    #     factor_7 = x[3] * x[4] + 1
    #     factor_8 = exp_x1 * H_cubed * factor_3
    #     factor_9 = V_diode / n_mod
    #     factor_10 = numpy.exp(factor_9)
    #     factor_11 = factor_6 - 1 - factor_10
    #     factor_12 = F * x[0]
    #     factor_13 = factor_12 / n_mod

    #     # Component-wise derivatives. Fix parameters when requested by setting derivative to zero.
    #     x0_d1 = (not I_sc_A_0_fixed) * (factor_8 * factor_6 * factor_4 + F * factor_7)
    #     x1_d1 = (not I_rs_1_A_0_fixed) * (factor_8 * factor_11)
    #     x2_d1 = (not n_1_0_fixed) * (
    #         factor_8 * neg_inv_n * (factor_2 * factor_11 + (factor_6 * factor_5 - factor_10 * factor_9)))
    #     x3_d1 = (not R_s_Ohm_0_fixed) * (
    #         factor_8 * (factor_6 * factor_13 - factor_10 * I / n_mod) + x[4] * (factor_12 - I))
    #     x4_d1 = (not G_p_S_0_fixed) * (factor_12 * x[3] - V_diode)
    #     x5_d1 = (not E_g_eV_0_fixed) * (factor_8 * factor_1 * factor_11)

    #     return numpy.vstack((x0_d1, x1_d1, x2_d1, x3_d1, x4_d1, x5_d1)).T

    # Compute fit.
    # CAUTION: This solution is for a transformed problem!
    sol = least_squares(
        fun,
        x0,
        method="dogbox",
        jac=jac,
        max_nfev=10000 * x0.size,
        bounds=([-numpy.inf, numpy.finfo(float).tiny, 0.0], numpy.inf),
    )

    # Transform back fit values.
    # Ensure that fixed parameters are actually fixed to IC (remove any side effects of tranformations).
    model_params_fit = {
        "N_s": model_params_ic["N_s"],
        "T_degC_0": model_params_ic["T_degC_0"],
        "I_sc_A_0": model_params_ic["I_sc_A_0"],
        "I_rs_1_A_0": (not model_params_fixed["I_rs_1_A_0"]) * numpy.exp(sol.x[0])
        + model_params_fixed["I_rs_1_A_0"] * model_params_ic["I_rs_1_A_0"],
        "n_1_0": (not model_params_fixed["n_1_0"])
        * sol.x[1]
        / scaled_thermal_voltage_V_0
        + model_params_fixed["n_1_0"] * model_params_ic["n_1_0"],
        "R_s_Ohm_0": 0.0,
        "G_p_S_0": (not model_params_fixed["G_p_S_0"]) * sol.x[2]
        + model_params_fixed["G_p_S_0"] * model_params_ic["G_p_S_0"],
        "E_g_eV_0": model_params_ic["E_g_eV_0"],
    }

    return {
        "model_params_ic": model_params_ic,
        "model_params_fit": model_params_fit,
        "I_sum_A_fit": fun(sol.x),
        "model_params_fixed": model_params_fixed,
        "sol": sol,
    }
