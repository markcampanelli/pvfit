"""
PVfit: Getting-started demo for single-diode model (SDM) parameter inference from
spec-sheet information.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import importlib.resources
import os
from pprint import pprint
from typing import OrderedDict

from matplotlib import pyplot
import numpy
from numpy.polynomial import Polynomial
import pandas

from pvfit.common import E_hemispherical_tilted_W_per_m2_stc, T_degC_stc
from pvfit.measurement.iv.types import FTData, IVPerformanceMatrix, SpecSheetParameters
from pvfit.modeling.dc.common import Material
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_simple_ae
import pvfit.modeling.dc.single_diode.model.simple.inference_spec_sheet as sdm_simple_inf_spec_sheet
from pvfit.types import NewtonOptions

# By convention, variable names for numeric values include the units.

# Load PV module data.
# See https://pvpmc.sandia.gov/pv-research/pv-lifetime-project/pv-lifetime-modules/
datafile = "Sandia_PV_Module_P-Matrix-and-TempCo-Data_2019.xlsx"
path = importlib.resources.files("pvfit.demos.sdm.data") / datafile
sheets_matrix = pandas.read_excel(
    path, sheet_name=None, header=tuple(range(5)), skipfooter=16
)
sheets_coeffs = pandas.read_excel(path, sheet_name=None, header=tuple(range(35)))
N_s = {
    "19074-001": 96,
    "19074-002": 60,
    "19074-003": 60,
    "19074-004": 60,
    "19074-005": 60,
    "19074-006": 60,
    "19074-007": 60,
    "19074-008": 60,
    "19074-009": 72,
}
cell_tech = {
    "19074-001": "HIT Mono",
    "19074-002": "N-PERT Si",
    "19074-003": "poly-Si PERC",
    "19074-004": "poly-Si",
    "19074-005": "mono-Si",
    "19074-006": "poly-Si",
    "19074-007": "mono-Si PERC",
    "19074-008": "mono-Si PERC",
    "19074-009": "mono-Si PERC",
}

sheet_key = os.environ.get("PVFIT_SHEET_KEY", "19074-002")
sheet_key_found = False

for sheet_key_full, sheet_matrix_value in sheets_matrix.items():
    if sheet_key in sheet_key_full:
        sheet_key_found = True
        break

if not sheet_key_found:
    raise ValueError(f"{sheet_key} not found in {path}")

iv_performance_matrix = IVPerformanceMatrix(
    material=Material.xSi,  # Generic choice.
    N_s=N_s[sheet_key],
    I_sc_A=sheet_matrix_value.iloc[:, 3].to_numpy(),
    I_mp_A=sheet_matrix_value.iloc[:, 5].to_numpy(),
    V_mp_V=sheet_matrix_value.iloc[:, 6].to_numpy(),
    V_oc_V=sheet_matrix_value.iloc[:, 4].to_numpy(),
    E_W_per_m2=sheet_matrix_value.iloc[:, 2].to_numpy(),
    T_degC=sheet_matrix_value.iloc[:, 1].to_numpy(),
    E_W_per_m2_0=E_hemispherical_tilted_W_per_m2_stc,
    T_degC_0=T_degC_stc,
)

# Calculate coefficients that would be found on spec sheet.
sheet_coeffs_value = sheets_coeffs[sheet_key_full]
T_degC_0_coef_data = sheet_coeffs_value.iloc[:, 1]

I_sc_A_0_coef_data = sheet_coeffs_value.iloc[:, 3]
dI_sc_dT_A_per_degC_0 = float(
    Polynomial.fit(T_degC_0_coef_data, I_sc_A_0_coef_data, deg=1).convert().coef[1]
)

P_mp_W_0_coef_data = sheet_coeffs_value.iloc[:, 7]
dP_mp_dT_W_per_degC_0 = float(
    Polynomial.fit(T_degC_0_coef_data, P_mp_W_0_coef_data, deg=1).convert().coef[1]
)

V_oc_V_0_coef_data = sheet_coeffs_value.iloc[:, 4]
dV_oc_dT_V_per_degC_0 = float(
    Polynomial.fit(T_degC_0_coef_data, V_oc_V_0_coef_data, deg=1).convert().coef[1]
)

spec_sheet_parameters = SpecSheetParameters(
    material=iv_performance_matrix.material,
    N_s=iv_performance_matrix.N_s,
    I_sc_A_0=iv_performance_matrix.I_sc_A_0,
    I_mp_A_0=iv_performance_matrix.I_mp_A_0,
    V_mp_V_0=iv_performance_matrix.V_mp_V_0,
    V_oc_V_0=iv_performance_matrix.V_oc_V_0,
    dI_sc_dT_A_per_degC_0=dI_sc_dT_A_per_degC_0,
    dP_mp_dT_W_per_degC_0=dP_mp_dT_W_per_degC_0,
    dV_oc_dT_V_per_degC_0=dV_oc_dT_V_per_degC_0,
    E_W_per_m2_0=iv_performance_matrix.E_W_per_m2_0,
    T_degC_0=iv_performance_matrix.T_degC_0,
)

print(
    f"{sheet_key_full}: {cell_tech[sheet_key]} module with {iv_performance_matrix.N_s} "
    "cells in series."
)

print("\nI-V curve parameters at STC using specification-sheet data:")
pprint(
    OrderedDict(
        {
            "I_sc_A": spec_sheet_parameters.I_sc_A_0,
            "I_mp_A": spec_sheet_parameters.I_mp_A_0,
            "P_mp_W": spec_sheet_parameters.P_mp_W_0,
            "V_mp_V": spec_sheet_parameters.V_mp_V_0,
            "V_oc_V": spec_sheet_parameters.V_oc_V_0,
            "dI_sc_dT_A_per_degC_0": dI_sc_dT_A_per_degC_0,
            "dP_mp_dT_W_per_degC_0": dP_mp_dT_W_per_degC_0,
            "dV_oc_dT_V_per_degC_0": dV_oc_dT_V_per_degC_0,
        }
    )
)

# Alternative operating condition.
F_alt = 0.5
T_degC_alt = 35.0

# F and/or T_degC can be vectorized, such as for a time-series of weather data.
F_series = numpy.array([0.95, 0.97, 0.99, 1.01, 0.97, 0.98])
T_degC_series = 35.0  # This scalar value will be approapriately broadcast.

# Fit simple SDM to spec-sheet data.

# model_parameters has both fittable and unfittable parameters.
# Additional outputs can be useful, but ignored here.
print("\nFitting model parameters to specification datasheet...")
model_parameters = sdm_simple_inf_spec_sheet.fit(
    spec_sheet_parameters=spec_sheet_parameters,
).model_parameters
print("Fitting model parameters to specification datasheet...done")

mape_mbpe, _ = sdm_simple_inf_spec_sheet.compute_fit_quality(
    iv_performance_matrix=iv_performance_matrix,
    model_parameters=model_parameters,
    newton_options=NewtonOptions(maxiter=1000),
)

print("\nModel parameters from fit to specification datasheet:")
pprint(model_parameters)
print("\nFit quality:")
pprint(mape_mbpe)

print("\nI-V curve parameters at STC using specification-datasheet fit:")

# Compute parameters for I-V curve at STC using auxiliary equations to compute the
# model parameters passed to the single-diode equation (SDE).
iv_curve_parameters_0 = sde_sim.iv_curve_parameters(
    model_parameters=sdm_simple_ae.compute_sde_model_parameters(
        ft_data=FTData(F=1.0, T_degC=spec_sheet_parameters.T_degC_0),
        model_parameters=model_parameters,
    ),
    newton_options=NewtonOptions(maxiter=1000),
)

print("\nI-V curve parameters at STC using spec-sheet fit to simple SDM:")
pprint(iv_curve_parameters_0)

# Save some fit I-V curve values for later.
I_sc_A_0 = iv_curve_parameters_0.I_sc_A
I_mp_A_0 = iv_curve_parameters_0.I_mp_A
V_mp_V_0 = iv_curve_parameters_0.V_mp_V
V_oc_V_0 = iv_curve_parameters_0.V_oc_V

# Compute at alternative operating condition.
iv_parameters_alt = sde_sim.iv_curve_parameters(
    model_parameters=sdm_simple_ae.compute_sde_model_parameters(
        ft_data=FTData(F=F_alt, T_degC=T_degC_alt),
        model_parameters=model_parameters,
    ),
    newton_options=NewtonOptions(maxiter=1000),
)

# Now make a nice comparison plot.
fig, ax = pyplot.subplots(1, 1, figsize=(12, 6))

# Plot Simple SDM to spec-sheet results.

# Create color cycler.
cycler = pyplot.rcParams["axes.prop_cycle"]()

# Plot the data fits.
for idx, (F, T_degC) in enumerate(
    zip(iv_performance_matrix.F, iv_performance_matrix.T_degC)
):
    # Plot Isc, Pmp, and Voc with same colors as fit lines.
    color = next(cycler)["color"]
    ax.plot(
        iv_performance_matrix.ivft_data.V_V[3 * idx : 3 * idx + 3],
        iv_performance_matrix.ivft_data.I_A[3 * idx : 3 * idx + 3],
        "o",
        color=color,
    )
    V_V = numpy.linspace(0, iv_performance_matrix.ivft_data.V_V[3 * idx + 2], 101)
    ax.plot(
        V_V,
        sde_sim.I_at_V(
            V_V=V_V,
            model_parameters=sdm_simple_ae.compute_sde_model_parameters(
                ft_data=FTData(F=F, T_degC=T_degC),
                model_parameters=model_parameters,
            ),
        )["I_A"],
    )

# Plot the LIC.
color = next(cycler)["color"]
ax.plot(
    [0.0, iv_parameters_alt.V_mp_V, iv_parameters_alt.V_oc_V],
    [iv_parameters_alt.I_sc_A, iv_parameters_alt.I_mp_A, 0.0],
    "*",
    color=color,
)
V_V = numpy.linspace(0, iv_parameters_alt.V_oc_V, 101)
ax.plot(
    V_V,
    sde_sim.I_at_V(
        V_V=V_V,
        model_parameters=sdm_simple_ae.compute_sde_model_parameters(
            ft_data=FTData(F=F_alt, T_degC=T_degC_alt),
            model_parameters=model_parameters,
        ),
    )["I_A"],
    "--",
    label=f"F={F_alt:.2f} suns, T={T_degC_alt:.0f} °C",
    color=color,
)
ax.legend(loc="lower left")
ax.set_xlabel("V (V)")
ax.set_ylabel("I (A)")
ax.set_title(
    f"{sheet_key_full} {cell_tech[sheet_key]}: 6-Parameter Simple Single-Diode Model "
    "(SDM) Fit to Spec-Sheet Data",
    fontdict={"fontsize": 14},
)
fig.tight_layout()

pyplot.show()
