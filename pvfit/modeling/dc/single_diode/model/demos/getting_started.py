"""
PVfit: Getting-started demo for single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from pprint import pprint

from matplotlib import pyplot
import numpy

from pvfit.measurement.iv.types import FTData
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.demos.data as data
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as sdm_ae
import pvfit.modeling.dc.single_diode.model.simple.inference_matrix as sdm_simple_inf_matrix
import pvfit.modeling.dc.single_diode.model.simple.inference_spec_sheet as sdm_simple_inf_spec_sheet


# By convention, variable names for numeric values include the units.

# Load PV module data.
iv_performance_matrix = data.MODULE["iv_performance_matrix"]
spec_sheet_parameters = data.MODULE["spec_sheet_parameters"]
ivft_data = iv_performance_matrix.ivft_data

print(
    f"{iv_performance_matrix.material.value} module with {iv_performance_matrix.N_s} "
    "cells in series."
)

print("\nI-V curve parameters at STC using specification-sheet data:")
pprint(
    {
        "I_sc_A": spec_sheet_parameters.I_sc_A_0,
        "I_mp_A": spec_sheet_parameters.I_mp_A_0,
        "P_mp_W": spec_sheet_parameters.P_mp_W_0,
        "V_mp_V": spec_sheet_parameters.V_mp_V_0,
        "V_oc_V": spec_sheet_parameters.V_oc_V_0,
    }
)

# Fit simple single-diode model (SDM) to I-V performance matrix data.

# model_parameters_matrix has both fittable and unfittable parameters, which can be
# used convienently in downstream functions, e.g., for power simulation.
# Additional outputs can be useful, but ignored here.
print("\nFitting model parameters to performance matrix...")
model_parameters_matrix = sdm_simple_inf_matrix.fit(
    iv_performance_matrix=iv_performance_matrix,
)["model_parameters"]
print("Fitting model parameters to performance matrix...done")

mape_mbpe_matrix, _ = sdm_simple_inf_matrix.compute_fit_quality(
    iv_performance_matrix=iv_performance_matrix,
    model_parameters=model_parameters_matrix,
)

print("\nModel parameters from fit to performance matrix:")
pprint(model_parameters_matrix)
print("\nFit quality:")
pprint(mape_mbpe_matrix)

# Compute parameters for I-V curve at STC using auxiliary equations to compute the
# model parameters passed to the single-diode equation (SDE).
iv_curve_parameters_0 = sde_sim.iv_curve_parameters(
    model_parameters=sdm_ae.compute_sde_model_parameters(
        ft_data=FTData(F=1.0, T_degC=iv_performance_matrix.T_degC_0),
        model_parameters=model_parameters_matrix,
    )
)

print("\nI-V curve parameters at STC using performance-matrix fit:")
pprint(iv_curve_parameters_0)

# Save some fit I-V curve values for later.
I_sc_A_0 = iv_curve_parameters_0["I_sc_A"]
I_mp_A_0 = iv_curve_parameters_0["I_mp_A"]
V_mp_V_0 = iv_curve_parameters_0["V_mp_V"]
V_oc_V_0 = iv_curve_parameters_0["V_oc_V"]

# Compute an alternative operating condition.
F_alt = 0.5
T_degC_alt = 35.0
iv_parameters_alt = sde_sim.iv_curve_parameters(
    model_parameters=sdm_ae.compute_sde_model_parameters(
        ft_data=FTData(F=F_alt, T_degC=T_degC_alt),
        model_parameters=model_parameters_matrix,
    )
)

print(f"\nAlternative operating condition: F={F_alt}, T={T_degC_alt} °C")
pprint(iv_parameters_alt)

# F and/or T_degC can be vectorized, such as for a time-series of weather data.
F_series = numpy.array([0.95, 0.97, 0.99, 1.01, 0.97, 0.98])
T_degC_series = 35.0  # This scalar value will be approapriately broadcast.
# Ignore additional outputs in the computation of the maximum power series.
P_mp_W_series, _, _, _ = sde_sim.P_mp(
    model_parameters=sdm_ae.compute_sde_model_parameters(
        ft_data=FTData(F=F_series, T_degC=T_degC_series),
        model_parameters=model_parameters_matrix,
    )
)

print(
    "\nPower simulation over a series of effective-irradiance ratios at one "
    "temperature:"
)
print(f"F_series = {list(F_series)}")
print(f"T_degC (fixed) = {T_degC_series}")
print(f"P_mp_W_series = {list(P_mp_W_series)}")

# Now make a nice plot.
fig, ax = pyplot.subplots(figsize=(8, 6))
# Plot the data fits.
for idx, (F, T_degC) in enumerate(
    zip(iv_performance_matrix.F, iv_performance_matrix.T_degC)
):
    # Plot Isc, Pmp, and Voc with same colors as fit lines.
    color = next(ax._get_lines.prop_cycler)["color"]
    ax.plot(
        ivft_data.V_V[3 * idx : 3 * idx + 3],
        ivft_data.I_A[3 * idx : 3 * idx + 3],
        "o",
        color=color,
    )
    V_V = numpy.linspace(0, ivft_data.V_V[3 * idx + 2], 101)
    ax.plot(
        V_V,
        sde_sim.I_at_V(
            V_V=V_V,
            model_parameters=sdm_ae.compute_sde_model_parameters(
                ft_data=FTData(F=F, T_degC=T_degC),
                model_parameters=model_parameters_matrix,
            ),
        )["I_A"],
    )
# Plot the LIC.
color = next(ax._get_lines.prop_cycler)["color"]
ax.plot(
    [0.0, iv_parameters_alt["V_mp_V"], iv_parameters_alt["V_oc_V"]],
    [iv_parameters_alt["I_sc_A"], iv_parameters_alt["I_mp_A"], 0.0],
    "*",
    color=color,
)
V_V = numpy.linspace(0, iv_parameters_alt["V_oc_V"], 101)
ax.plot(
    V_V,
    sde_sim.I_at_V(
        V_V=V_V,
        model_parameters=sdm_ae.compute_sde_model_parameters(
            ft_data=FTData(F=F_alt, T_degC=T_degC_alt),
            model_parameters=model_parameters_matrix,
        ),
    )["I_A"],
    "--",
    label=f"F={F_alt:.2f} suns, T={T_degC_alt:.0f} °C",
    color=color,
)
ax.set_title("6-Parameter SDM Fit to Performance Matrix", fontdict={"fontsize": 14})
ax.set_xlabel("V (V)")
ax.set_ylabel("I (A)")
fig.legend(loc="center")
fig.tight_layout()

pyplot.show()

# For comparison, fit simple SDM to spec-sheet data.

# model_parameters_spec_sheet has both fittable and unfittable parameters.
# Additional outputs can be useful, but ignored here.
print("\nFitting model parameters to specification datasheet...")
model_parameters_spec_sheet = sdm_simple_inf_spec_sheet.fit(
    spec_sheet_parameters=spec_sheet_parameters,
)["model_parameters"]
print("Fitting model parameters to specification datasheet...done")

mape_mbpe_spec_sheet, _ = sdm_simple_inf_matrix.compute_fit_quality(
    iv_performance_matrix=iv_performance_matrix,
    model_parameters=model_parameters_spec_sheet,
)

print("\nModel parameters from fit to specification datasheet:")
pprint(model_parameters_spec_sheet)
print("\nFit quality:")
pprint(mape_mbpe_spec_sheet)

print("\nI-V curve parameters at STC using specification-datasheet fit:")
pprint(
    sde_sim.iv_curve_parameters(
        model_parameters=sdm_ae.compute_sde_model_parameters(
            ft_data=FTData(F=1.0, T_degC=spec_sheet_parameters.T_degC_0),
            model_parameters=model_parameters_spec_sheet,
        )
    )
)
