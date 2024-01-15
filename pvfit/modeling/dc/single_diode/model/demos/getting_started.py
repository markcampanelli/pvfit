"""
PVfit: Getting-started demo for single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC
"""

from pprint import pprint

from matplotlib import pyplot
import numpy

from pvfit.measurement.iv.types import FTData
import pvfit.modeling.dc.single_diode.equation.simulation as sde_sim
import pvfit.modeling.dc.single_diode.model.demos.data as data
import pvfit.modeling.dc.single_diode.model.simple.auxiliary_equations as ae
import pvfit.modeling.dc.single_diode.model.simple.inference_matrix as sdm_simple_inf_matrix
import pvfit.modeling.dc.single_diode.model.simple.types as sdm_simple_types


# By convention, variable names for numeric values include the units.

iv_performance_matrix = data.HIT_MODULE["matrix"]
model_parameters_unfittable = sdm_simple_types.ModelParametersUnfittable(
    N_s=data.HIT_MODULE["N_s"],
    T_degC_0=iv_performance_matrix.T_degC_0,
)
material = data.HIT_MODULE["material"]
ivft_data = iv_performance_matrix.ivft_data

print(f"HIT module with {model_parameters_unfittable['N_s']} cells in series:")
print(f"I_sc_A_0_data = {iv_performance_matrix.I_sc_A_0}")
print(f"I_A_data =\n{ivft_data.I_A}")
print(f"V_V_data =\n{ivft_data.V_V}")
print(f"F_data =\n{ivft_data.F}")
print(f"T_degC_data =\n{ivft_data.T_degC}")

# Fit simple SDM to matrix data. Additional outputs can be useful, but ignored here.
# model_parameters has both fittable and unfittable parameters, which can be
# used convienently in downstream functions, e.g., for power simulation.
model_parameters, _, _ = sdm_simple_inf_matrix.fit(
    iv_performance_matrix=iv_performance_matrix,
    model_parameters_unfittable=model_parameters_unfittable,
    material=material,
)

print("model_parameters from fit to performance matrix:")
pprint(model_parameters)

# Compute parameters for I-V curve at STC.
iv_curve_parameters_0 = sde_sim.iv_curve_parameters(
    model_parameters=ae.compute_sde_model_parameters(
        ft_data=FTData(F=1.0, T_degC=iv_performance_matrix.T_degC_0),
        model_parameters=model_parameters,
    )
)

print("I-V curve parameters at STC:")
pprint(iv_curve_parameters_0)

# Save some I-V curve values for later.
I_sc_A_0 = iv_curve_parameters_0["I_sc_A"]
I_mp_A_0 = iv_curve_parameters_0["I_mp_A"]
V_mp_V_0 = iv_curve_parameters_0["V_mp_V"]
V_oc_V_0 = iv_curve_parameters_0["V_oc_V"]

# Compute an alternative operating condition.
F_alt = 0.5
T_degC_alt = 35.0
iv_parameters_alt = sde_sim.iv_curve_parameters(
    model_parameters=ae.compute_sde_model_parameters(
        ft_data=FTData(F=F_alt, T_degC=T_degC_alt),
        model_parameters=model_parameters,
    )
)

print(f"Alternative operating condition: F={F_alt}, T={T_degC_alt} °C")
pprint(iv_parameters_alt)

# F and/or T_degC can be vectorized, such as for a time-series of weather data.
F_series = numpy.array([0.95, 0.97, 0.99, 1.01, 0.97, 0.98])
T_degC_series = 35.0  # This scalar value will be approapriately broadcast.
# Ignore additional outputs in the computation of the maximum power series.
P_mp_W_series, _, _, _ = sde_sim.P_mp(
    model_parameters=ae.compute_sde_model_parameters(
        ft_data=FTData(F=F_series, T_degC=T_degC_series),
        model_parameters=model_parameters,
    )
)

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
            model_parameters=ae.compute_sde_model_parameters(
                ft_data=FTData(F=F, T_degC=T_degC),
                model_parameters=model_parameters,
            ),
        ),
        label=f"F={F:.2f} suns, T={T_degC:.0f} °C",
        color=color,
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
        model_parameters=ae.compute_sde_model_parameters(
            ft_data=FTData(F=F_alt, T_degC=T_degC_alt),
            model_parameters=model_parameters,
        ),
    ),
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
