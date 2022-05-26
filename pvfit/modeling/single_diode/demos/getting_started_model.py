# Python 3.8+
import os
import pprint
import sys

from matplotlib import pyplot as plt
import numpy
import pandas
import requests

from pvfit.common.constants import T_degC_stc
import pvfit.modeling.single_diode.model as sdm

# By convention, variable names include the units of the value.

# Load IEC 61853-1 data for a 72-cell HIT module measured over variable
# irradiance and temperature. See https://www.nrel.gov/docs/fy14osti/61610.pdf and
# bill.marion@nrel.gov for the complete data set.
filename, N_s, material = os.path.join(os.path.dirname(__file__), "HIT05667.csv"), 72, 'x-Si'
data_table = pandas.read_csv(filename, skiprows=22, nrows=18, encoding='utf-8')
data_table.sort_values(['Irradiance Corrected to (W/m2)', 'Temperature Corrected to (degC)'], inplace=True)
print(data_table)

# We will flatten all channels from data table into vectors for model fitting.
V_V_data = numpy.full(0, numpy.nan)
I_A_data = numpy.copy(V_V_data)
F_data = numpy.copy(V_V_data)
T_degC_data = numpy.copy(V_V_data)
G_data_minimal = numpy.copy(V_V_data)
F_data_minimal = numpy.copy(V_V_data)
T_degC_data_minimal = numpy.copy(V_V_data)

# Pick out Isc at STC, which is needed to determine F for each I-V curve.
I_sc_A_0_meas_row_idx = numpy.logical_and(
    data_table.values[:, 2] == 25, data_table.values[:, 3] == 1000)
I_sc_A_0_meas = float(data_table.values[I_sc_A_0_meas_row_idx, 4])

# Ordering of the I-V-F-T points does not matter, as long as it's consistent
# between vectors.
for index, row in data_table.iterrows():
    V_V_data = numpy.concatenate((V_V_data, numpy.array([0., row[7], row[5]])))
    I_A_data = numpy.concatenate((I_A_data, numpy.array([row[4], row[6], 0.])))
    # Use effective irradiance ratio F, not the ratio of irradiances.
    F_data = numpy.concatenate((F_data, numpy.full(3, row[4] / I_sc_A_0_meas)))
    T_degC_data = numpy.concatenate((T_degC_data, numpy.full(3, row[2])))
    F_data_minimal = numpy.concatenate(
        (F_data_minimal, numpy.array([row[4] / I_sc_A_0_meas])))
    T_degC_data_minimal = numpy.concatenate(
        (T_degC_data_minimal, numpy.array([row[2]])))
    G_data_minimal = numpy.concatenate((G_data_minimal, numpy.array([row[3]])))

print(f"I_sc_A_0_meas = {I_sc_A_0_meas}")
print(f"I_A_data =\n{I_A_data}")
print(f"V_V_data =\n{V_V_data}")
print(f"F_data =\n{F_data}")
print(F"T_degC_data =\n{T_degC_data}")

# Use the PVfit REST API to fit the data. (Requires internet connection!)
# Note that we approximate the material as x-Si to guess the initial value for
# the band gap. Note the switching of the units position in variable names.
response = requests.post('https://api.pvfit.app/v2/sdm/global',
                         headers={'Content-Type': 'application/json'},
                         json={'V_V': V_V_data.tolist(),
                               'I_A': I_A_data.tolist(),
                               'F': F_data.tolist(),
                               'T_degC': T_degC_data.tolist(),
                               'N_s': N_s, 'T_degC_0': T_degC_stc,
                               'material': 'x-Si'})

# Did call succeed? (If not 200, then remaining calls do not work.)
success = False
if response.status_code == 200:
    print(f"REST API call succeeded with response status code \
{response.status_code}.")
    response_json_dict = response.json()
    # Check that the fit was actually successfull.
    if response_json_dict['success']:
        success = True
        # Note that N_s and T_degC_0 are included in the fit result for
        # completeness and ease of use in subsequent function calls.
        model_params_fit = response_json_dict['model_params_fit']
        # Adjust some keys for version change.
        model_params_fit = {'N_s': model_params_fit['N_s'],
                            'T_degC_0': model_params_fit['T_degC_0'],
                            'I_sc_A_0': model_params_fit['I_sc_A_0'],
                            'I_rs_1_A_0': model_params_fit['I_rs_A_0'],
                            'n_1_0': model_params_fit['n_0'],
                            'R_s_Ohm_0': model_params_fit['R_s_Ohm_0'],
                            'G_p_S_0': model_params_fit['G_p_S_0'],
                            'E_g_eV_0': model_params_fit['E_g_eV_0']}
        print(f"Fit succeeded:\nmodel_params_fit = {model_params_fit}")
    else:
        print(f"Fit did not succeed:\nmodel_params_fit = \
{model_params_fit}\nfit_info = {response_json_dict['fit_info']}")
else:
    print(f"API request failed with response status code \
{response.status_code}.")

if not success:
    model_params_fit = {
        'N_s': 72, 'T_degC_0': 25.0, 'I_sc_A_0': 5.531975277850156,
        'I_rs_1_A_0': 7.739216483430394e-11, 'n_1_0': 1.086322575846391,
        'R_s_Ohm_0': 0.5948166451105333, 'G_p_S_0': 0.00134157981991618,
        'E_g_eV_0': 1.1557066550514934}
    print(f"API call failed: Falling back to previously computed fit parameters:\n\
model_params_fit = {model_params_fit}")

# The response has further information such as I-V curve parameters at the
# reference condition (STC), which we recalculate here for demonstration
# purposes. Note that the fit parameters dictionary can be succintly passed to
# various modeling functions. Arguments to functions are always keyword-only,
# in order to be explicit yet flexible with argument ordering.
result = sdm.iv_params(F=1, T_degC=T_degC_stc, **model_params_fit)
print("I-V curve parameters at STC:")
pprint.pprint(result)

# Save some I-V curve values for later.
I_sc_A_0 = result['I_sc_A']
I_mp_A_0 = result['I_mp_A']
V_mp_V_0 = result['V_mp_V']
V_oc_V_0 = result['V_oc_V']

# Compute an alternative operating condition.
F_alt = 0.5
T_degC_alt = 35.
iv_params_alt = sdm.iv_params(F=F_alt, T_degC=T_degC_alt, **model_params_fit)
print(f"Alternative operating condition: F={F_alt}, T={T_degC_alt} °C")
pprint.pprint(iv_params_alt)

# F and/or T_degC can be vectorized, such as for a time-series of weather data.
F_series = numpy.array([0.95, 0.97, 0.99, 1.01, 0.97, 0.98])
T_degC_series = 35.  # This scalar value will be approapriately broadcast.
# Note that the maximum power vector is picked out from the result dictionary.
P_mp_W_series = sdm.iv_params(
    F=F_series, T_degC=T_degC_series, **model_params_fit)['P_mp_W']
print(f"P_mp_W_series = {list(P_mp_W_series)}")

# Now make a nice plot.
fig, ax = plt.subplots(figsize=(8, 6))
# Plot the data fits.
for idx, (F, T_degC) in enumerate(zip(F_data_minimal, T_degC_data_minimal)):
    # Plot Isc, Pmp, and Voc with same colors as fit lines.
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(V_V_data[3*idx:3*idx+3], I_A_data[3*idx:3*idx+3], 'o', color=color)
    V_V = numpy.linspace(0, V_V_data[3*idx+2], 101)
    ax.plot(V_V, sdm.I_at_V_F_T(V_V=V_V, F=F, T_degC=T_degC, **model_params_fit)['I_A'],
            label=f"F={F:.2f} suns, T={T_degC:.0f} °C", color=color)
# Plot the LIC.
color = next(ax._get_lines.prop_cycler)['color']
ax.plot([0., iv_params_alt['V_mp_V'], iv_params_alt['V_oc_V']],
        [iv_params_alt['I_sc_A'], iv_params_alt['I_mp_A'], 0.], '*', color=color)
V_V = numpy.linspace(0, iv_params_alt['V_oc_V'], 101)
ax.plot(V_V, sdm.I_at_V_F_T(V_V=V_V, F=F_alt, T_degC=T_degC_alt, **model_params_fit)['I_A'], '--',
        label=f"F={F_alt:.2f} suns, T={T_degC_alt:.0f} °C", color=color)
ax.set_title("6-Parameter SDM Fit to IEC 61853-1 Data", fontdict={'fontsize': 14})
ax.set_xlabel("V (V)")
ax.set_ylabel("I (A)")
fig.legend(loc="center")
fig.tight_layout()

# This logic allows automated testing of this script.
if "pytest" not in sys.modules:
    plt.show()
