"""
PVfit: Getting-started demo for single-diode equation (SDE).

Copyright 2023 Intelligent Measurement Systems LLC
"""

import pprint

from matplotlib import pyplot
import numpy

from pvfit.measurement.iv.types import IVCurve
import pvfit.modeling.dc.single_diode.equation.simple.inference_iv_curve as sde_inf_iv
import pvfit.modeling.dc.single_diode.equation.simple.simulation as sde_sim
from pvfit.modeling.dc.single_diode.equation.simple.types import (
    ModelParametersUnfittable,
)

# By convention, variable names for numeric values include the units.

# Arguments to functions are always keyword-only, in order to be explicit yet
# flexible with argument ordering.

# Load I-V data for a 72-cell x-Si module measured at constant irradiance and
# temperature. Ordering of the I-V points does not matter, as long as it's
# consistent between vectors.

iv_curve = IVCurve(
    # The terminal voltage data.
    V_V=numpy.array(
        [
            41.6512756,
            41.6472664,
            41.62723,
            41.5991859,
            41.5751457,
            41.5591164,
            41.52706,
            41.50302,
            41.47898,
            41.454937,
            41.4228821,
            41.4068565,
            41.3788071,
            41.35076,
            41.3227119,
            41.3026772,
            41.2706223,
            41.2425766,
            41.2065125,
            41.1824722,
            41.1544228,
            41.1263771,
            41.0983276,
            41.0662727,
            41.0262032,
            40.9901428,
            40.970108,
            40.9460678,
            40.91802,
            40.8899727,
            40.8619232,
            40.8338737,
            40.80583,
            40.7737732,
            40.749733,
            40.71367,
            40.6856232,
            40.65357,
            40.6135,
            40.58545,
            40.5453835,
            40.51333,
            40.4652443,
            40.429184,
            40.39312,
            40.34504,
            40.2969551,
            40.25288,
            40.2048,
            40.1567154,
            40.10062,
            40.04853,
            39.99644,
            39.964386,
            39.9123,
            39.876236,
            39.84819,
            39.82014,
            39.7880859,
            39.75603,
            39.72798,
            39.69192,
            39.6598663,
            39.62781,
            39.59175,
            39.55168,
            39.51562,
            39.48757,
            39.4475021,
            39.41144,
            39.3833923,
            39.3393173,
            39.2952423,
            39.2671928,
            39.2271271,
            39.1830521,
            39.1469879,
            39.1029129,
            39.0628433,
            39.0227776,
            38.9746933,
            38.9346237,
            38.8865433,
            38.8464737,
            38.7983932,
            38.75031,
            38.7102432,
            38.66216,
            38.61007,
            38.56199,
            38.5099,
            38.45781,
            38.40572,
            38.35363,
            38.3095551,
            38.2574654,
            38.20137,
            38.15329,
            38.09719,
            38.03709,
            37.9769859,
            37.90887,
            37.84476,
            37.7846565,
            37.71654,
            37.6444168,
            37.580307,
            37.5161972,
            37.43606,
            37.3679428,
            37.2918129,
            37.215683,
            37.13154,
            37.0473976,
            36.96726,
            36.8831139,
            36.7949638,
            36.71082,
            36.6146545,
            36.51849,
            36.4263344,
            36.31414,
            36.2219849,
            36.10979,
            35.9976,
            35.8894157,
            35.7652,
            35.6490021,
            35.5207825,
            35.3765373,
            35.240303,
            35.0880432,
            34.9357834,
            34.77551,
            34.61123,
            34.44294,
            34.26263,
            34.0703,
            33.849926,
            33.6215324,
            33.3891373,
            33.1246834,
            32.86023,
            32.5517044,
            32.1910858,
            31.7863941,
            31.3336182,
            30.7967,
            30.1756363,
            29.3261833,
            28.0960789,
            26.296999,
            23.91292,
            21.3284969,
            19.0365753,
            17.1773949,
            16.0795155,
            15.4905071,
            14.69715,
            13.711463,
            12.58153,
            11.5677948,
            10.642211,
            9.756696,
            8.923269,
            8.045768,
            7.21634865,
            6.57525158,
            6.030319,
            5.525455,
            5.056653,
            4.555796,
            3.994836,
            3.39380741,
            2.792779,
            2.19976425,
            1.64281118,
            1.310242,
            1.19805014,
            0.9175702,
            0.741268456,
        ]
    ),
    # The terminal current data.
    I_A=numpy.array(
        [
            -0.00121053064,
            0.00484212255,
            0.046000164,
            0.0883687362,
            0.131947845,
            0.1743164,
            0.219106048,
            0.2614746,
            0.30263266,
            0.349843353,
            0.3958435,
            0.437001556,
            0.479370147,
            0.5265808,
            0.5689494,
            0.6125285,
            0.6597392,
            0.713002563,
            0.767476439,
            0.809845,
            0.857055664,
            0.9054769,
            0.9575297,
            1.01200366,
            1.067688,
            1.124583,
            1.16695154,
            1.20810962,
            1.25410974,
            1.29647827,
            1.34005737,
            1.384847,
            1.43326831,
            1.480479,
            1.5276897,
            1.57732141,
            1.62937427,
            1.68384814,
            1.738322,
            1.79400635,
            1.84969079,
            1.90900683,
            1.96953332,
            2.032481,
            2.09542847,
            2.16200781,
            2.2322185,
            2.3024292,
            2.375061,
            2.45253515,
            2.52879858,
            2.60627246,
            2.684957,
            2.72853613,
            2.8096416,
            2.85201025,
            2.8967998,
            2.940379,
            2.983958,
            3.027537,
            3.07111621,
            3.11590576,
            3.166748,
            3.21032715,
            3.25874853,
            3.30838013,
            3.35559082,
            3.4040122,
            3.451223,
            3.50206518,
            3.55048633,
            3.602539,
            3.65096045,
            3.70301318,
            3.75143456,
            3.80469775,
            3.85554,
            3.90880346,
            3.96327734,
            4.01533031,
            4.0685935,
            4.124278,
            4.17754126,
            4.233226,
            4.2901206,
            4.345805,
            4.40148926,
            4.45959473,
            4.518911,
            4.57580566,
            4.63633251,
            4.69806957,
            4.75496435,
            4.81428051,
            4.866333,
            4.915965,
            4.97649145,
            5.039439,
            5.10238647,
            5.16412354,
            5.230703,
            5.29607153,
            5.36022949,
            5.426809,
            5.49459839,
            5.559967,
            5.62412548,
            5.69312572,
            5.762126,
            5.831126,
            5.90012646,
            5.9691267,
            6.040548,
            6.111969,
            6.18218,
            6.253601,
            6.326233,
            6.396444,
            6.47270727,
            6.540497,
            6.61918163,
            6.69181347,
            6.766866,
            6.84312963,
            6.91576147,
            6.992025,
            7.06828833,
            7.14455175,
            7.21718359,
            7.29707861,
            7.375763,
            7.450816,
            7.530711,
            7.60818529,
            7.68202734,
            7.75708055,
            7.83213329,
            7.90839672,
            7.98466,
            8.060924,
            8.132345,
            8.208609,
            8.277609,
            8.34903,
            8.420451,
            8.491873,
            8.558452,
            8.62382,
            8.685557,
            8.743663,
            8.7969265,
            8.834453,
            8.855032,
            8.863505,
            8.863505,
            8.864716,
            8.871979,
            8.87682152,
            8.885295,
            8.890137,
            8.89497948,
            8.89497948,
            8.89619,
            8.89619,
            8.8974,
            8.8974,
            8.898611,
            8.8974,
            8.8974,
            8.89619,
            8.898611,
            8.89619,
            8.898611,
            8.899821,
            8.904663,
            8.907084,
            8.909506,
            8.908295,
            8.907084,
            8.911926,
            8.908295,
        ]
    ),
)

model_parameters_unfittable = ModelParametersUnfittable(
    # The number of cells in series in each parallel string.
    N_s=72,
    # The cell temperature (a.k.a. diode-junction temperature).
    T_degC=25.0,
)

# Fit model parameters, ignoring additional return values, which are initial condition
# used and low-level ODR solver result for a transformed problem.
model_parameters = sde_inf_iv.fit(
    iv_curve=iv_curve,
    model_parameters_unfittable=model_parameters_unfittable,
)["model_parameters"]

print("SDE model parameters fit result:")
pprint.pprint(model_parameters)

# The fit model parameters dictionary can be succinctly passed to various
# modeling functions. Compute the I-V curve parameters for the fit model.
iv_curve_parameters_fit = sde_sim.iv_curve_parameters(model_parameters=model_parameters)
print("\nI-V curve parameters from fit model:")
pprint.pprint(iv_curve_parameters_fit)

# Save some I-V curve values for later.
I_sc_A = iv_curve_parameters_fit["I_sc_A"]
I_mp_A = iv_curve_parameters_fit["I_mp_A"]
V_mp_V = iv_curve_parameters_fit["V_mp_V"]
V_oc_V = iv_curve_parameters_fit["V_oc_V"]

# Note that the I-V curve parameters include two additional points on the
# I-V curve that are relevant to the Sandia Array Performance Model
# (SAPM), namely (V_x, I_x) and (V_xx, I_xx), see
# https://pvpmc.sandia.gov/modeling-steps/2-dc-module-iv/point-value-models/sandia-pv-array-performance-model/.
# We now verify these values to further demonstrate the PVfit API.
# Compute the voltages.
V_x_V = V_oc_V / 2
V_xx_V = (V_mp_V + V_oc_V) / 2

# Demonstrate a vectorized computation of currents from voltages.
print("\nAlternative computation of SAPM points:")
V_V = numpy.array([V_x_V, V_xx_V])
print(f"[V_x_V, V_xx_V] = {V_V}")
I_A = sde_sim.I_at_V(V_V=V_V, model_parameters=model_parameters)["I_A"]
print(f"[I_x_A, I_xx_A] = {I_A}")

# Unpack the computed currents for subsequent plot.
I_x_A, I_xx_A = I_A[0], I_A[1]

# Now make a nice plot.
fig, ax1 = pyplot.subplots(figsize=(8, 6))

# Plot the fit I-V curve against the data.
V_V = numpy.linspace(0.0, V_oc_V, num=100)
I_A = sde_sim.I_at_V(V_V=V_V, model_parameters=model_parameters)["I_A"]
ax1.plot(iv_curve.V_V, iv_curve.I_A, ".", label="I-V data <-")
ax1.plot(V_V, I_A, label="fit to data <-")
ax1.plot(
    [0, V_x_V, V_mp_V, V_xx_V, V_oc_V],
    [I_sc_A, I_x_A, I_mp_A, I_xx_A, 0],
    "x",
    label="special fit points <-",
)
ax1.set_title(f"PVfit: 72-cell x-Si Module @ T={model_parameters['T_degC']}°C")
ax1.set_xlabel("V [V]")
ax1.set_ylabel("I [A]")

# Plot the residuals for the sum of currents at the diode node, which is very
# useful for examining fit quality.
ax2 = ax1.twinx()
ax2.plot(
    iv_curve.V_V,
    sde_sim.I_sum_diode_anode_at_I_V(
        iv_data=iv_curve,
        model_parameters=model_parameters,
    )["I_sum_diode_anode_A"],
    "+",
    label="residuals ->",
)
ax2.set_ylabel("I sum @ diode node (A)")

fig.legend()
fig.tight_layout()

pyplot.show()
