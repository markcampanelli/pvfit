# Python 3.8+
import pprint
import sys

from matplotlib import pyplot as plt
import numpy
import requests

import pvfit.modeling.single_diode.equation as sde

# By convention, variable names include the units of the value.

# Load I-V data for a 72-cell x-Si module measured at constant irradiance
# and temperature. Ordering of the I-V points does not matter, as long as
# it's consistent between vectors.

# Measured terminal voltage.
V_V_data = numpy.array(
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
)
# Measured terminal current.
I_A_data = numpy.array(
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
)
# The number of cells in series in each parallel string.
N_s = 72
# The cell temperature  (a.k.a. the effective diode-junction temperature).
T_degC = 25.0

# Use the PVfit REST API to fit the data. (Requires internet connection!)
response = requests.post(
    "https://api.pvfit.app/v2/sdm/local",
    headers={"Content-Type": "application/json"},
    json={
        "V_V": V_V_data.tolist(),
        "I_A": I_A_data.tolist(),
        "N_s": N_s,
        "T_degC": T_degC,
    },
)

# Did call succeed? (If not 200, then remaining calls do not work.)
success = False
if response.status_code == 200:
    print(
        f"REST API call succeeded with response status code \
{response.status_code}."
    )
    response_json_dict = response.json()
    # Check that the fit was actually successfull.
    if response_json_dict["success"]:
        success = True
        # Note that N_s and T_degC are included in the fit result for
        # completeness and ease of use in subsequent function calls.
        model_params_fit = response_json_dict["model_params_fit"]
        # Adjust some keys for version change.
        model_params_fit = {
            "N_s": model_params_fit["N_s"],
            "T_degC": model_params_fit["T_degC"],
            "I_ph_A": model_params_fit["I_ph_A"],
            "I_rs_1_A": model_params_fit["I_rs_A"],
            "n_1": model_params_fit["n"],
            "R_s_Ohm": model_params_fit["R_s_Ohm"],
            "G_p_S": model_params_fit["G_p_S"],
        }
        print(f"Fit succeeded:\nmodel_params_fit = {model_params_fit}")
    else:
        print(
            f"Fit did not succeed:\nmodel_params_fit = \
{model_params_fit}\nfit_info = {response_json_dict['fit_info']}"
        )
else:
    print(
        f"REST API call failed with response status code \
{response.status_code}."
    )

if not success:
    model_params_fit = {
        "N_s": 72,
        "T_degC": 25.0,
        "I_ph_A": 8.903002024717399,
        "I_rs_1_A": 1.98620148842876e-07,
        "n_1": 1.2782631787313674,
        "R_s_Ohm": 0.3013297637749544,
        "G_p_S": 0.0005906833299600464,
    }
    print(
        f"Falling back to previously computed fit parameters:\n\
model_params_fit = {model_params_fit}"
    )

# The response has further information such as I-V curve parameters, which
# we recalculate here for demonstration purposes. Note that the fit
# parameters dictionary can be succintly passed to various modeling
# functions. Arguments to functions are always keyword-only, in order to
# be explicit yet flexible with argument ordering.
result = sde.iv_params(**model_params_fit)
print("I-V curve parameters result:")
pprint.pprint(result)

# Save some I-V curve values for later.
I_sc_A = result["I_sc_A"]
I_mp_A = result["I_mp_A"]
V_mp_V = result["V_mp_V"]
V_oc_V = result["V_oc_V"]

# Note that the I-V curve parameters include two additional points on the
# I-V curve that are relevant to the Sandia Array Performance Model
# (SAPM), namely (V_x, I_x) and (V_xx, I_xx), see
# https://pvpmc.sandia.gov/modeling-steps/2-dc-module-iv/point-value-models/sandia-pv-array-performance-model/.
# We now verify these values to further demonstrate the PVfit API.
# Compute the voltages.
V_x_V = V_oc_V / 2
V_xx_V = (V_mp_V + V_oc_V) / 2

# Demonstrate a vectorized computation of currents from voltages.
V_V = numpy.array([V_x_V, V_xx_V])
print(f"[V_x_V, V_xx_V] = {V_V}")
# The API consistently returns a result dictionary, so that one must
# specifically choose the currents from it.
result = sde.I_at_V(V_V=V_V, **model_params_fit)
# There are extra computed results included that are occassionally useful.
print(f"result = {result}")
# Get the currents (a numpy array).
I_A = result["I_A"]
print(f"[I_x_A, I_xx_A] = {I_A}")

# Unpack the computed currents.
I_x_A, I_xx_A = I_A[0], I_A[1]

# Now make a nice plot.
V_V = numpy.linspace(0.0, V_oc_V, num=100)
# Note the "shortcut" to getting only currents from the result dictionary.
I_A = sde.I_at_V(V_V=V_V, **model_params_fit)["I_A"]
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(V_V_data, I_A_data, ".", label="I-V data <-")
ax1.plot(V_V, I_A, label="fit to data <-")
ax1.plot(
    [0, V_x_V, V_mp_V, V_xx_V, V_oc_V],
    [I_sc_A, I_x_A, I_mp_A, I_xx_A, 0],
    "x",
    label="special fit points <-",
)
ax1.set_title(f"PVfit: 72-cell x-Si Module @ T={T_degC}°C")
ax1.set_xlabel("V [V]")
ax1.set_ylabel("I [A]")
ax2 = ax1.twinx()
# The plot of the sum of currents residuals is very useful for examining
# goodness of fit. Note that the REST API returned these residuals, but we
# re-compute them inline here for demonstration purposes.
ax2.plot(
    V_V_data,
    sde.current_sum_at_diode_node(V_V=V_V_data, I_A=I_A_data, **model_params_fit)[
        "I_sum_A"
    ],
    "+",
    label="residuals ->",
)
ax2.set_ylabel("I sum @ diode node (A)")
fig.legend()
fig.tight_layout()

# This logic allows automated testing of this script.
if "pytest" not in sys.modules:
    plt.show()
