"""
PVfit: Using meteorological (MET) station data in single-diode model (SDM).

Copyright 2023 Intelligent Measurement Systems LLC

PVfit expects "effective radiance ratio" input F = Isc/Isc0, as well as cell temperature
input T_degC. This expectation is both for initial model calibration and subsequent
simulation.

While F and T_degC are typically available calibrating the SDM, reference-device
measurements of F and T_degC are often not available for model simulation. Rather, MET
station data often must be used. We demonstrate one way to do this herein using the
Perez model with global horizontal irradiance (GHI), direct normal irradaiance (DNI),
diffuse horizontal irradiance (DHI), and ambient temperature. The model of Marion is
used to apply the IAM (from IEC 61853-2) to all of the POA irradiance components.

We use the Isc equation from the Sandia Array Performance Model (SAPM) to compute
effective irradiance ratio F, the Faiman model is used to compute module temperature
(using IEC 61853-2 measurements of U0 and U1), and then cell temperature T_degC is
computed from module temperature using the SAPM. 

References:
https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-sky-diffuse/perez-sky-diffuse-model/
https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/point-value-models/sandia-pv-array-performance-model/
https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/module-temperature/faiman-module-temperature-model/
https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/sandia-cell-temperature-model/

Hourly MET-station data for one day is taken from Scenario 4 of 2023 PVPMC Blind
Modeling Comparison. Thank you PVPMC!
https://pvpmc.sandia.gov/model-validation/2023-blind-modeling-comparisons/
"""

import numpy
import pandas
import pvlib


from pvfit.common import E_hemispherical_tilted_W_per_m2_stc, T_degC_stc
from pvfit.measurement.iv.types import FTData
from pvfit.modeling.dc.common import iam_factory


# LG320N1K-A5 320W LG NeON2
module_config = {
    "I_sc_A_0": 10.19,  # IEC 61853-1
    "dI_sc_dT_A_per_degC_0": 0.03 / 100 * 10.19,  # IEC 61853-1
    "iam_angle_deg": numpy.array(
        [0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]  # IEC 61853-2
    ),
    "iam_frac": numpy.array(
        [
            1.0000,
            1.0003,
            1.0005,
            1.0002,
            0.9943,
            0.9904,
            0.9819,
            0.9719,
            0.9557,
            0.9299,
            0.8843,
            0.8026,
            0.6817,
            0.4846,
            0.0000,
        ]
    ),  # IEC 61853-2
    "sapm": {
        "DeltaT_degC": 3.0,  # Stock value
    },
    "faiman": {
        "u0": 24.229,  # IEC 61853-2
        "u1": 7.182,  # IEC 61853-2
    },
}

iam = iam_factory(
    iam_angle_deg=module_config["iam_angle_deg"],
    iam_frac=module_config["iam_frac"],
)

location = {
    "name": "Albuquerque, NM, USA",
    "utc_offset": 7,
    "lat": 35.0546,
    "lon": -106.5401,
    "alt_m": 1600.0,
    "surface_tilt_deg": 35.0,
    "surface_azimuth_deg": 180.0,
}
location["pressure"] = pvlib.atmosphere.alt2pres(location["alt_m"])

weather = pandas.DataFrame()
weather["Year"] = 24 * [2022]
weather["Month"] = 24 * [7]
weather["Day"] = 24 * [1]
weather["Hour"] = list(range(24))
weather["Minute"] = 24 * [30]
weather["GHI (W/m2)"] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    97.64824306964874,
    242.8584406534831,
    461.178125,
    689.7611607869466,
    805.0706844987541,
    940.9629252115885,
    999.6489013671875,
    1021.733504231771,
    974.7346252441406,
    876.0769887288411,
    721.4664657592773,
    188.550729190602,
    167.9417485255821,
    130.8688592910767,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
weather["DNI (W/m2)"] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    378.5309042930603,
    630.9762669881185,
    778.0343149820964,
    705.4404176712036,
    695.2593346957502,
    933.8973124186198,
    904.6687403361003,
    958.8807312011719,
    950.420541381836,
    937.0442982991536,
    877.8845815022786,
    3.278694611261873,
    28.67140308314679,
    56.96313136443496,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
weather["DHI (W/m2)"] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    43.40976047515869,
    51.18269685109456,
    76.85209617614746,
    220.0052853902181,
    229.3682575883537,
    81.10905456542969,
    110.9699996948242,
    73.67663548787435,
    74.04667282104492,
    70.59681409200033,
    79.45995457967122,
    177.3235096650965,
    153.5331483728745,
    113.0418345133464,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
weather["Tamb (°C)"] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    22.9378125667572,
    23.53649991353353,
    25.19266662597656,
    26.63983332316081,
    27.39500006313982,
    28.50416673024495,
    29.80716660817464,
    30.70533351898193,
    31.56216669082642,
    31.8295000076294,
    32.68850008646647,
    30.98588225420784,
    29.43215684329762,
    26.98208324114482,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
weather["WS (m/s)"] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.227812491357327,
    2.42381666302681,
    2.38083333770434,
    2.129383325576782,
    2.868655171887628,
    3.009183336297671,
    2.300783324241638,
    2.603383340438207,
    3.16379998922348,
    2.588333337505659,
    2.483066666126251,
    5.476647068472469,
    7.583490184709137,
    9.090791702270508,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
weather["RH (%)"] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    43.46406269073486,
    40.7136666615804,
    36.77950000762939,
    31.44399998982747,
    31.91862053706728,
    28.42116664250692,
    25.3660000483195,
    23.41933342615763,
    21.31333338419596,
    19.89099979400635,
    18.14783333142599,
    22.40764718897202,
    26.46431365667605,
    36.32333342234293,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
weather["Albedo"] = 24 * [0.181936]

print(weather.to_string())

# This will hold timestamped T_degC and F.
output = pandas.DataFrame()
output["Year"] = weather["Year"]
output["Month"] = weather["Month"]
output["Day"] = weather["Day"]
output["Hour"] = weather["Hour"]
output["Minute"] = weather["Minute"]

# Use UTC for timestamps to avoid daylight savings issues in solar position.
timestamps = (
    pandas.DatetimeIndex(
        weather.apply(
            lambda record: pandas.Timestamp(
                year=int(record["Year"]),
                month=int(record["Month"]),
                day=int(record["Day"]),
                hour=int(record["Hour"]),
                minute=int(record["Minute"]),
            ),
            axis=1,
        )
    )
    + pandas.Timedelta(hours=location["utc_offset"])
).tz_localize("UTC")

sol_pos = pvlib.solarposition.get_solarposition(
    time=timestamps,
    latitude=location["lat"],
    longitude=location["lon"],
    altitude=location["alt_m"],
    temperature=weather["Tamb (°C)"].to_numpy(),
    pressure=location["pressure"],
)

aoi_deg = pvlib.irradiance.aoi(
    location["surface_tilt_deg"],
    location["surface_azimuth_deg"],
    sol_pos["apparent_zenith"],
    sol_pos["azimuth"],
)

dni_extra = pvlib.irradiance.get_extra_radiation(timestamps)

airmass = pvlib.atmosphere.get_relative_airmass(sol_pos["apparent_zenith"])

# Computational workflow for F, where F computation requires T_degC:
# DNI+GHI+DHI -> POA components -> AOI-corrected POA irradiance -> F
model = "perez"
model_perez = "allsitescomposite1990"

# This gets all sky-diffuse components.
sky_diffuse_components = pvlib.irradiance.perez(
    location["surface_tilt_deg"],
    location["surface_azimuth_deg"],
    weather["DHI (W/m2)"].to_numpy(),
    weather["DNI (W/m2)"].to_numpy(),
    dni_extra,
    sol_pos["apparent_zenith"],
    sol_pos["azimuth"],
    airmass=airmass,
    return_components=True,
).fillna(0.0)

# This gets remaining POA irradiance components.
components = pvlib.irradiance.poa_components(
    aoi_deg,
    weather["DNI (W/m2)"].to_numpy(),
    sky_diffuse_components["sky_diffuse"].to_numpy(),
    pvlib.irradiance.get_ground_diffuse(
        location["surface_tilt_deg"],
        weather["GHI (W/m2)"],
        albedo=weather["Albedo"].to_numpy(),
    ).to_numpy(),
).fillna(0.0)
components["poa_isotropic"] = sky_diffuse_components["isotropic"]
components["poa_circumsolar"] = sky_diffuse_components["circumsolar"]
components["poa_horizon"] = sky_diffuse_components["horizon"]

# (Back of) module temperature.
T_degC_module = pvlib.temperature.faiman(
    poa_global=components["poa_global"].to_numpy(),
    temp_air=weather["Tamb (°C)"].to_numpy(),
    wind_speed=weather["WS (m/s)"].to_numpy(),
    u0=module_config["faiman"]["u0"],
    u1=module_config["faiman"]["u1"],
)
components["T_degC_module"] = T_degC_module

# Cell temperature, which is also used in final calculation of F.
T_degC = pvlib.temperature.sapm_cell_from_module(
    module_temperature=T_degC_module,
    poa_global=components["poa_global"].to_numpy(),
    deltaT=module_config["sapm"]["DeltaT_degC"],
    irrad_ref=E_hemispherical_tilted_W_per_m2_stc,
)
output["T_degC"] = T_degC

# Compute IAMs for diffuse components.
diffuse_iam = pvlib.iam.marion_diffuse(
    "physical", surface_tilt=location["surface_tilt_deg"]
)

# Compute IAM-corrected total POA irradiance.
components["poa_total_iam_corr"] = (
    iam(angle_deg=aoi_deg) * (components["poa_direct"] + components["poa_circumsolar"])
    + diffuse_iam["sky"] * components["poa_isotropic"]
    + diffuse_iam["horizon"] * components["poa_horizon"]
    + diffuse_iam["ground"] * components["poa_ground_diffuse"]
)

print(components.to_string())

# This expression for F=Isc/Isc0 is derived from the SAPM equation for Isc.
# Assumes that Isc is linear with IAM-corrected POA irradiance and applies a simple
# linear temperature correction. Spectral effects are ignored by using a unity air-mass
# modifier (no correction info available), as well as ignoring spectral effects on the
# temperature coefficient for Isc.
F = numpy.maximum(
    0.0,
    components["poa_total_iam_corr"].to_numpy()
    / E_hemispherical_tilted_W_per_m2_stc
    * (
        1
        + module_config["dI_sc_dT_A_per_degC_0"]
        / module_config["I_sc_A_0"]
        * (T_degC - T_degC_stc)
    ),
)
output["F"] = F

print(output.to_string())

# This data object would then be input to subsequent SDM simulation functions.
ft_data = FTData(F=F, T_degC=T_degC)
