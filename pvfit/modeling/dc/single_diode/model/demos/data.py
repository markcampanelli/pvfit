"""
PVfit: Sample data for example SDM parameter fitting.

Copyright 2023 Intelligent Measurement Systems LLC
"""

import numpy

from pvfit.measurement.iv.types import IVPerformanceMatrix, SpecSheetParameters
from pvfit.modeling.dc.common import E_hemi_W_per_m2_stc, Material, T_degC_stc

# Data for a Mission Solar mono-Si PV Module MSE300SQ5T, Measurement ID 19074-007.
# Thanks to Sandia National Laboratories and The PV Performance Modelling Collaborative.
# Source: https://pvpmc.sandia.gov/datasets/pv-lifetime-module-datasets-clone/


iv_performance_matrix = IVPerformanceMatrix(
    material=Material.monoSi,
    N_s=72,
    I_sc_A=numpy.array(
        [
            0.93336103381495,
            1.86021209967887,
            3.72980333051901,
            5.61464758380342,
            7.49606919536406,
            9.38198733814774,
            0.934733784908152,
            1.86564176483951,
            3.74363227577913,
            5.6330535807915,
            7.52430967073737,
            9.42522174117526,
            10.3632363215559,
            0.948977395725177,
            1.88859669121413,
            3.79035530927024,
            5.69109521246322,
            7.59054044812054,
            9.50435247014529,
            10.4503374584888,
            0.958300352581392,
            1.90037769064321,
            3.81419359243616,
            5.73701421503951,
            7.65038857810952,
            9.57371209346676,
            10.5382737062275,
        ]
    ),
    I_mp_A=numpy.array(
        [
            0.884129921041035,
            1.77603479715148,
            3.55570720915067,
            5.35479733158537,
            7.14959533066752,
            8.9478101916018,
            0.882434564774753,
            1.7674401684236,
            3.55287121691982,
            5.33846051759644,
            7.14431566637876,
            8.94563187783032,
            9.8330188942933,
            0.88870464486678,
            1.78083984014997,
            3.55366652438326,
            5.3453345487955,
            7.13445681466712,
            8.9159163455301,
            9.79719973543822,
            0.886447488495805,
            1.76860076982958,
            3.5350646914537,
            5.31596354527716,
            7.09526236824461,
            8.85793470541076,
            9.73490114502917,
        ]
    ),
    V_mp_V=numpy.array(
        [
            31.5917161865038,
            32.4089160664915,
            33.0750193222601,
            33.2432496858023,
            33.2613740623055,
            33.1952060126833,
            30.2295158269893,
            31.1010105611524,
            31.7779564489429,
            32.0042578074436,
            32.0288074402818,
            31.9608779018761,
            31.8901410225115,
            27.0177448280654,
            27.9530100927563,
            28.6612540722085,
            28.888640503419,
            28.9278045428985,
            28.8491314918701,
            28.8115360058225,
            23.7466718286528,
            24.7957943734284,
            25.5481979347726,
            25.8161665244669,
            25.8691079105164,
            25.8364826461413,
            25.8062537038654,
        ]
    ),
    V_oc_V=numpy.array(
        [
            36.6221388115327,
            37.7613175834051,
            38.911684518874,
            39.6050822879865,
            40.0970307625748,
            40.4869952641344,
            35.35874942097,
            36.5392968617771,
            37.7522487236879,
            38.472716569743,
            38.991403356579,
            39.3745346423522,
            39.5599462911284,
            32.2631341825221,
            33.5232298484849,
            34.8224225143853,
            35.5970966488472,
            36.1561538476712,
            36.5560330596728,
            36.7735495894715,
            29.0513776054511,
            30.4332939850915,
            31.8458611044042,
            32.6926943871273,
            33.2947651482518,
            33.7573974001532,
            33.9667314467376,
        ]
    ),
    E_W_per_m2=numpy.array(
        [
            100,
            200,
            400,
            600,
            800,
            1000,
            100,
            200,
            400,
            600,
            800,
            1000,
            1100,
            100,
            200,
            400,
            600,
            800,
            1000,
            1100,
            100,
            200,
            400,
            600,
            800,
            1000,
            1100,
        ]
    ),
    T_degC=numpy.array(
        [
            15,
            15,
            15,
            15,
            15,
            15,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            75,
            75,
            75,
            75,
            75,
            75,
            75,
        ]
    ),
    E_W_per_m2_0=E_hemi_W_per_m2_stc,
    T_degC_0=T_degC_stc,
)

spec_sheet_parameters = SpecSheetParameters(
    material=Material.monoSi,
    N_s=72,
    I_sc_A_0=iv_performance_matrix.I_sc_A_0,
    I_mp_A_0=iv_performance_matrix.I_mp_A_0,
    V_mp_V_0=iv_performance_matrix.V_mp_V_0,
    V_oc_V_0=iv_performance_matrix.V_oc_V_0,
    dI_sc_dT_A_per_degC_0=0.00314,
    dP_mp_dT_W_per_degC_0=-1.1417,
    dV_oc_dT_V_per_degC_0=-0.1125,
    E_W_per_m2_0=E_hemi_W_per_m2_stc,
    T_degC_0=T_degC_stc,
)

MODULE = {
    "iv_performance_matrix": iv_performance_matrix,
    "spec_sheet_parameters": spec_sheet_parameters,
}
