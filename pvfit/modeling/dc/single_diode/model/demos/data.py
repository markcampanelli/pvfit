"""
PVfit: Sample data for example SDM parameter fitting.

Copyright 2023 Intelligent Measurement Systems LLC
"""

from io import StringIO

import pandas

from pvfit.modeling.dc.common import Material


# Performance-matrix data for a 72-cell HIT module measured over variable
# irradiance and temperature. See https://www.nrel.gov/docs/fy14osti/61610.pdf and
# bill.marion@nrel.gov for the complete data set.

HIT_MODULE = {
    "N_s": 72,
    "material": Material.xSi,
    "matrix": pandas.read_csv(
        StringIO(
            """T_degC,G_W_per_m2,I_sc_A,V_oc_V,I_mp_A,V_mp_V,P_mp_W
15,100,0.558,46.74,0.51,40.1,20.47
25,100,0.57,45.37,0.526,38.39,20.19
15,200,1.111,48.23,1.013,41.7,42.25
25,200,1.126,46.9,1.041,40.15,41.79
25,400,2.232,48.34,2.076,41.01,85.16
50,400,2.234,44.89,2.059,37.23,76.65
25,600,3.334,49.17,3.114,41.3,128.58
50,600,3.329,45.81,3.081,37.77,116.35
65,600,3.349,43.77,3.074,35.63,109.53
25,800,4.433,49.76,4.151,41.38,171.76
50,800,4.441,46.42,4.115,37.94,156.15
65,800,4.459,44.43,4.106,35.83,147.12
25,1000,5.532,50.21,5.177,41.43,214.48
50,1000,5.539,46.9,5.112,38.1,194.79
65,1000,5.566,44.92,5.119,35.93,183.92
25,1100,6.079,50.39,5.632,41.52,233.84
50,1100,6.101,47.11,5.639,37.87,213.58
65,1100,6.129,45.16,5.604,35.99,201.73"""
        )
    ),
}
