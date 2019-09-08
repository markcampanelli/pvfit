import scipy.constants


# Some useful physical constants (CODATA: https://physics.nist.gov/cuu/Constants/)

# elementary charge, 1.6021766208e-19 Coulombs
q_C = scipy.constants.value('elementary charge')

# Boltzmann constant, 1.38064852e-23 J/K
k_B_J_per_K = scipy.constants.value('Boltzmann constant')

# Boltzmann constant, 8.6173303e-5 eV/K
k_B_eV_per_K = scipy.constants.value('Boltzmann constant in eV/K')

# STC reference values

# Temperature
T_degC_stc = 25.  # Reference temperature in degrees Celsius
T_K_stc = scipy.constants.convert_temperature(T_degC_stc, 'Celsius', 'Kelvin')  # Reference temperature in Kelvin

# Total irradiance
G_hemi_W_per_m2_stc = 1000.  # At standard spectrum

# Materials
materials = \
    {   # From De Soto et al. 2006.
     'CIGS': {  # Copper Indium Gallium Selenide (CIGS).
              'E_g_eV_stc': 1.15,  # Band gap at STC, from De Soto et al. 2006.
             },
     'CIS':  {  # Copper Indium diSelenide (CIS).
              'E_g_eV_stc': 1.010,  # Band gap at STC, from De Soto et al. 2006.
             },
     'CdTe': {  # Cadmium Telluride (CdTe).
              'E_g_eV_stc': 1.475,  # Band gap at STC, from De Soto et al. 2006.
             },
     'GaAs': {  # Gallium Arsenide (GaAs).
              'E_g_eV_stc': 1.43,  # Band gap at 300 K, Kittel, C., Intro. to Solid State Physics, 6th ed. 1986, p 185.
             },
     'x-Si': {  # Mono-/multi-crystalline Silicon (x-Si).
        'E_g_eV_stc': 1.121,  # Band gap at STC, from De Soto et al. 2006.
             }
    }

# Solvers

# Defaults taken from scipy.optimize.newton v1.0.0
newton_tol_default = 1.48e-8
newton_maxiter_default = 50

# Defaults for 'bounded' mode taken from scipy.optimize.minimize_scalar v1.0.0
minimize_scalar_xatol_default = 1e-5
minimize_scalar_maxiter_default = 500

# TODO least_squares() defaults.
