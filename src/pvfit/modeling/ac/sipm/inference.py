"""
Sandia Inverter Performance Model (SIPM) inference.

https://energy.sandia.gov/wp-content/gallery/uploads/Performance-Model-for-Grid-Connected-Photovoltaic-Inverters.pdf

WARNING: This code is experimental and not subject to semantic versioning.
"""

import numpy
from scipy.optimize import least_squares


def fit(*, Vdc, Pdc, Pac, Vdco, Paco, Pnt):
    """
    Use least squares to fit the parameters of the Sandia Array Performance Model.
    """
    efficiency = numpy.amax(Pac / Pdc)

    # Construct IC vector for fit. [C0, C1, C2, C3, Pdco, Pso]
    x0 = numpy.array([0, 0, 0, 0, Paco / efficiency, 0.1 * Paco])

    # Inline these functions here for transformed model, with closures over data.
    def fun(x):
        deltaV = Vdc - Vdco
        A = x[4] * (1 + x[1] * deltaV)
        B = x[5] * (1 + x[2] * deltaV)
        C = x[0] * (1 + x[3] * deltaV)
        AminusB = A - B
        PdcminusB = Pdc - B
        return ((Paco / AminusB) - C * AminusB) * PdcminusB + C * PdcminusB**2 - Pac

    # Compute fit.
    sol = least_squares(
        fun,
        x0,
        method="dogbox",
        jac="3-point",
        max_nfev=10000 * x0.size,
        bounds=([-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, 0.0, 0.0], numpy.inf),
    )

    model_params_fit = {
        "Paco": Paco,
        "Pdco": sol.x[4],
        "Vdco": Vdco,
        "Pso": sol.x[5],
        "C0": sol.x[0],
        "C1": sol.x[1],
        "C2": sol.x[2],
        "C3": sol.x[3],
        "Pnt": Pnt,
    }

    return {"model_params_fit": model_params_fit, "sol": sol}
