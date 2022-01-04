
import numpy

def Pac(*, Vdc, Pdc, Paco, Pdco, Vdco, Pso, C0, C1, C2, C3, Pnt):
    deltaV = Vdc - Vdco
    A = Pdco * (1 + C1 * deltaV)
    B = Pso * (1 + C2 * deltaV)
    C = C0 * (1 + C3 * deltaV)
    AminusB = A - B
    PdcminusB = Pdc - B
    return numpy.maximum(0., ((Paco / AminusB) - C * AminusB) * PdcminusB + C * PdcminusB**2)
