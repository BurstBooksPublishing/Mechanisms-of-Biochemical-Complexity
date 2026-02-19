import numpy as np
from scipy.optimize import fsolve

# constants
pKa = [2.15, 7.20, 12.37]           # phosphoric acid pKa values
Ksp_apatite = 10**(-58.2)           # approximate Ksp at 25 C
Kw = 1e-14

def phosphate_fractions(pH):
    H = 10**(-pH)
    Ka1, Ka2, Ka3 = 10**(-pKa[0]), 10**(-pKa[1]), 10**(-pKa[2])
    # denominator for fractions
    D = H**3 + Ka1*H**2 + Ka1*Ka2*H + Ka1*Ka2*Ka3
    f_H3PO4 = H**3 / D
    f_H2PO4 = Ka1*H**2 / D
    f_HPO4  = Ka1*Ka2*H / D
    f_PO4   = Ka1*Ka2*Ka3 / D
    return f_H3PO4, f_H2PO4, f_HPO4, f_PO4

def saturation_index(pH, P_total_mol_L, Ca_mol_L):
    # compute free PO4^3- concentration assuming no precipitation
    f_PO4 = phosphate_fractions(pH)[3]
    PO4 = P_total_mol_L * f_PO4
    OH = Kw / 10**(-pH)
    IAP = Ca_mol_L**5 * PO4**3 * OH
    return np.log10(IAP / Ksp_apatite)

# example usage: pH 9.5, total P 1e-3 M, Ca2+ 1e-3 M
si = saturation_index(9.5, 1e-3, 1e-3)
print("Saturation index (SI) = ", si)  # SI<0: undersaturated, P mobilization feasible