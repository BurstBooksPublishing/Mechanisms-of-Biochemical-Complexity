import numpy as np
from scipy.integrate import solve_ivp

# Parameters (example values; fit to experiments)
k_inc = 1e3       # M^-2 s^-1 effective incorporation constant
k_deg = 1e-4      # s^-1 protein degradation
k_mis = 1e2       # M^-2 s^-1 misincorporation constant
aaRS_ots = 1e-6   # M OTS synthetase
aaRS_nat = 1e-6   # M native synthetase
rib = 1e-6        # M ribosome concentration
cAA = 1e-3        # M canonical amino acid pool
ncAA_init = 1e-4  # M initial ncAA

def odes(t, y):
    ncAA, Pnc, Pmis = y
    inc_flux = k_inc * aaRS_ots * ncAA * rib
    mis_flux = k_mis * aaRS_nat * cAA * rib
    dncAA = -inc_flux                   # consumption by incorporation
    dPnc = inc_flux - k_deg * Pnc
    dPmis = mis_flux - k_deg * Pmis
    return [dncAA, dPnc, dPmis]

y0 = [ncAA_init, 0.0, 0.0]
tspan = (0, 3600)                      # one hour
sol = solve_ivp(odes, tspan, y0, rtol=1e-6, atol=1e-9)

# Post-processing example: fidelity over time
fidelity = (k_inc*aaRS_ots*sol.y[0]*rib)/(k_mis*aaRS_nat*cAA*rib + 1e-12)
# fidelity array gives time-resolved measure to compare with target