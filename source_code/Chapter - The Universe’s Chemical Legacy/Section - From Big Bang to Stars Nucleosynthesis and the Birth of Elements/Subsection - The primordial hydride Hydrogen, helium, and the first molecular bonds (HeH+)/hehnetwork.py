import numpy as np
from scipy.integrate import solve_ivp

# Physical constants and initial conditions (cm^-3)
nH_tot = 1e2                 # total hydrogen nuclei density
Y_He = 0.08                  # He/H by number
nHe = Y_He * nH_tot
nH = 0.99 * nH_tot           # initial neutral H
nHp = 1e-2 * nH_tot          # initial protons
ne = nHp                     # charge neutrality initially

# Placeholder rate coefficients (cm^3 s^-1 or cm^3 s^-1 for two-body)
k_ra = 1e-16                 # He + H+ -> HeH+ + photon (radiative assoc.)
k_cx = 1e-9                  # HeH+ + H -> He + H2+
alpha_dr = 1e-7              # HeH+ + e- -> He + H (diss. recomb.)
k1 = 1e-9                    # H2+ + H -> H2 + H+

def dydt(t, y):
    nH, nHp, nH2p, nHeH, ne = y
    # formation and loss terms
    rate_ra = k_ra * nHe * nHp
    rate_cx = k_cx * nHeH * nH
    rate_dr = alpha_dr * nHeH * ne
    rate_h2form = k1 * nH2p * nH
    # H2+ production from charge transfer
    dnH2p = rate_cx - rate_h2form
    # HeH+ equation
    dnHeH = rate_ra - rate_cx - rate_dr
    # Proton and electron bookkeeping (approximate)
    dnHp = -rate_ra + rate_h2form  # H+ consumed by ra, produced by H2 formation
    dne = -rate_dr + 0.0           # electrons change by recombination only
    dnH = -rate_cx - rate_h2form   # neutral H lost to reactions (approx.)
    return [dnH, dnHp, dnH2p, dnHeH, dne]

y0 = [nH, nHp, 0.0, 0.0, ne]
t_span = (0.0, 1e13)  # seconds; choose timescale relevant to problem
sol = solve_ivp(dydt, t_span, y0, method='BDF', rtol=1e-6, atol=1e-12)

# Example output: final HeH+ fraction relative to He
nHeH_final = sol.y[3, -1]
print("HeH+ fraction of He:", nHeH_final / nHe)