import numpy as np
from scipy.integrate import solve_ivp

# Physical parameters
T = 10.0                     # K
n_H2 = 1e4                   # cm^-3, H2 number density (constant reservoir)
n_CO = 1e-5 * n_H2           # cm^-3, CO abundance relative to H2
zeta = 1.0e-17               # s^-1, cosmic-ray ionization rate per H2

# Reaction rate coefficients (cm^3 s^-1)
k_H3p_CO = 1.7e-9            # H3+ + CO -> HCO+ + H2
k_dr_HCO = 2.0e-7 * (T/300.0)**(-0.5)  # HCO+ + e- -> neutrals
k_dr_H3 = 6.0e-8  * (T/300.0)**(-0.5)  # H3+ + e- -> neutrals

# ODE system: y = [H3+, HCO+, e-]
def dydt(t, y):
    H3p, HCOp, e = y
    # Formation of H3+ by cosmic rays (approximate direct production)
    R_form_H3p = zeta * n_H2
    # Reactions
    r1 = k_H3p_CO * H3p * n_CO       # H3+ + CO -> HCO+
    r2 = k_dr_HCO * HCOp * e         # HCO+ + e- -> neutrals
    r3 = k_dr_H3  * H3p * e          # H3+ + e- -> neutrals
    dH3p = R_form_H3p - r1 - r3
    dHCOp = r1 - r2
    de = R_form_H3p - r2 - r3        # electrons produced by ionization, consumed by recombination
    return [dH3p, dHCOp, de]

# Initial conditions (negligible ions)
y0 = [1e-12, 1e-12, 1e-12]
t_span = (0.0, 3.154e13)  # integrate to ~1 Myr in seconds
sol = solve_ivp(dydt, t_span, y0, rtol=1e-8, atol=1e-12)

# Example output: final abundances (cm^-3)
H3p_fin, HCOp_fin, e_fin = sol.y[:, -1]
print("Final abundances (cm^-3): H3+ {:.3e}, HCO+ {:.3e}, e- {:.3e}".format(
    H3p_fin, HCOp_fin, e_fin))