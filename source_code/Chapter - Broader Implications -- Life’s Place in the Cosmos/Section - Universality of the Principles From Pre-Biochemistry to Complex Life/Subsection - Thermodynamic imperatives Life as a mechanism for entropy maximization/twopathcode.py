import numpy as np
from scipy.integrate import solve_ivp

# Parameters (SI units); replace with experimental deltaG values
T = 300.0                       # temperature K
k1 = 1e-3                       # uncatalyzed rate constant (s^-1)
k2 = 1e-1                       # catalyzed rate constant (M^-1 s^-1)
deltaG1 = -20e3                 # J mol^-1 for pathway 1
deltaG2 = -30e3                 # J mol^-1 for pathway 2
S_in = 1.0                      # feed concentration (M)
dilution = 1e-4                 # chemostat dilution rate (s^-1)

# ODE system: [S, C]
def odes(t, y):
    S, C = y
    r1 = k1 * S
    r2 = k2 * C * S
    dS = dilution*(S_in - S) - r1 - r2          # substrate balance
    dC = -dilution*C + 1e-2*r2                  # simple autocatalytic production term
    return [dS, dC]

# integrate to steady state
y0 = [0.5, 0.01]
sol = solve_ivp(odes, [0, 2e5], y0, rtol=1e-8, atol=1e-10)
S_ss, C_ss = sol.y[:, -1]

# compute steady fluxes and entropy production
r1_ss = k1 * S_ss
r2_ss = k2 * C_ss * S_ss
sigma = -(r1_ss*deltaG1 + r2_ss*deltaG2) / T   # J s^-1 mol^-1 / K -> W K^-1 per mol
print(f"Steady S={S_ss:.4f} M, C={C_ss:.4f} M, sigma={sigma:.3e} J s^-1 K^-1")
# Extend by converting to per-volume heat flux with Avogadro's number when needed.