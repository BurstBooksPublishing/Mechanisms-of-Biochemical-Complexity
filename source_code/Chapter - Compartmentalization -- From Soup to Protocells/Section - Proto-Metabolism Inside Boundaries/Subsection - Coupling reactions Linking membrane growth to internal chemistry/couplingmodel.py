import numpy as np
from scipy.integrate import solve_ivp

# Parameters (example values, tune to experimental data)
k_prod = 1e-3       # M^-1 s^-1, production rate constant
k_inc = 1e-2        # s^-1, incorporation rate
k_hyd = 1e-4        # s^-1, hydrolysis of P
k_loss = 1e-5       # s^-1, resource loss
F_R = 1e-6          # M s^-1, resource influx
alpha = 1e-12       # m^2 per molecule incorporated
c_geom = (36*np.pi)**(1/3)  # geometric constant for sphere
# conversion factors
NA = 6.02214076e23   # Avogadro

def odes(t, y):
    R, P, A, V, C = y
    dR = -k_prod*R*C - k_loss*R + F_R
    dP = k_prod*R*C - k_inc*P - k_hyd*P
    dA = alpha * k_inc * P - 0.0  # no active area loss included
    # Volume increase from osmosis or solute production; simple proportional model
    dV = 1e-18 * (k_prod*R*C)     # m^3 s^-1 per production event (tunable)
    dC = -1e-9 * C                 # slow catalyst decay
    return [dR, dP, dA, dV, dC]

# Initial conditions: molar concentrations for R,P,C; area (m^2) and volume (m^3)
y0 = [1e-3, 0.0, 1e-12, 1e-18, 1e-6]

t_span = (0, 1e5)
sol = solve_ivp(odes, t_span, y0, atol=1e-12, rtol=1e-8)

# Post-process: check area-to-volume scaling and detect instability
A = sol.y[2]
V = sol.y[3]
ratio = A / (c_geom * V**(2/3))
# ratio > 1 indicates excess area for given volume (curvature instability)