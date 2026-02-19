import numpy as np
from scipy.integrate import solve_ivp

# Rate constants (s^-1 or M^-1 s^-1 as appropriate). Tune per environment.
k1, km1 = 1.0, 1e-1   # imine formation and reverse
k2 = 1e3              # cyanide addition (M^-1 s^-1)
k3 = 1e-4             # nitrile hydrolysis (s^-1)

# ODE system: [RCHO], [NH3], [CN-], [Imine], [Aminonitrile], [AA]
def strecker_odes(t, y):
    RCHO, NH3, CN, I, N, AA = y
    # Imine formation and reverse
    v1f = k1 * RCHO * NH3
    v1r = km1 * I
    # Cyanide addition
    v2 = k2 * I * CN
    # Hydrolysis of aminonitrile
    v3 = k3 * N
    # ODEs
    dRCHO = -v1f + v1r
    dNH3  = -v1f + v1r + v3  # NH3 released on hydrolysis
    dCN   = -v2
    dI    = v1f - v1r - v2
    dN    = v2 - v3
    dAA   = v3
    return [dRCHO, dNH3, dCN, dI, dN, dAA]

# Initial concentrations (M)
y0 = [1e-3, 1e-3, 1e-4, 0.0, 0.0, 0.0]
tspan = (0, 1e6)  # seconds (~11.6 days)
sol = solve_ivp(strecker_odes, tspan, y0, dense_output=True, rtol=1e-6)

# Example access: concentration of glycine at final time
glycine_final = sol.y[5, -1]
print(f"Final glycine (M): {glycine_final:.3e}")