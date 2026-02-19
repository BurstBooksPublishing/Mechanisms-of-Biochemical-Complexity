import numpy as np
from scipy.integrate import solve_ivp
# Production-ready RHS for the dimensionless Oregonator.
def oregonator(t, y, q=0.002, f=1.0, phi=0.1):
    x, yb, z = y
    dx = q*yb - x*yb + x*(1.0 - x)
    dy = -q*yb - x*yb + f*z
    dz = phi*(x - z)
    return [dx, dy, dz]

# Example integration: parameter set near oscillatory regime.
params = dict(q=0.002, f=1.4, phi=0.08)
y0 = [0.1, 0.1, 0.1]            # small perturbation from steady state
t_span = (0.0, 500.0)          # dimensionless time units
t_eval = np.linspace(*t_span, 5000)

sol = solve_ivp(lambda t, y: oregonator(t, y, **params),
                t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

# sol.t and sol.y contain time and concentrations for analysis or plotting.