"""
Hydrosphere box model: integrates volume V (m^3) and salt mass m_s (kg).
- Q_in: inflow m^3/s with salinity S_in (kg/m^3)
- Q_out: outflow m^3/s
- evap: evaporation flux m^3/s (positive removes water)
- q_hydro: hydrothermal volumetric input (low salinity)
"""
import numpy as np
from scipy.integrate import solve_ivp

def deriv(t, y, params):
    V, m_s = y
    Q_in, S_in, Q_out, evap, q_hydro = params
    # Net volumetric change (evap positive -> volume loss)
    dVdt = Q_in + q_hydro - Q_out - evap
    # Salt mass change: inflow brings salt, outflow removes salt proportionally
    S = m_s / V if V > 0 else 0.0
    dm_sdt = Q_in * S_in - Q_out * S  # hydrothermal assumed near zero salinity
    return [dVdt, dm_sdt]

# Parameters (representative)
Q_in = 1e3         # m^3/s river input
S_in = 0.035       # kg/m^3 ~ 35 g/L -> 35 kg/m^3 (adjust units if desired)
Q_out = 1e3        # m^3/s
evap = 0.0         # closed-basin: set >0 for evaporation
q_hydro = 10.0     # hydrothermal input m^3/s (low salinity)

params = (Q_in, S_in, Q_out, evap, q_hydro)
V0 = 1e9           # initial volume m^3
m_s0 = V0 * 0.035  # initial salt mass kg

sol = solve_ivp(lambda t,y: deriv(t,y,params), [0, 1e7], [V0, m_s0],
                method='RK45', dense_output=True)
# Post-process for concentration: C(t) = m_s(t)/V(t)
t = np.linspace(0, 1e7, 500)
V_t, m_s_t = sol.sol(t)
C_t = m_s_t / V_t
# (Plotting code omitted; export C_t for analysis)