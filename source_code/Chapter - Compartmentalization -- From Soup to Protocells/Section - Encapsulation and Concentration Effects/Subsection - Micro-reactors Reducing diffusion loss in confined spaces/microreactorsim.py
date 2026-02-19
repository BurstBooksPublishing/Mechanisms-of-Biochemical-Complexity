import numpy as np
from scipy.integrate import solve_ivp

def simulate_microreactor(R, P, k, influx_rate, c0=0.0, t_span=(0,100), t_eval=None):
    """
    Simulate dc/dt = influx_rate/Vol - (k + 3P/R)*c for a spherical micro-reactor.
    R: radius (m), P: permeability (m/s), k: first-order rate (s^-1)
    influx_rate: mol/s entering (scalar); assumes well-mixed interior
    c0: initial concentration (mol/m^3)
    t_span: integration window (s)
    t_eval: times to output
    Returns times, concentrations.
    """
    Vol = 4.0/3.0 * np.pi * R**3
    k_loss = k + 3.0*P/R
    def rhs(t, c):
        return influx_rate/Vol - k_loss * c
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(rhs, t_span, [c0], t_eval=t_eval, method='BDF', atol=1e-12, rtol=1e-9)
    return sol.t, sol.y[0]

# Example usage:
# t, c = simulate_microreactor(1e-6, 1e-7, 0.1, influx_rate=1e-15, t_span=(0,50))