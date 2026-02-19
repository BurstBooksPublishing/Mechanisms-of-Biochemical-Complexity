import numpy as np
from scipy.integrate import solve_ivp

def cycle_odes(t, y, k_base, cat_factor, d):
    """
    ODEs for n-species cycle:
    X_i -> X_{i+1} with rate k = k_base * cat_factor
    Each X_i has first-order loss d.
    y: concentrations vector length n.
    """
    n = y.size
    k = k_base * cat_factor
    dydt = np.empty(n)
    # production from previous node and loss to next + background loss
    for i in range(n):
        prev = y[(i - 1) % n]
        prod = k * prev
        loss = k * y[i]  # forward conversion out of i
        dydt[i] = prod - loss - d * y[i]
    return dydt

def simulate_cycle(n=4, k_base=1e-5, cat_factor=1.0, d=1e-4,
                   y0=None, t_span=(0.0, 1e6), atol=1e-12, rtol=1e-9):
    """
    Integrate to approximate steady state and compute cycle flux.
    Returns time points, concentrations, and steady flux estimate J = k * [X_n].
    """
    if y0 is None:
        y0 = np.full(n, 1e-6)  # small seed concentrations
    sol = solve_ivp(fun=lambda t,y: cycle_odes(t, y, k_base, cat_factor, d),
                    t_span=t_span, y0=y0, atol=atol, rtol=rtol)
    y_ss = sol.y[:, -1]
    k_eff = k_base * cat_factor
    # flux through any bond equals k_eff * concentration of upstream node
    J = k_eff * y_ss[-1]
    return sol.t, sol.y, J

# Example usage (to be executed in analysis environment):
# t, concs, J = simulate_cycle(n=4, k_base=1e-5, cat_factor=50, d=1e-4)
# print("Steady flux J =", J)