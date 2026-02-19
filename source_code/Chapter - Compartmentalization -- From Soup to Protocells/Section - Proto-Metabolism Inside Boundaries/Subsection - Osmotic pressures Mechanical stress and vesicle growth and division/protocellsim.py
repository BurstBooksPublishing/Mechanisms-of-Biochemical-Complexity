#!/usr/bin/env python3
"""
Protocell osmotic model:
Integrate dA/dt = k_a * A and dV/dt = A * Lp * (RT*delta_c - 2*sigma/R)
Trigger division when sigma > sigma_crit.
"""
import numpy as np
from scipy.integrate import solve_ivp

def model(t, y, params):
    A, V, delta_c = y  # area (m^2), volume (m^3), internal-external conc diff (mol/m^3)
    R = np.sqrt(A / (4*np.pi))
    sigma = params['K_a'] * (A - params['A0']) / params['A0']  # linear tension
    delta_pi = params['Rgas']*params['T']*delta_c
    Jw = params['L_p'] * (delta_pi - 2*sigma / R)
    dVdt = A * Jw
    dAdt = params['k_a'] * A
    # optional: internal reaction producing solute (first-order)
    dcdt = params['k_prod']  # mol/(m^3 s), simple source term
    return [dAdt, dVdt, dcdt]

def run_sim(params, y0, t_span=(0,1000.0)):
    sol = solve_ivp(lambda t,y: model(t,y,params), t_span, y0,
                    atol=1e-9, rtol=1e-6, max_step=1.0)
    # post-process: detect division times where sigma crosses sigma_crit
    A_t, V_t, c_t = sol.y
    R_t = np.sqrt(A_t/(4*np.pi))
    sigma_t = params['K_a']*(A_t - params['A0'])/params['A0']
    div_idx = np.where(sigma_t > params['sigma_crit'])[0]
    division_times = sol.t[div_idx] if div_idx.size else np.array([])
    return sol.t, A_t, V_t, c_t, sigma_t, division_times

# Example parameters for an oleic-acid protocell (order-of-magnitude)
params = {
    'L_p': 1e-6,        # m s^-1 Pa^-1
    'Rgas': 8.314,      # J mol^-1 K^-1
    'T': 298.15,        # K
    'K_a': 0.05,        # N m^-1
    'A0': 3.14e-11,     # m^2 (reference area ~ micrometer vesicle)
    'k_a': 1e-4,        # s^-1 area growth rate
    'k_prod': 1e-6,     # mol m^-3 s^-1 internal solute production
    'sigma_crit': 5e-3  # N m^-1 (5 mN/m)
}
y0 = [3.14e-11, 4.19e-18, 0.0]  # A, V (1 um radius), delta_c initial
t, A, V, c, sigma, div_times = run_sim(params, y0, t_span=(0,3600))
# Results ready for plotting or further analysis (not shown).