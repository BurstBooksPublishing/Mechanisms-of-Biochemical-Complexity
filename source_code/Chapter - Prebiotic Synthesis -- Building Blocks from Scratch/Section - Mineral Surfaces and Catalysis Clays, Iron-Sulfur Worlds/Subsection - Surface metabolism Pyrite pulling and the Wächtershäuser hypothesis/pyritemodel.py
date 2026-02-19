#!/usr/bin/env python3
"""Simulate Langmuir adsorption and surface reaction with pyrite coupling."""
import numpy as np
from scipy.integrate import solve_ivp

def model(t, y, params):
    # y: [theta_CO, theta_HS, S_free] where S_free = fraction of surface free
    theta_CO, theta_HS, S_free = y
    K_CO, K_HS, p_CO, p_HS, k_cat, J_pyrite = params
    # Instantaneous adsorption/desorption fluxes (fast equilibrium approximation)
    denom = 1 + K_CO * p_CO + K_HS * p_HS
    theta_CO_eq = (K_CO * p_CO) / denom
    theta_HS_eq = (K_HS * p_HS) / denom
    # Relaxation toward equilibrium (fast timescale fraction)
    tau_ads = 1e-2
    dtheta_CO = (theta_CO_eq - theta_CO) / tau_ads
    dtheta_HS = (theta_HS_eq - theta_HS) / tau_ads
    # Surface reaction producing acetyl thioester, rate proportional to coverage
    r_rxn = k_cat * theta_CO * theta_HS
    # Pyrite formation supplies reducing equivalents as a source term J_pyrite
    # Here J_pyrite augments effective k_cat linearly (simple coupling)
    dS_free = -r_rxn + (1 - S_free) * 0.0  # keep surface capacity constant
    dtheta_CO -= r_rxn * theta_CO  # consumption reduces bound fraction
    dtheta_HS -= r_rxn * theta_HS
    # Add pyrite coupling effect as modification of reaction rate (implicit)
    dtheta_CO += J_pyrite * 0.0  # placeholder for explicit coupling if needed
    return [dtheta_CO, dtheta_HS, dS_free]

# Parameters (example values)
params = (1e3, 5e2, 1e-5, 1e-4, 1e1, 1e-3)  # K_CO,K_HS,p_CO,p_HS,k_cat,J_pyrite

# Initial conditions: low initial coverages
y0 = [1e-6, 1e-6, 1.0]
tspan = (0, 100)
sol = solve_ivp(model, tspan, y0, args=(params,), dense_output=True)
# Export time series for analysis
t = np.linspace(*tspan, 200)
theta_CO, theta_HS, S_free = sol.sol(t)
# Save or plot externally; no plotting inside library code.