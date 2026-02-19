#!/usr/bin/env python3
"""
Minimal ODE model: feedstock F -> monomers -> template T (templated copying)
and peptide P autocatalytically produced from monomers.
Peptide catalyzes template formation and stabilizes T (reduces degradation).
"""
import numpy as np
from scipy.integrate import solve_ivp

# Parameters (SI-like units, illustrative)
params = {
    "k_feed": 1.0,       # supply rate of feedstock F
    "k_conv": 0.1,       # baseline conversion F -> monomers
    "k_template": 0.01,  # baseline template formation
    "k_catP": 0.5,       # catalytic enhancement of template formation by P
    "k_pep_form": 0.02,  # peptide formation from monomers
    "k_deg_T": 0.05,     # template degradation
    "k_deg_P": 0.02,     # peptide degradation
    "phi": 10.0          # concentration factor (e.g., surface adsorption)
}

def rhs(t, y, p):
    F, P, T = y
    # effective template formation with peptide catalysis (Michaelis-like)
    K_M = 0.1
    k_eff_T = p["k_template"] + p["phi"] * p["k_catP"] * P / (K_M + P)
    dF = p["k_feed"] - p["k_conv"]*F - 0.0*F
    dP = p["k_pep_form"]*p["phi"]*F - p["k_deg_P"]*P
    dT = k_eff_T * p["k_conv"]*F - p["k_deg_T"]*T
    return [dF, dP, dT]

# initial conditions and integration
y0 = [10.0, 0.01, 0.01]  # F, P, T
tspan = (0.0, 1000.0)
sol = solve_ivp(lambda t,y: rhs(t,y,params), tspan, y0, rtol=1e-8, atol=1e-10)

# report steady-state approximations (last timepoint)
F_ss, P_ss, T_ss = sol.y[:, -1]
print(f"Steady-state concentrations: F={F_ss:.4f}, P={P_ss:.6f}, T={T_ss:.6f}")