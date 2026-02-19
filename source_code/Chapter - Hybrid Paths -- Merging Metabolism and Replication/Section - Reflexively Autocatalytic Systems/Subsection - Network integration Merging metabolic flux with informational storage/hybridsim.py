#!/usr/bin/env python3
"""
Hybrid metabolic-informational ODE model.
Metabolites c (n,), replicators p (m,). Catalysis A modifies reaction rates.
Requires: numpy, scipy
"""
import numpy as np
from scipy.integrate import solve_ivp

def kinetics(c, p, params):
    # Unpack
    k0 = params['k0']          # basal rate constants (r,)
    Km = params['Km']          # Michaelis constants (r,n)
    A  = params['A']           # catalysis matrix (m,r)
    # mass-action-like reaction rates modulated by metabolites and catalysis
    catal_factor = 1.0 + (p[:,None] @ A)  # shape (1,r) after broadcasting
    substrate = np.prod(c[None,:]**params['stoich'], axis=1)  # crude substrate dependence
    v = k0 * substrate * catal_factor.ravel()
    return v

def rhs(t, y, params):
    n = params['n']; m = params['m']; r = params['r']
    c = y[:n]; p = y[n:]
    v = kinetics(c, p, params)                       # reaction rates (r,)
    dc = params['I'] - params['D']*c - params['S'] @ v
    R = params['rep_base'] * (params['monomer_flux_coeff'] * v.sum())  # uniform replication driver
    dp = p * (R - params['Phi'])
    return np.concatenate([dc, dp])

# Example parameterization for a small system
def make_params():
    np.random.seed(0)
    n = 3; m = 4; r = 3
    params = {
        'n': n, 'm': m, 'r': r,
        'I': np.array([1.0, 0.5, 0.2]),      # inflow
        'D': np.eye(n)*0.1,                 # dilution
        'S': np.array([[1.0, -1.0, 0.0],    # stoichiometry (n x r)
                       [0.0,  1.0, -1.0],
                       [0.0,  0.0,  1.0]]),
        'k0': np.array([0.1, 0.05, 0.02]),
        'Km': np.ones((r,n)),
        'A': np.abs(np.random.randn(m,r))*0.2, # small catalytic enhancements
        'stoich': np.array([1.0, 1.0, 1.0]),   # simple product of c entries
        'rep_base': 0.01,
        'monomer_flux_coeff': 1.0,
        'Phi': 0.005
    }
    return params

if __name__ == '__main__':
    params = make_params()
    y0 = np.concatenate([np.array([0.1,0.1,0.1]), np.ones(params['m'])*0.01])
    sol = solve_ivp(lambda t,y: rhs(t,y,params), (0,2000), y0, rtol=1e-6, atol=1e-9)
    # sol.y contains trajectories for analysis and plotting (not shown here)