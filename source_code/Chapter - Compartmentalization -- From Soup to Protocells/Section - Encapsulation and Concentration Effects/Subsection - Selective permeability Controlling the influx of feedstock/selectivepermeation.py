#!/usr/bin/env python3
"""
Simulate internal concentrations for two species with different permeability.
Requires: numpy, scipy
"""
import numpy as np
from scipy.integrate import solve_ivp

def ode(t, y, params):
    # y = [C_in_X, C_in_Y]; params contains P, A, V, C_out, k_cons arrays
    C_in = y
    P = params['P']           # array([P_X, P_Y]) in m/s
    A = params['A']           # membrane area m^2
    V = params['V']           # volume m^3
    C_out = params['C_out']   # array([C_out_X, C_out_Y]) in mol/m^3
    k_cons = params['k_cons'] # array([kX, kY]) in s^-1
    phi = P * A / V           # influx rates s^-1
    dCdt = phi * (C_out - C_in) - k_cons * C_in
    return dCdt

# Parameters for 1 µm radius vesicle
r = 1e-6
A = 4 * np.pi * r**2
V = 4/3 * np.pi * r**3
params = {
    'P': np.array([1e-7, 1e-8]),          # P_X >> P_Y (selective)
    'A': A,
    'V': V,
    'C_out': np.array([1.0, 1.0]),        # mol/m^3 external
    'k_cons': np.array([0.01, 0.001])     # s^-1
}

y0 = np.array([0.0, 0.0])
t_span = (0.0, 200.0)  # seconds
sol = solve_ivp(ode, t_span, y0, args=(params,), rtol=1e-8, atol=1e-12)

# sol.y contains time courses for C_in_X and C_in_Y
# downstream analysis: compute steady-state fractions or feed into network solver