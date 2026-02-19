import numpy as np
from scipy.integrate import solve_ivp

# parameters (example values)
k_non = 1e-3           # M^-1 s^-1 nonenzymatic second-order rate
k_cat = 1.0            # s^-1 catalytic turnover
K_M = 1e-6             # M Michaelis constant
E_total = 1e-7         # M enzyme/ribozyme concentration

def rates(t, y):
    A, B, AB, ES = y  # A,B substrates; AB product; ES enzyme-substrate complex
    T_free = max(0.0, 1e-6 - (A + B + AB))  # simple template bookkeeping (M)
    # nonenzymatic ligation (templated effective second-order)
    v_non = k_non * A * B
    # enzymatic binding and catalysis (rapid equilibrium approx replaced by explicit ES dynamics)
    k_on = k_cat / K_M
    k_off = k_on * K_M
    v_bind = k_on * E_total * A - k_off * ES
    v_cat = k_cat * ES
    # ODEs
    dA = -v_non - v_bind + v_off if 'v_off' in locals() else -v_non - v_bind
    dB = -v_non
    dAB = v_non + v_cat
    dES = v_bind - v_cat
    return [dA, dB, dAB, dES]

y0 = [1e-6, 1e-6, 0.0, 0.0]  # initial concentrations (M)
sol = solve_ivp(rates, (0, 3600), y0, rtol=1e-8, atol=1e-12)
# results: sol.t (time), sol.y (concentrations) for analysis or plotting