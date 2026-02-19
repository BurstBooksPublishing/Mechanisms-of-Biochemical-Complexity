import numpy as np
from scipy.integrate import solve_ivp

# ASF distribution function
def asf_distribution(alpha, max_n=20):
    n = np.arange(1, max_n+1)
    Wn = (1 - alpha) * alpha**(n - 1)
    return n, Wn

# Simple ODE model: d[CO]/dt = -k_consumption*[CO], product pools follow ASF
def kinetics(t, y, k, alpha):
    CO = y[0]
    P = y[1:]  # product pools for n=1..N
    dCO = -k * CO
    production = k * CO * (1 - alpha) * alpha**(np.arange(len(P)))
    dP = production
    return np.concatenate(([dCO], dP))

# Parameters (example)
k = 1e-4            # s^-1, effective surface conversion rate
alpha = 0.7         # chain growth probability
CO0 = 1.0           # initial CO concentration (arbitrary units)
N = 12              # number of homolog bins

# initial state
y0 = np.concatenate(([CO0], np.zeros(N)))
t_span = (0, 1e5)   # seconds
sol = solve_ivp(kinetics, t_span, y0, args=(k, alpha), dense_output=True)

# Evaluate final ASF and fraction in C1..CN
t_final = t_span[1]
y_final = sol.sol(t_final)
CO_final = y_final[0]
products_final = y_final[1:]
# normalize product distribution for reporting
prod_frac = products_final / products_final.sum()

# Example output (production-ready code would return or save these arrays)
print("Final CO:", CO_final)
print("Product fractions (C1..C12):", np.round(prod_frac, 3))