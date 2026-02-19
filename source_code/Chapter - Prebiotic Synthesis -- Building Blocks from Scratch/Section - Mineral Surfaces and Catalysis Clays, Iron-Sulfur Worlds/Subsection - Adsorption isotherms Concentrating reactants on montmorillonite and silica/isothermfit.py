import numpy as np
from scipy.optimize import curve_fit

# Langmuir and Freundlich models
def langmuir(C, qmax, K):
    return qmax * (K * C) / (1 + K * C)

def freundlich(C, Kf, n):
    return Kf * C**(1.0 / n)

# Example data arrays: C (mol L^-1), q (mol kg^-1)
C = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4])
q = np.array([1e-8, 6e-8, 1.1e-7, 4.9e-7, 9.2e-7])

# Initial guesses and bounds for robust fitting
p0_langmuir = [1e-3, 1e6]  # qmax, K
bounds_langmuir = ([0, 0], [np.inf, np.inf])

p0_freundlich = [1e-7, 2.0]  # Kf, n
bounds_freundlich = ([0, 0.5], [np.inf, 10])

# Fit Langmuir
params_L, cov_L = curve_fit(langmuir, C, q, p0=p0_langmuir, bounds=bounds_langmuir)
qmax_fit, K_fit = params_L
# Fit Freundlich
params_F, cov_F = curve_fit(freundlich, C, q, p0=p0_freundlich, bounds=bounds_freundlich)
Kf_fit, n_fit = params_F

# Print results (real code should log or return these)
print(f"Langmuir: qmax={qmax_fit:.3e} mol/kg, K={K_fit:.3e} L/mol")
print(f"Freundlich: Kf={Kf_fit:.3e}, n={n_fit:.3f}")