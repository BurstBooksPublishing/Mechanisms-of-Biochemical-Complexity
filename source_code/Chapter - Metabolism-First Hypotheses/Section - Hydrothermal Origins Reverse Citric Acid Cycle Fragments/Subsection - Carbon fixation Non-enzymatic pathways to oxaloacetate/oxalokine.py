import numpy as np
from scipy.integrate import solve_ivp

# Parameters (example values; calibrate to experiment)
k_forward = 1.0    # M^-1 s^-1, aldol formation scaled by metal
k_dec = 1e-3       # s^-1, oxaloacetate decarboxylation
metal_factor = 10.0  # dimensionless catalyst enhancement

# ODE system: [pyruvate, glyoxylate, oxaloacetate]
def rhs(t, y):
    P, G, O = y
    v_form = k_forward * metal_factor * P * G   # mass-action
    v_dec = k_dec * O
    dP = -v_form + v_dec        # pyruvate consumed and reformed
    dG = -v_form                # glyoxylate consumed
    dO = +v_form - v_dec        # oxaloacetate formed and decarboxylated
    return [dP, dG, dO]

# Initial concentrations (M)
y0 = [1e-3, 1e-3, 0.0]
t_span = (0, 1e4)  # seconds
sol = solve_ivp(rhs, t_span, y0, atol=1e-12, rtol=1e-9)

# Post-process (example): final yields
final = sol.y[:, -1]
print("Final [P], [G], [O] (M):", final)