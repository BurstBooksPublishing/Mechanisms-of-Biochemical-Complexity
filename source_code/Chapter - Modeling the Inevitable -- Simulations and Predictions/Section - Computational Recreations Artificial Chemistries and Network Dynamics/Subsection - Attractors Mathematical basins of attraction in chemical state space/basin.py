import numpy as np
from scipy.integrate import solve_ivp

# model parameters (tune for bistability)
alpha_u, alpha_v = 10.0, 10.0
gamma_u, gamma_v = 1.0, 1.0
m, n = 4.0, 4.0

def toggle_rhs(t, y):
    u, v = y
    du = alpha_u / (1.0 + v**n) - gamma_u * u
    dv = alpha_v / (1.0 + u**m) - gamma_v * v
    return [du, dv]

def classify_final(sol, thresh=0.0):
    u_final, v_final = sol.y[0, -1], sol.y[1, -1]
    return 0 if (u_final - v_final) > thresh else 1

def compute_basin(xmin, xmax, ymin, ymax, nx, ny, t_span=(0,200), rtol=1e-8, atol=1e-10):
    ux = np.linspace(xmin, xmax, nx)
    vy = np.linspace(ymin, ymax, ny)
    basin = np.empty((ny, nx), dtype=int)
    for i, v0 in enumerate(vy):
        for j, u0 in enumerate(ux):
            sol = solve_ivp(toggle_rhs, t_span, [u0, v0], rtol=rtol, atol=atol, dense_output=False)
            basin[i, j] = classify_final(sol)
    return ux, vy, basin

# example usage (adjust grid and bounds as needed)
# ux, vy, basin = compute_basin(0.0, 5.0, 0.0, 5.0, nx=200, ny=200)