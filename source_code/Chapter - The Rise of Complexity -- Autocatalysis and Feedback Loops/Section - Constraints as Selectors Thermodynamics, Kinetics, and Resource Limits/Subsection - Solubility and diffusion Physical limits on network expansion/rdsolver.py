import numpy as np
from scipy.integrate import solve_bvp

def steady_rd(x, y, p):
    # y[0]=c(x); second derivative = (k*c^2)/D
    D, k = p
    return np.vstack((y[1], (k/D)*y[0]**2))

def bc(ya, yb, p):
    # left boundary: c(0)=c_sat; right boundary: zero gradient at L
    c_sat = p[2]  # pass saturation as extra parameter
    return np.array([ya[0]-c_sat, yb[1]])

def compute_profile(L=1e-3, N=200, D=1e-9, k=1e3, c_sat=1e-3):
    x = np.linspace(0, L, N)
    # initial guess: linear decay
    y_init = np.vstack((c_sat*(1 - x/L), -c_sat/L*np.ones_like(x)))
    # pack parameters; solve_bvp expects p as array passed to funcs
    p = np.array([D, k, c_sat])
    sol = solve_bvp(lambda xi, yi: steady_rd(xi, yi, p[:2]),
                    lambda ya, yb: bc(ya, yb, p),
                    x, y_init, max_nodes=5000)
    if not sol.success:
        raise RuntimeError("BVP solver failed")
    return sol.x, sol.y[0]

# Example usage: vary k and plot penetration depth (not shown here).