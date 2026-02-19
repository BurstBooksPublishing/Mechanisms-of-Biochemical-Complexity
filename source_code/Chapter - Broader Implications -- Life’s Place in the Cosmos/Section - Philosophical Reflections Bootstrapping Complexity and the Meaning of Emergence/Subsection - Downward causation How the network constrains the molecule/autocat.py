import numpy as np
from scipy.integrate import solve_ivp

# Stoichiometric matrix S: rows species A,B,C; cols reactions r1,r2,r3
S = np.array([[-1,  1,  0],   # A consumed in r1, produced in r2
              [ 1, -1,  1],   # B produced in r1, consumed in r2, produced in r3 (autocatalysis)
              [ 0,  0, -1]])  # C consumed in r3 (sink)

# Mass-action rate function v(x) = k * prod(reactants)
k = np.array([1.0, 0.5, 2.0])  # rate constants

def rates(x):
    A, B, C = x
    # r1: A + catalyst -> B (catalyst assumed implicit)
    r1 = k[0] * A
    # r2: B -> A  (reversible leak)
    r2 = k[1] * B
    # r3: B + A -> 2 B  (autocatalytic step: net B production)
    r3 = k[2] * B * A
    return np.array([r1, r2, r3])

def dxdt(t, x):
    return S.dot(rates(x))

# simulate two different initial conditions
t_span = (0, 50)
x0_list = [np.array([1e-3, 1e-3, 0.0]), np.array([1.0, 1e-3, 0.0])]

sols = [solve_ivp(dxdt, t_span, x0, dense_output=True) for x0 in x0_list]

# After simulation, inspect steady states (last value)
for i, sol in enumerate(sols):
    print(f"IC {i}: final concentrations A,B,C = {sol.y[:, -1]}")
# Optional: export sol.y and sol.t for plotting externally