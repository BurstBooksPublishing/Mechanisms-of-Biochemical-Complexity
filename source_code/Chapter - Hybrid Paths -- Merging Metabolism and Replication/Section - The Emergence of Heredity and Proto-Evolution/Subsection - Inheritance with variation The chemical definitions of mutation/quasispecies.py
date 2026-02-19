import numpy as np
from scipy.integrate import solve_ivp

# Build mutation matrix Q for binary alphabet with per-site fidelity q
def build_Q(num_types, L, q):
    # num_types must equal 2**L; types indexed by integers 0..2^L-1
    Q = np.zeros((num_types, num_types))
    for j in range(num_types):
        bits_j = np.array(list(np.binary_repr(j, width=L)), dtype=int)
        for i in range(num_types):
            bits_i = np.array(list(np.binary_repr(i, width=L)), dtype=int)
            mismatches = np.sum(bits_i != bits_j)
            Q[i, j] = (q**(L-mismatches)) * ((1-q)**mismatches)
    return Q

# Quasispecies RHS
def quasispecies_rhs(t, x, w, Q):
    Wx = w * x
    production = Q @ Wx
    wbar = np.dot(w, x)
    return production - wbar * x

# Example parameters: L=6, master index 0, master fitness A, others B
L = 6
num = 2**L
q = 0.98  # per-site fidelity
Q = build_Q(num, L, q)
A = 1.5
B = 1.0
w = np.full(num, B)
w[0] = A  # master type has higher fitness

# Initial condition: mostly mutants
x0 = np.full(num, 1.0/num)
t_span = (0.0, 500.0)
sol = solve_ivp(quasispecies_rhs, t_span, x0, args=(w, Q), rtol=1e-9, atol=1e-12)

# Output steady-state frequency of master
x_final = sol.y[:, -1]
print("Master frequency:", x_final[0])