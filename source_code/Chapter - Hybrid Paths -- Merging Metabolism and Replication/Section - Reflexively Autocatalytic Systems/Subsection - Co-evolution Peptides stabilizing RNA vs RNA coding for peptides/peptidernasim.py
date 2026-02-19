"""
Simulate the minimal peptide-RNA model from Eq. (1).
Requires: numpy, scipy, matplotlib.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f_fidelity(R, q=0.98, L=10):
    # simple fidelity model: q^L multiplied by template availability
    return (q**L)

def rhs(t, y, params):
    R, P = y
    sR, sP, kR, alpha, K, muR, kP, muP, q, L = params
    f = f_fidelity(R, q=q, L=L)
    dR = sR + kR*R*(1.0 + (alpha*P)/(K+P)) - muR*R
    dP = sP + kP*f*R - muP*P
    return [dR, dP]

def simulate(params, y0=(1e-6, 1e-6), t_span=(0,1000), dense=1000):
    t_eval = np.linspace(t_span[0], t_span[1], dense)
    sol = solve_ivp(rhs, t_span, y0, args=(params,), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y

# Example parameters calibrated to test mutualism
params = (1e-6, 0.0, 0.5, 5.0, 1.0, 0.1, 0.2, 0.05, 0.98, 12)
t, y = simulate(params)
R, P = y

plt.semilogy(t, R, label='RNA (R)')
plt.semilogy(t, P, label='Peptide (P)')
plt.xlabel('time')
plt.ylabel('concentration')
plt.legend()
plt.show()