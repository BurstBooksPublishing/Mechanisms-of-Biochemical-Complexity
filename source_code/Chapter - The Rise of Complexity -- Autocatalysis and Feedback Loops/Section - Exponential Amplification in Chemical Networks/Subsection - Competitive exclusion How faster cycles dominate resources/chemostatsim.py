import numpy as np
from scipy.integrate import solve_ivp

# Parameters: feed F, dilution D, rates k1,k2, decay d1,d2
params = dict(F=1.0, D=0.1, k1=2.0, k2=1.0, d1=0.05, d2=0.05)

def rhs(t, y):
    S, X1, X2 = y
    F, D, k1, k2, d1, d2 = (params[k] for k in ('F','D','k1','k2','d1','d2'))
    dS = F - D*S - k1*S*X1 - k2*S*X2
    dX1 = k1*S*X1 - (d1 + D)*X1
    dX2 = k2*S*X2 - (d2 + D)*X2
    return [dS, dX1, dX2]

y0 = [params['F']/params['D'], 1e-3, 1e-3]  # initial S near feed equilibrium, tiny seeds
t_span = (0, 500)
sol = solve_ivp(rhs, t_span, y0, rtol=1e-8, atol=1e-12)

S_final, X1_final, X2_final = sol.y[:,-1]
print(f"Final S = {S_final:.6f}, X1 = {X1_final:.6f}, X2 = {X2_final:.6f}")
# For analysis, compare k/(d+D) ratios to predict winner.
ratio1 = params['k1']/(params['d1'] + params['D'])
ratio2 = params['k2']/(params['d2'] + params['D'])
print(f"Selection ratios: X1 {ratio1:.3f}, X2 {ratio2:.3f}")