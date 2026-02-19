import numpy as np
from scipy.integrate import solve_ivp

# Model parameters (set to plausible prebiotic values)
params = {
    'k_s': 1e-6,    # spontaneous formation rate (M/s)
    'k_a': 1e-2,    # autocatalytic rate constant (M^-1 s^-1)
    'k_i': 1e-1,    # heterodimerization rate (M^-1 s^-1)
    'k_d': 1e-4,    # first-order loss (s^-1)
    'A0' : 1e-3     # precursor concentration (M)
}

def cpl_bias(t):
    # returns differential photolysis rate favoring L over D (s^-1)
    # positive -> preferentially destroys D, biasing L
    if 1e3 < t < 2e3:
        return 1e-5
    return 0.0

def rhs(t, y, p):
    L, D = y
    A = p['A0']  # assume fast reservoir; replace with ODE if needed
    bias = cpl_bias(t)
    # differential photolysis: D decays faster by 'bias'
    dL = p['k_s']*A + p['k_a']*A*L - p['k_i']*L*D - p['k_d']*L
    dD = p['k_s']*A + p['k_a']*A*D - p['k_i']*L*D - (p['k_d']+bias)*D
    return [dL, dD]

# initial conditions near racemic with tiny fluctuation
y0 = [1e-9*(1+1e-6), 1e-9*(1-1e-6)]
t_span = (0.0, 5e4)
sol = solve_ivp(rhs, t_span, y0, args=(params,), dense_output=True, atol=1e-12, rtol=1e-9)

# compute ee and totals
t = np.linspace(*t_span, 2000)
L, D = sol.sol(t)
total = L + D
ee = np.where(total>0, (L-D)/total, 0.0)
# results arrays: t, L, D, total, ee