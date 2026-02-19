import numpy as np
import random

# Species indices: 0:A (feedstock), 1:B (intermediate), 2:C (autocatalyst)
rates = {
    'A_to_B': 1.0,      # unimolecular formation on surface (s^-1)
    'B_to_C': 0.1,      # conversion (s^-1)
    'C_autocat': 10.0,  # autocatalytic production: B + C -> 2C (M^-1 s^-1 effective)
    'loss_C': 0.01      # dilution/decay of C (s^-1)
}
V = 1e-15  # compartment volume in liters (sets stochastic scale)
state = np.array([100, 0, 1], dtype=int)  # initial molecule counts

def propensities(s):
    a = []
    a.append(rates['A_to_B'] * s[0])                       # A -> B
    a.append(rates['B_to_C'] * s[1])                       # B -> C
    a.append(rates['C_autocat'] * s[1] * s[2] / V)         # B + C -> 2C
    a.append(rates['loss_C'] * s[2])                      # C -> ∅
    return np.array(a)

t, tmax = 0.0, 1000.0
traj = [(t, state.copy())]
while t < tmax:
    a = propensities(state)
    a0 = a.sum()
    if a0 <= 0:
        break
    r1, r2 = random.random(), random.random()
    tau = -np.log(r1) / a0
    cum = np.cumsum(a)
    reaction = int(np.searchsorted(cum, r2 * a0))
    # Apply reaction stoichiometry
    if reaction == 0:
        state[0] -= 1; state[1] += 1
    elif reaction == 1:
        state[1] -= 1; state[2] += 1
    elif reaction == 2:
        state[1] -= 1; state[2] += 1  # B consumed, C increases by 1
    elif reaction == 3:
        state[2] -= 1
    t += tau
    traj.append((t, state.copy()))
# traj contains time series for analysis