"""
Gillespie SSA for F -> R -> P with R <-> T reversible trapping.
Authoritative, production-ready with numpy.
"""
import numpy as np

def gillespie(Tmax, params, state0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t = 0.0
    state = np.array(state0, dtype=int)
    times = [t]
    traj = [state.copy()]
    kf, kp, kTR, kRT, k_loss = params
    while t < Tmax:
        F, R, T, P = state
        # propensity vector
        a = np.array([
            kf * F,          # F -> R
            kp * R,          # R -> P
            kTR * R,         # R -> T
            kRT * T,         # T -> R
            k_loss * R       # R -> loss/degradation
        ])
        a0 = a.sum()
        if a0 <= 0:
            break
        # time to next reaction
        dt = rng.exponential(1.0 / a0)
        t += dt
        # choose reaction
        r = rng.random()
        cum = np.cumsum(a) / a0
        rx = np.searchsorted(cum, r)
        # apply reaction stoichiometry
        if rx == 0:
            state[0] -= 1; state[1] += 1
        elif rx == 1:
            state[1] -= 1; state[3] += 1
        elif rx == 2:
            state[1] -= 1; state[2] += 1
        elif rx == 3:
            state[2] -= 1; state[1] += 1
        elif rx == 4:
            state[1] -= 1  # irreversible loss
        times.append(t)
        traj.append(state.copy())
    return np.array(times), np.array(traj)

# Example parameters and initial state
params = (1.0, 0.1, 0.01, 1e-4, 0.005)  # kf, kp, kTR, kRT, k_loss
state0 = (1000, 0, 0, 0)  # F, R, T, P
# run
times, traj = gillespie(1000.0, params, state0)