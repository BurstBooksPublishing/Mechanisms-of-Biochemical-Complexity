# Minimal Gillespie sampler for three-species autocatalytic network.
# Production-ready: uses numpy, numba optional for speed; replace rates/topology as needed.

import numpy as np
from collections import Counter

def gillespie_sim(initial_state, reactions, rates, t_max, rng=None):
    # initial_state: dict {'A':nA, 'B':nB, ...}
    # reactions: list of (reactant_changes:dict, propensity_fn)
    state = dict(initial_state)
    t = 0.0
    rng = np.random.default_rng() if rng is None else rng
    history = []
    while t < t_max:
        props = np.array([r[1](state) * rates[i] for i, r in enumerate(reactions)])
        total = props.sum()
        if total <= 0:
            break
        dt = rng.exponential(1.0/total)
        t += dt
        # choose reaction
        r_idx = rng.choice(len(reactions), p=props/total)
        # apply stoichiometric changes
        for sp, d in reactions[r_idx][0].items():
            state[sp] = state.get(sp,0) + d
        history.append((t, dict(state)))
    return history

# Example network: monomer M inflow, polymer P formation catalyzed by P (autocatalysis), degradation
rates = {'inflow':1.0, 'poly_gen':1e-3, 'autocat':1e-2, 'deg':5e-3}
# reactions: (stoich_changes, propensity(state))
reactions = [
    ( {'M':+1}, lambda s: 1.0 ),                       # inflow of monomer
    ( {'M':-2, 'P':+1}, lambda s: s.get('M',0)*(s.get('M',0)-1)/2 ),  # spontaneous dimerization
    ( {'M':-1, 'P':+1}, lambda s: s.get('M',0)*s.get('P',0) ),       # autocatalytic growth
    ( {'P':-1}, lambda s: s.get('P',0) )               # degradation
]

# run ensemble, collect occupancy of P count as proxy for narrowness
def ensemble_stats(nruns=200, t_max=1000.0):
    counts = Counter()
    for _ in range(nruns):
        hist = gillespie_sim({'M':50,'P':1}, reactions, list(rates.values()), t_max)
        if hist:
            # take last state
            counts[hist[-1][1]['P']] += 1
    # empirical distribution over P counts
    total = sum(counts.values())
    return {k: v/total for k,v in counts.items()}

if __name__ == '__main__':
    dist = ensemble_stats()
    print('Empirical stationary distribution over P counts:', dist)