import numpy as np
import math
import random

# Parameters (physically motivated): concentrations in mol L^-1, rates in L mol^-1 s^-1 or s^-1
k_attach = 1e3       # bimolecular attachment (enhanced by water-mediated proton transfer)
k_hydro   = 1e-2     # unimolecular hydrolysis (water-catalyzed)
M0        = 1e-3     # initial monomer concentration
V         = 1e-15    # reaction volume (L), small compartment
max_time  = 1000.0

# State: dictionary mapping length -> molecule count (counts in integer molecules)
NA = 6.022e23
n_monomers = int(M0 * V * NA)
state = {1: n_monomers}  # all monomers initially

def propensity(state):
    props = []
    # attachment: any pair i + 1 -> i+1 (choose a chain and a monomer)
    total_chains = sum(state.get(l,0) for l in state)
    for l, count in list(state.items()):
        # bimolecular pair: chain (length l) + monomer (length 1)
        a = k_attach * (count) * state.get(1,0) / V  # convert to concentration terms implicitly
        if a>0: props.append(('attach', l, a))
        # hydrolysis: breaks a chain of length l into two pieces; model as single-site cleavage
        if l>1:
            b = k_hydro * count
            props.append(('hydro', l, b))
    return props

t = 0.0
history = []
while t < max_time:
    props = propensity(state)
    if not props: break
    a_total = sum(p[2] for p in props)
    r1 = random.random()
    dt = -math.log(r1) / a_total
    t += dt
    # select reaction
    r2 = random.random() * a_total
    cum = 0.0
    for rx in props:
        cum += rx[2]
        if r2 <= cum:
            kind, l, _ = rx
            break
    # execute
    if kind == 'attach':
        # consume one monomer and one chain of length l -> chain length l+1
        if state.get(1,0)>0 and state.get(l,0)>0:
            state[1] -= 1
            state[l] -= 1
            state[l+1] = state.get(l+1,0) + 1
    else:  # hydrolysis: break chain of length l into (i, l-i) uniformly
        if state.get(l,0)>0:
            state[l] -= 1
            i = random.randint(1, l-1)
            state[i] = state.get(i,0) + 1
            state[l-i] = state.get(l-i,0) + 1
    history.append((t, dict(state)))
# Postprocess: compute mean chain length trajectory externally.