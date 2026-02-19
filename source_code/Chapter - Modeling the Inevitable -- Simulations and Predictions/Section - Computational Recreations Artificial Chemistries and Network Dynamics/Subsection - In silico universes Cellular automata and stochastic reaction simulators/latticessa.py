import numpy as np
import math
import random

# Parameters (tunable)
Lx, Ly = 50, 50                   # lattice dimensions
c_autocat = 1.0                   # rate for A + B -> 2A
c_decay = 0.1                     # rate for A -> nothing
c_influx = 0.01                   # per-voxel B influx rate
D_A, D_B = 0.2, 0.2               # diffusion hop rates per particle
max_time = 100.0

# Initialize state arrays: integer counts per voxel
A = np.zeros((Lx, Ly), dtype=int)
B = np.zeros((Lx, Ly), dtype=int)
# seed a small A cluster and uniform B
A[Lx//2, Ly//2] = 5
B[:, :] = 10

def neighbors(x, y):
    # von Neumann periodic neighbors
    return [((x-1)%Lx, y), ((x+1)%Lx, y), (x, (y-1)%Ly), (x, (y+1)%Ly)]

def compute_propensities():
    # returns flat list of propensities and event descriptors
    props = []
    events = []
    for x in range(Lx):
        for y in range(Ly):
            nA = int(A[x, y]); nB = int(B[x, y])
            # autocatalysis
            a1 = c_autocat * nA * nB
            if a1 > 0:
                props.append(a1); events.append(('auto', x, y))
            # decay
            a2 = c_decay * nA
            if a2 > 0:
                props.append(a2); events.append(('decay', x, y))
            # influx (modeled as Poisson source)
            a3 = c_influx
            props.append(a3); events.append(('influx', x, y))
            # diffusion hops for A and B to neighbors
            if nA > 0:
                a4 = D_A * nA
                props.append(a4); events.append(('diffA', x, y))
            if nB > 0:
                a5 = D_B * nB
                props.append(a5); events.append(('diffB', x, y))
    return np.array(props, dtype=float), events

t = 0.0
rng = random.random
history = [(t, A.sum(), B.sum())]

while t < max_time:
    props, events = compute_propensities()
    a0 = props.sum()
    if a0 <= 0:
        break
    # time to next event (exponential)
    tau = -math.log(rng()) / a0
    t += tau
    # choose event index
    r = rng() * a0
    idx = np.searchsorted(props.cumsum(), r)
    etype, x, y = events[idx]
    if etype == 'auto':
        # A + B -> 2A
        if A[x, y] > 0 and B[x, y] > 0:
            A[x, y] += 1; B[x, y] -= 1
    elif etype == 'decay':
        if A[x, y] > 0:
            A[x, y] -= 1
    elif etype == 'influx':
        B[x, y] += 1
    elif etype == 'diffA':
        # choose neighbor and hop one A particle
        if A[x, y] > 0:
            nx, ny = random.choice(neighbors(x, y))
            A[x, y] -= 1; A[nx, ny] += 1
    elif etype == 'diffB':
        if B[x, y] > 0:
            nx, ny = random.choice(neighbors(x, y))
            B[x, y] -= 1; B[nx, ny] += 1
    history.append((t, A.sum(), B.sum()))
# history contains time series of global counts for analysis/visualization