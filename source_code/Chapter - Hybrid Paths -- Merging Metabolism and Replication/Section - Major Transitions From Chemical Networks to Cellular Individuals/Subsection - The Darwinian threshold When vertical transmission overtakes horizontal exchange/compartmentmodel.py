import numpy as np
import random

# Parameters
N_compartments = 200            # number of compartments
init_templates = 5              # initial templates per compartment
r_replication = 0.5             # replication rate per template per dt
h_transfer = 0.05               # horizontal transfer probability per template per dt
division_threshold = 20         # total molecules to trigger division
dt = 0.1
steps = 10000

# State: list of compartments; each compartment is dict{'templates': {lineage_id:count}}
comps = [ {'templates': {i: init_templates}} for i in range(N_compartments) ]

def step(comps):
    # replication events
    for c in comps:
        total = sum(c['templates'].values())
        # replication attempts proportional to counts
        for lineage, count in list(c['templates'].items()):
            n_new = np.random.binomial(count, 1 - np.exp(-r_replication*dt))
            if n_new:
                c['templates'][lineage] = c['templates'].get(lineage,0) + n_new
        # horizontal transfer: each template may move to random compartment
        for lineage, count in list(c['templates'].items()):
            n_transfer = np.random.binomial(count, 1 - np.exp(-h_transfer*dt))
            if n_transfer:
                c['templates'][lineage] -= n_transfer
                target = random.choice(comps)
                target['templates'][lineage] = target['templates'].get(lineage,0) + n_transfer
        # clean zeros
        zeros = [k for k,v in c['templates'].items() if v<=0]
        for k in zeros: del c['templates'][k]
    # division: split compartments exceeding threshold
    new_comps = []
    for c in comps:
        if sum(c['templates'].values()) >= division_threshold:
            # stochastic partitioning
            childA, childB = {}, {}
            for lineage, count in c['templates'].items():
                # binomial partition
                a = np.random.binomial(count, 0.5)
                b = count - a
                if a: childA[lineage]=a
                if b: childB[lineage]=b
            new_comps.extend([{'templates':childA},{'templates':childB}])
        else:
            new_comps.append(c)
    # enforce fixed population by random culling
    random.shuffle(new_comps)
    return new_comps[:N_compartments]

# Run and record lineage persistence metric
lineage_of_interest = 0
history = []
for t in range(steps):
    comps = step(comps)
    # fraction of compartments containing lineage_of_interest
    frac = sum(1 for c in comps if lineage_of_interest in c['templates']) / N_compartments
    history.append(frac)
# 'history' contains temporal signal of vertical persistence for analysis