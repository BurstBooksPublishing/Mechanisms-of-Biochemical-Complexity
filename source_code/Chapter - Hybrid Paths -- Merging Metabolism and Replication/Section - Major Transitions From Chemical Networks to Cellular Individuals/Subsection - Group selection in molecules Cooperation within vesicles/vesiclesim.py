import numpy as np
# Parameters
num_vesicles = 200               # number of vesicles in population
N_init = 50                      # initial molecules per vesicle
cost = 0.05                      # cost c for cooperators
benefit = 0.3                    # per-capita catalytic benefit b
B = 2.0                          # vesicle-level growth coupling
division_threshold = 2 * N_init  # molecule count triggering division
migration_rate = 0.01            # probability per molecule to migrate per step
timesteps = 10000

# Initialize vesicles: fraction 0.1 cooperators randomly
vesicles = [np.random.binomial(N_init, 0.1, size=1)[0] for _ in range(num_vesicles)]
# Track composition as tuple (n_cooperators, n_defectors)
pop = [(int(n), N_init-int(n)) for n in vesicles]

rng = np.random.default_rng(42)

def step(pop):
    new_pop = []
    for nC, nD in pop:
        N = nC + nD
        if N == 0:
            new_pop.append((0,0)); continue
        x = nC / N
        # molecule replication probabilities proportional to fitness from Eq. (2)
        wC = 1.0 - cost + benefit * x
        wD = 1.0 + benefit * x
        rates = np.array([wC, wD])
        rates /= rates.sum()
        # multinomial births conserving expected N; small stochasticity via Poisson
        births = rng.multinomial(np.random.poisson(1), rates)
        nC += births[0]; nD += births[1]
        # migration: simple exchange with global pool
        migrantsC = rng.binomial(nC, migration_rate)
        migrantsD = rng.binomial(nD, migration_rate)
        nC -= migrantsC; nD -= migrantsD
        # collect migrants to re-distribute later
        new_pop.append((nC, nD))
    # redistribute migrants uniformly
    # (left as an exercise to implement exact mass conservation)
    return new_pop

# Main loop with simple division rule
for t in range(timesteps):
    pop = step(pop)
    # vesicle growth and division according to vesicle fitness W_g = 1 + B*x
    updated = []
    for nC, nD in pop:
        N = nC + nD
        if N == 0:
            updated.append((0,0)); continue
        x = nC / N
        # expected growth proportional to W_g
        growth_factor = 1.0 + B * x
        # stochastic growth
        added = rng.poisson((growth_factor-1.0)*N)
        # distribute added molecules proportional to current composition
        if N>0 and added>0:
            addC = rng.binomial(added, x)
            nC += int(addC); nD += int(added - addC)
        # division if above threshold
        if nC + nD >= division_threshold:
            # bottleneck: split molecules randomly into two daughters
            total = nC + nD
            draw = rng.hypergeometric(nC, nD, total//2)
            updated.append((int(draw), int(total//2 - draw)))
            updated.append((nC - int(draw), nD - int(total//2 - draw)))
        else:
            updated.append((nC, nD))
    pop = updated
# At this point analyze pop composition distributions for cooperation prevalence.