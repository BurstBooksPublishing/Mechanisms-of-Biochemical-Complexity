"""
Simulate directed evolution on a population of real-valued phenotypes.
Mutation: Gaussian perturbation. Selection: top fraction survive.
Tracks best phenotype per round.
"""
import numpy as np

def directed_evolution(pop_size=10000, rounds=20, mut_sd=0.1, select_frac=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.normal(loc=0.0, scale=1.0, size=pop_size)  # initial phenotypes
    best_trace = np.empty(rounds)
    for t in range(rounds):
        # mutate
        mutants = pop + rng.normal(0, mut_sd, pop_size)
        # fitness: example Gaussian peak at 2.0 (target phenotype)
        fitness = np.exp(-0.5 * ((mutants - 2.0) / 0.5)**2)
        # select top fraction
        k = max(1, int(select_frac * pop_size))
        idx = np.argpartition(fitness, -k)[-k:]
        selected = mutants[idx]
        # amplify back to population size with replacement proportional to fitness
        probs = fitness[idx] / fitness[idx].sum()
        pop = np.random.choice(selected, size=pop_size, p=probs)
        best_trace[t] = pop.max()
    return best_trace

if __name__ == "__main__":
    trace = directed_evolution()
    print("Best phenotypes per round:", trace)