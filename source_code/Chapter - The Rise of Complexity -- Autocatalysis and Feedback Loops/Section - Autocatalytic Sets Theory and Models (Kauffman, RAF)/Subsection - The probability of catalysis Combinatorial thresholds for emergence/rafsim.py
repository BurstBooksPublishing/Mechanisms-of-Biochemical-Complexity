import random
import numpy as np
import networkx as nx

def make_random_network(n_mol, n_rxn, p, food_indices, rng=None):
    rng = random if rng is None else rng
    # molecules indexed 0..n_mol-1, reactions indexed 0..n_rxn-1
    # reactants and products sampled deterministically for reproducibility
    reactants = [tuple(rng.choices(range(n_mol), k=2)) for _ in range(n_rxn)]
    products  = [rng.choice(range(n_mol)) for _ in range(n_rxn)]
    catalysis = {(i,j): (rng.random() < p) for i in range(n_mol) for j in range(n_rxn)}
    return reactants, products, catalysis

def is_raf(reactants, products, catalysis, food_set):
    n_rxn = len(reactants)
    available = set(food_set)
    used = set()
    changed = True
    while changed:
        changed = False
        for j in range(n_rxn):
            if j in used: 
                continue
            a,b = reactants[j]
            # reaction can proceed only if reactants available and catalyzed
            if a in available and b in available:
                # check if any available molecule catalyzes reaction j
                if any(catalysis.get((i,j), False) and i in available for i in available):
                    available.add(products[j])
                    used.add(j)
                    changed = True
    # RAF exists if at least one reaction can be added beyond food
    return len(used) > 0

# Example usage
if __name__ == '__main__':
    rng = random.Random(42)
    n_mol, n_rxn = 500, 2000
    food = set(range(5))                       # small food set
    for p in np.linspace(0.0005, 0.02, 20):
        react, prod, cat = make_random_network(n_mol, n_rxn, p, food, rng=rng)
        if is_raf(react, prod, cat, food):
            print(f'p={p:.4f} -> RAF detected')
        else:
            print(f'p={p:.4f} -> no RAF')