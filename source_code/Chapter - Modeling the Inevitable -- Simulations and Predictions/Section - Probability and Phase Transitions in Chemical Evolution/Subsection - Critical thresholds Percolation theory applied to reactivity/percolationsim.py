#!/usr/bin/env python3
"""
Estimate percolation threshold for catalytic networks.
Nodes: molecules m0..m{N-1}, reactions r0..r{R-1}.
Each reaction produces a single product chosen uniformly.
Each molecule catalyzes each reaction with probability p.
Project catalysis to molecule--molecule edges if molecule catalyzes
a reaction that produces another molecule.
"""
from typing import Tuple
import random
import numpy as np
import networkx as nx

def estimate_threshold(N: int, R: int, p_list: np.ndarray,
                       frac_thresh: float = 0.2, seed: int = None) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    # choose substrate pairs and product indices for reactions
    substrates = rng.integers(0, N, size=(R, 2))
    products = rng.integers(0, N, size=R)
    largest_fracs = np.zeros_like(p_list, dtype=float)
    for i, p in enumerate(p_list):
        # catalysis matrix: R x N boolean, True if molecule j catalyzes reaction i
        catalysis = rng.random(size=(R, N)) < p
        # build molecule projection: adjacency set
        G = nx.Graph()
        G.add_nodes_from(range(N))
        # for each reaction, link every catalyst to the product node
        for ri in range(R):
            prod = int(products[ri])
            catalysts = np.nonzero(catalysis[ri])[0]
            for c in catalysts:
                if c != prod:
                    G.add_edge(int(c), prod)
        # compute largest connected component fraction
        if G.number_of_nodes() == 0:
            largest_fracs[i] = 0.0
        else:
            comps = list(nx.connected_components(G))
            largest_fracs[i] = max(len(c) for c in comps) / N
    # find first p where fraction exceeds threshold
    idx = np.argmax(largest_fracs >= frac_thresh)
    p_crit = float(p_list[idx]) if largest_fracs[idx] >= frac_thresh else np.nan
    return p_crit, largest_fracs

# Example usage (run in scripts, not at import)
# if __name__ == "__main__":
#     p_vals = np.linspace(0, 0.02, 101)
#     p_crit, fracs = estimate_threshold(N=500, R=1000, p_list=p_vals, frac_thresh=0.1, seed=42)
#     print(f"Estimated p_crit={p_crit}")