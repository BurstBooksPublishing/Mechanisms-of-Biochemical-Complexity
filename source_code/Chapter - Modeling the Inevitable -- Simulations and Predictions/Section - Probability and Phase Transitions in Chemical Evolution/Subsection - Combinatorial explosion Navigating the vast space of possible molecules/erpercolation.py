#!/usr/bin/env python3
# Estimate sequence counts and compute largest-component fraction vs p.
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sequence_space_size(a:int, L:int) -> float:
    # return a**L, careful with large L -> use log for safety
    return math.exp(L * math.log(a))

def er_largest_component_fraction(n:int, p:float, seed=None) -> float:
    # build ER graph and return fraction of nodes in largest connected component
    G = nx.fast_gnp_random_graph(n, p, seed=seed, directed=False)
    if n == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    return largest / n

if __name__ == "__main__":
    # Parameters
    alphabet = 20           # amino acids
    lengths = [10, 20, 50]  # peptide lengths to report
    n_nodes = 10000         # surrogate chemical subspace size
    ps = np.logspace(-6, -2, 20)  # p sweep across percolation region

    for L in lengths:
        S = sequence_space_size(alphabet, L)
        print(f"L={L}, sequences ~ {S:.3e}")

    fractions = [er_largest_component_fraction(n_nodes, p, seed=42) for p in ps]

    # Plot (optional visualization)
    plt.semilogx(ps, fractions, marker='o')
    plt.axvline(1/(n_nodes-1), color='red', linestyle='--', label='theoretical p_c')
    plt.xlabel('edge probability p')
    plt.ylabel('largest component fraction')
    plt.title('ER percolation in surrogate chemical network')
    plt.legend()
    plt.grid(True)
    plt.show()