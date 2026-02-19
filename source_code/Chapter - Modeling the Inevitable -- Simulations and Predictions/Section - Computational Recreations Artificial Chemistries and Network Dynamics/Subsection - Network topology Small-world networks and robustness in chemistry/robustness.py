#!/usr/bin/env python3
"""
Small-world and robustness analysis for chemical reaction networks.
Requires: networkx, numpy, matplotlib
"""
from typing import Iterable, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def small_worldness(G: nx.Graph) -> Tuple[float, float, float]:
    """Return (C, L, sigma) for graph G."""
    if not nx.is_connected(G):
        # compute L on largest component for comparability
        Gc = G.subgraph(max(nx.connected_components(G), key=len))
    else:
        Gc = G
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(Gc)
    # random graph with same degree sequence via configuration model (simple approx)
    deg_seq = [d for _, d in G.degree()]
    Gr = nx.configuration_model(deg_seq, seed=42)
    Gr = nx.Graph(Gr)  # strip parallel edges and self-loops
    Gr.remove_edges_from(nx.selfloop_edges(Gr))
    Cr = nx.average_clustering(Gr)
    # handle possible disconnectedness in Gr
    if nx.is_connected(Gr):
        Lr = nx.average_shortest_path_length(Gr)
    else:
        Lr = nx.average_shortest_path_length(Gr.subgraph(
            max(nx.connected_components(Gr), key=len)))
    sigma = (C / Cr) / (L / Lr) if Cr > 0 and Lr > 0 else float('nan')
    return C, L, sigma

def robustness_curve(G: nx.Graph, fractions: Iterable[float], targeted: bool=False) -> np.ndarray:
    """Return array S(f) for given removal fractions. targeted=True removes highest-degree nodes first."""
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    S = np.zeros(len(fractions))
    for i, f in enumerate(fractions):
        k = int(round(f * N))
        if targeted:
            # remove top-degree nodes
            remove = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)][:k]
        else:
            rng = np.random.default_rng(42 + i)
            remove = list(rng.choice(nodes, size=k, replace=False))
        H = G.copy()
        H.remove_nodes_from(remove)
        if H.number_of_nodes() == 0:
            S[i] = 0.0
        else:
            S[i] = len(max(nx.connected_components(H), key=len)) / N
    return S

# Example usage (can be placed under if __name__ == '__main__'):
G = nx.watts_strogatz_graph(500, 6, 0.1, seed=1)  # surrogate chemical network
C, L, sigma = small_worldness(G)
fractions = np.linspace(0.0, 0.5, 51)
S_random = robustness_curve(G, fractions, targeted=False)
S_targeted = robustness_curve(G, fractions, targeted=True)
# plotting
plt.plot(fractions, S_random, label='random removal')
plt.plot(fractions, S_targeted, label='targeted removal')
plt.xlabel('fraction removed'); plt.ylabel('normalized giant component S(f)')
plt.legend(); plt.title(f'C={C:.3f}, L={L:.2f}, sigma={sigma:.2f}')
plt.show()