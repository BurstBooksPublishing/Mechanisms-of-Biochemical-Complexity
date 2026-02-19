import numpy as np
import networkx as nx
from math import log, exp

def shannon_normalized(concentrations):
    # concentrations: 1D numpy array, nonnegative
    total = concentrations.sum()
    if total <= 0:
        return 0.0
    p = concentrations / total
    p = p[p>0]
    H = -np.sum(p * np.log(p))
    N = concentrations.size
    return H / np.log(N)  # normalized to [0,1]

def scc_fraction(digraph):
    # digraph: NetworkX DiGraph representing catalytic edges
    if digraph.number_of_nodes() == 0:
        return 0.0
    sccs = list(nx.strongly_connected_components(digraph))
    largest = max((len(s) for s in sccs), default=0)
    return largest / digraph.number_of_nodes()

def entropy_production_estimate(flux_matrix, chem_potentials):
    # flux_matrix: numpy array (i->j flux), chem_potentials: vector
    # coarse estimate: sum_jk J_jk * (mu_j - mu_k) / T ; assume T=1
    J = flux_matrix
    mu = chem_potentials
    dmu = mu[:, None] - mu[None, :]
    return max(0.0, np.sum(J * dmu))  # nonnegative proxy

def compute_life_score(digraph, concentrations, flux_matrix=None, chem_potentials=None,
                       weights=None, sigmoid_params=(10.0, 0.5)):
    # default weights for [I, M, A]
    if weights is None:
        weights = np.array([0.4, 0.4, 0.2])
    I = shannon_normalized(np.asarray(concentrations))
    M = scc_fraction(digraph)
    if flux_matrix is None or chem_potentials is None:
        A = 0.0
    else:
        ep = entropy_production_estimate(flux_matrix, chem_potentials)
        # normalize A by a robust scale (median or supplied scale)
        A = np.tanh(ep / (1.0 + ep))  # maps to [0,1)
    raw = np.dot(weights, np.array([I, M, A]))
    alpha, beta = sigmoid_params
    L = 1.0 / (1.0 + exp(-alpha * (raw - beta)))
    return {'L': L, 'I': I, 'M': M, 'A': A}
# Example usage: build catalysis DiGraph, supply concentrations, optional fluxes.