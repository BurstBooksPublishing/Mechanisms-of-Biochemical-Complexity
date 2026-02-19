import numpy as np

def expected_cost(code_map, codon_freq, mut_matrix, dist_matrix):
    """
    Compute expected translational error cost E[f] (Eq. 1).
    code_map: array of length Ncodon with amino acid indices.
    codon_freq: length-Ncodon array q(c), sums to 1.
    mut_matrix: Ncodon x Ncodon mutation/mistranslation probabilities P(c->c').
    dist_matrix: Naamino x Naamino distance d(a,a').
    """
    # Validate shapes
    Ncod = mut_matrix.shape[0]
    if code_map.shape[0] != Ncod or codon_freq.shape[0] != Ncod:
        raise ValueError("Inconsistent codon dimensions")
    # Map codons to amino acid indices matrix
    aa_i = code_map[:, None]
    aa_j = code_map[None, :]
    # Distance matrix for codon pairs via their assigned amino acids
    d_c = dist_matrix[aa_i, aa_j]
    # Elementwise product q(c) * P(c->c') * d(f(c),f(c'))
    term = codon_freq[:, None] * mut_matrix * d_c
    return float(term.sum())

# Example usage (toy arrays not shown): generate random code_map permutations
# and compare expected_cost to canonical mapping to compute z-score.