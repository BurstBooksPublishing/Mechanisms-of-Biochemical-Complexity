import math

# Example nearest-neighbor stacking energies (kcal/mol) at 37°C; extend for production.
NN = {
    ('GC','GC'):-3.4, ('CG','CG'):-2.4, ('AU','AU'):-1.0, ('UA','UA'):-0.9,
    ('GC','CG'):-2.1, ('CG','GC'):-2.3, ('GU','UG'):-1.5, ('UG','GU'):-1.4
}

R = 1.987e-3  # kcal/(mol·K)
T = 310.15    # K (37°C)

def loop_entropy_penalty(n, c=1.7, n0=3, init=3.0):
    """Return loop free-energy penalty (kcal/mol)."""
    return init + R*T*c*math.log(max(n,1)/n0)

def hairpin_dG(stem_pairs, loop_len, nn_table=NN):
    """
    Compute $\Delta$G for a hairpin.
    stem_pairs: list of adjacent base-pair tuples, e.g. [('GC','CG'), ...].
    loop_len: number of unpaired nucleotides in the terminal loop.
    """
    dg_stack = 0.0
    for pair in stem_pairs:
        dg_stack += nn_table.get(pair, -2.0)  # fallback average energy
    dg_loop = loop_entropy_penalty(loop_len)
    # simple ion term approximation (monovalent salt effect omitted here)
    dg_ion = 0.0
    return dg_stack + dg_loop + dg_ion

# Usage example: 5 base-pair stem with a tetraloop.
example_stem = [('GC','CG')]*5
print("Estimated $\Delta$ G (kcal/mol):", hairpin_dG(example_stem, loop_len=4))