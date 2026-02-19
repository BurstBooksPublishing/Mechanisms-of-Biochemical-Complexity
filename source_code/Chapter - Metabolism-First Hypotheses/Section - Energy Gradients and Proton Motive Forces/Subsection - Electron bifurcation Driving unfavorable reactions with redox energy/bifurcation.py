"""
Compute bifurcation energetics and feasibility.
Inputs are potentials in volts versus the same reference.
"""
import numpy as np
from typing import Tuple

FARADAY = 96485.33289  # C mol^-1
R = 8.314462618  # J mol^-1 K^-1

def bifurcation_feasible(E_A: float, E_B: float, E_D: float, T: float = 298.15) -> Tuple[float, bool]:
    """
    E_A: exergonic acceptor potential (V)
    E_B: endergonic acceptor potential (V)
    E_D: donor midpoint potential (V)
    T: temperature (K)
    Returns (deltaG_kJ_per_mol, feasible)
    """
    # total free energy for two one-electron transfers (J mol^-1)
    deltaE_total = (E_A - E_D) + (E_B - E_D)
    deltaG_J = -FARADAY * deltaE_total
    deltaG_kJ = deltaG_J / 1000.0
    feasible = deltaG_J < 0  # net free energy negative
    return deltaG_kJ, feasible

# Example: E_A = -0.32 V (NADP+/NADPH-like), E_B = -0.45 V (ferredoxin-like), E_D = -0.30 V (H2/H+ approx)
if __name__ == "__main__":
    dG, ok = bifurcation_feasible(E_A=-0.32, E_B=-0.45, E_D=-0.30)
    print(f"$\Delta$G_tot = {dG:.2f} kJ mol^-1, feasible: {ok}")