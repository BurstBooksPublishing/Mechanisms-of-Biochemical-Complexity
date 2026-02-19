import numpy as np

def fold_enhancement(phi: float, alpha: float = 3.0) -> float:
    """
    Estimate fold enhancement for bimolecular encounters in a crowded medium.
    phi: crowder volume fraction (0 <= phi < 1)
    alpha: empirical obstruction parameter (material dependent)
    Returns: multiplicative factor relative to dilute case.
    """
    if not (0.0 <= phi < 1.0):
        raise ValueError("phi must be in [0,1).")
    # empirical obstruction: diffusion reduction due to crowding
    g = np.exp(-alpha * phi / (1.0 - phi))
    # concentration increase due to excluded volume (Eq. eq:ceff)
    conc_factor = 1.0 / (1.0 - phi)
    return g * conc_factor

# Example: phi = 0.25, alpha = 3 (typical polymer crowding)
phi = 0.25
factor = fold_enhancement(phi, alpha=3.0)
print(f"phi={phi:.2f}, estimated fold enhancement = {factor:.2f}")