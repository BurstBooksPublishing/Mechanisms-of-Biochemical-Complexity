import numpy as np
from scipy.constants import R, N_A, physical_constants

F = physical_constants['Faraday constant'][0]  # C/mol

def donnan_mono(c0, a, T=298.15):
    """
    Compute Donnan equilibrium for symmetric 1:1 electrolyte.
    Parameters
    ----------
    c0 : float
        External salt concentration (mol/L), c_{+}^{out} = c_{-}^{out} = c0.
    a : float
        Internal fixed negative charge concentration (mol/L).
    T : float
        Temperature in Kelvin.
    Returns
    -------
    c_in_plus, c_in_minus, delta_phi
        Internal cation and anion concentrations (mol/L), Donnan potential (V).
    """
    if c0 < 0 or a < 0:
        raise ValueError("c0 and a must be non-negative.")
    # analytic solution from quadratic system
    sqrt_term = np.sqrt((a/2.0)**2 + c0**2)
    c_in_plus = +a/2.0 + sqrt_term
    c_in_minus = -a/2.0 + sqrt_term
    # Donnan potential via Nernst for cation
    delta_phi = (R * T / F) * np.log(c_in_plus / c0) if c0 > 0 else np.nan
    return c_in_plus, c_in_minus, delta_phi

# Example usage:
# c_in_plus, c_in_minus, phi = donnan_mono(0.01, 0.1)