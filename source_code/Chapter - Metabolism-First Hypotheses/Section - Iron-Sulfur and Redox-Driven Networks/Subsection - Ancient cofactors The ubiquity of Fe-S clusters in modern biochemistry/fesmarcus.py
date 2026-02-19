"""
Compute Marcus electron-transfer rates for Fe-S cofactors.
Returns k_ET in s^-1 for given parameters.
"""

import numpy as np

# Physical constants (SI)
kB = 1.380649e-23          # Boltzmann constant, J/K
hbar = 1.054571817e-34     # reduced Planck, J*s
F = 96485.33212            # Faraday constant, C/mol
e_charge = 1.602176634e-19 # elementary charge, C

def marcus_rate(V, lam, deltaG, T=298.15):
    """
    Marcus nonadiabatic ET rate.
    V: electronic coupling (J)
    lam: reorganization energy (J)
    deltaG: standard free energy change (J)
    T: temperature (K)
    """
    prefactor = (2*np.pi/hbar) * V**2
    denom = np.sqrt(4*np.pi*lam*kB*T)
    expo = -((deltaG + lam)**2) / (4*lam*kB*T)
    return prefactor/denom * np.exp(expo)

def coupling_from_distance(V0, beta, r):
    """
    Exponential distance decay of coupling.
    V0: coupling at r=0 (J)
    beta: decay constant (1/m)
    r: distance (m)
    """
    return V0 * np.exp(-beta * r)

# Example parameters (typical scales)
T = 323.15                    # hydrothermal temperature, K
lam = 0.7 * e_charge          # reorg energy ~0.7 eV
deltaE_mV = -400              # midpoint potential difference, mV
n = 1
deltaG = -n * (deltaE_mV/1000.0) * F  # convert mV to J

V0 = 1e-19                    # reference coupling, J
beta = 1.2e10                 # 1/Å -> 1/m (1.2 Å^-1)
r = 1.2e-9                    # 12 Å in meters

V = coupling_from_distance(V0, beta, r)
k_et = marcus_rate(V, lam, deltaG, T)

# Print results
print(f"V = {V:.3e} J, k_ET = {k_et:.3e} s^-1")