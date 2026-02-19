import numpy as np

def hydrophobic_aggregation(N: int, r_m: float, gamma: float,
                            dgamma_dT: float, T: float = 298.15):
    """
    Compute area change, free energy, and entropy change for aggregation.
    N        : number of spherical hydrophobes
    r_m      : radius of each sphere (m)
    gamma    : interfacial free energy (J m^-2)
    dgamma_dT: temperature derivative of gamma (J m^-2 K^-1)
    T        : temperature (K)
    Returns (DeltaA, DeltaG, DeltaS, DeltaG_kBT)
    """
    single_area = 4.0 * np.pi * r_m**2
    agg_area = 4.0 * np.pi * r_m**2 * N**(2.0/3.0)
    deltaA = N * single_area - agg_area
    deltaG = gamma * deltaA
    deltaS = -deltaA * dgamma_dT
    kBT = 1.380649e-23 * T
    return deltaA, deltaG, deltaS, deltaG / kBT

# Example invocation (parameters in SI units)
if __name__ == "__main__":
    N = 50
    r = 0.25e-9           # 0.25 nm
    gamma = 0.05          # J m^-2
    dgamma_dT = -1.5e-4   # J m^-2 K^-1
    A, G, S, G_kBT = hydrophobic_aggregation(N, r, gamma, dgamma_dT)
    print(f"DeltaA = {A:.3e} m^2, DeltaG = {G:.3e} J, TDeltaS = {T*S:.3e} J, DeltaG = {G_kBT:.1f} kBT")