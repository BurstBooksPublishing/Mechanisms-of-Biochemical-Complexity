import numpy as np

def symmetric_shock_state(rho0, c0, s, V_imp, Cv, T0=300.0):
    """
    rho0 : initial density (kg/m^3)
    c0   : bulk sound speed (m/s)
    s    : Hugoniot slope (dimensionless)
    V_imp: impact velocity (m/s)
    Cv   : specific heat at constant volume (J/kg/K)
    T0   : initial temperature (K)
    returns (P_peak_Pa, T_peak_K)
    """
    Up = V_imp / 2.0                         # particle velocity for symmetric impact
    Us = c0 + s * Up                         # shock velocity from linear Hugoniot
    P = rho0 * Us * Up                       # Rankine-Hugoniot momentum relation (Pa)
    # approximate specific internal energy change using Hugoniot energy relation
    # assume initial pressure P0 ~ 0 and approximate volumetric change via linear compressibility:
    # use DeltaE = 0.5 * P * (1/rho0 - 1/(rho0*(1+P/(rho0*c0**2)))) as a simple proxy
    V1 = 1.0 / rho0
    rho2 = rho0 * (1.0 + P/(rho0 * c0**2))   # crude stiffened-gas estimate
    V2 = 1.0 / rho2
    deltaE = 0.5 * P * (V1 - V2)
    deltaT = deltaE / Cv
    T_peak = T0 + deltaT
    return P, T_peak

# Example usage: for porous ice-rock analog (values illustrative)
rho0 = 2000.0       # kg/m^3
c0 = 3000.0         # m/s
s = 1.2
V_imp = 10000.0     # m/s (10 km/s)
Cv = 1000.0         # J/kg/K
P_peak, T_peak = symmetric_shock_state(rho0, c0, s, V_imp, Cv)
print(f"Peak pressure: {P_peak/1e9:.2f} GPa, Peak temperature: {T_peak:.0f} K")