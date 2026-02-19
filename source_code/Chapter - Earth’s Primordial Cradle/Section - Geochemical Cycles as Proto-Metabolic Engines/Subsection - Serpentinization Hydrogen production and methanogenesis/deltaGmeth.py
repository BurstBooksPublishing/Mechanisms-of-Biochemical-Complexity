# compute_deltaG_meth.py -- production-ready function to evaluate Gibbs energy
import math

R = 8.31446261815324  # J/(mol·K)

def deltaG_meth(deltaG0_Jmol, T_K, a_CH4=1.0, a_H2O=1.0, a_CO2=1.0, a_H2=1.0):
    """
    Compute $\Delta$G for CO2 + 4 H2 -> CH4 + 2 H2O.
    deltaG0_Jmol: standard Gibbs energy (J/mol) at T reference (user-provided).
    Activities/fugacities a_* are unitless; supply fugacity for gases (bar) if preferred.
    """
    Q = (a_CH4 * (a_H2O**2)) / (a_CO2 * (a_H2**4))
    return deltaG0_Jmol + R * T_K * math.log(Q)

# Example usage: deltaG0 ~ -131e3 J/mol (biochemical standard, pH 7, 298 K)
if __name__ == "__main__":
    deltaG0 = -131000.0
    T = 323.15  # 50 °C in K
    # typical vent-like fugacities/activities (example values)
    dG = deltaG_meth(deltaG0, T, a_CH4=1e-3, a_H2O=1.0, a_CO2=1e-4, a_H2=1e-2)
    print(f"$\Delta$G = {dG/1000.0:.2f} kJ/mol")