#!/usr/bin/env python3
"""
Compute proton electrochemical potential and energy per proton.
Returns: pmf (V), delta_mu (J/mol), energy_per_proton_kJmol (kJ/mol)
"""
import numpy as np

F = 96485.33212       # C/mol, Faraday constant
R = 8.314462618       # J/(mol K), gas constant

def proton_pmf(pH_in, pH_out, delta_psi=0.0, T=298.15):
    # pH_in: pH of alkaline fluid (inside pore)
    # pH_out: pH of bulk ocean (outside)
    # delta_psi: psi_in - psi_out in volts (V)
    delta_pH = pH_in - pH_out
    pmf = delta_psi - (2.303 * R * T / F) * delta_pH  # volts
    # electrochemical molar energy change (J/mol)
    delta_mu = F * pmf
    return pmf, delta_mu, delta_mu / 1000.0  # V, J/mol, kJ/mol

# Example: pH_in=9.5, pH_out=6.5, no membrane potential
if __name__ == "__main__":
    pmf, delta_mu_j, delta_mu_kj = proton_pmf(9.5, 6.5, delta_psi=0.0)
    print(f"PMF: {pmf:.3f} V, $\Delta$μ per mol H+: {delta_mu_j:.1f} J/mol ({delta_mu_kj:.2f} kJ/mol)")
    # Expect ~0.177 V and ~17.1 kJ/mol for $\Delta$pH=3 at 298 K.