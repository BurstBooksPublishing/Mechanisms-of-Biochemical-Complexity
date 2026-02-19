#!/usr/bin/env python3
# Compute proton-motive force and $\Delta$G for CO2 + H2 -> HCOO^- + H+
import math

# constants
R = 8.31446261815324        # J mol^-1 K^-1
F = 96485.33212331002       # C mol^-1
T = 298.15                  # K

def pmf_mV(delta_pH, delta_psi_mV=0.0, T=T):
    # returns pmf in mV and energy per mol H+ in kJ/mol
    factor = 2.303*R*T/F     # V per pH unit
    pmf_V = (delta_psi_mV/1000.0) - factor*delta_pH
    energy_kJ_per_mol = F * pmf_V / 1000.0
    return pmf_V*1000.0, energy_kJ_per_mol

def deltaG_from_potentials(E_acceptor_V, E_donor_V, n=2):
    # $\Delta$G = -n F (E_acceptor - E_donor)
    deltaE = E_acceptor_V - E_donor_V
    return -n * F * deltaE / 1000.0  # kJ/mol

# example: delta pH = 4
pmf_val_mV, energy_per_H = pmf_mV(4.0)
print(f"pmf = {pmf_val_mV:.1f} mV, energy per H+ = {energy_per_H:.1f} kJ/mol")

# redox example: approximate standard potentials (V vs SHE)
E_CO2_to_formate = -0.43   # V, CO2 + 2H+ + 2e- -> HCOO- + H2O (approx)
E_Hplus_to_H2 = 0.00       # V at standard conditions
dg_redox = deltaG_from_potentials(E_CO2_to_formate, E_Hplus_to_H2, n=2)
print(f"$\Delta$G°' (CO2 + H2 -> formate) ≈ {dg_redox:.1f} kJ/mol (standard approx)")