import numpy as np

R = 1.9872036e-3  # kcal mol^-1 K^-1

def tm_from_dh_ds(dh, ds, Ct):
    # Return melting temperature in Kelvin for non-self-complementary duplex.
    return dh / (ds + R * np.log(Ct / 4.0))

def deltaG(dh, ds, T):
    # Gibbs free energy at temperature T in Kelvin.
    return dh - T * ds

def fidelity_probs(dG_list, T):
    # Compute Boltzmann probabilities from a list of standard deltaG (kcal/mol).
    exps = np.exp(-np.array(dG_list) / (R * T))
    return exps / exps.sum()

# Example: correct and one mismatch (values in kcal/mol).
dh_correct, ds_correct = -70.0, -0.20  # example aggregate values
dh_mismatch, ds_mismatch = -66.0, -0.195

T = 298.15
Ct = 1e-6  # 1 μM total strand concentration

Tm_corr = tm_from_dh_ds(dh_correct, ds_correct, Ct)
dG_corr = deltaG(dh_correct, ds_correct, T)
dG_mis  = deltaG(dh_mismatch, ds_mismatch, T)

probs = fidelity_probs([dG_corr, dG_mis], T)
print(f"Tm_correct={Tm_corr:.1f} K, P_correct={probs[0]:.3f}")  # display results