import math

R = 8.314462618e-3  # kJ mol^-1 K^-1
def delta_delta_G(k_cat, k_uncat, T=298.15):
    """Return $\Delta$$\Delta$G‡ in kJ/mol from rate constants k_cat and k_uncat."""
    if k_cat <= 0 or k_uncat <= 0:
        raise ValueError("Rate constants must be positive.")
    ratio = k_cat / k_uncat
    return -R * T * math.log(ratio)

# Example usage (replace with experimental values):
# ddG = delta_delta_G(k_cat=1e1, k_uncat=1e-7, T=298.15)
# print(f"$\Delta$$\Delta$G‡ = {ddG:.2f} kJ/mol")