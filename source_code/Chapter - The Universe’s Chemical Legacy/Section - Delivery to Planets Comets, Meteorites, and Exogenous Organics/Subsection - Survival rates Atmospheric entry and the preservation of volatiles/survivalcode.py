import numpy as np

# Physical parameters (user-tunable)
alpha = 1e-7          # thermal diffusivity, m^2/s
tau = 10.0            # effective heating duration, s
lambda_depth = 1e3    # decay constant for exponential depth PDF, 1/m (scale = 1 mm)
n_mc = 200000         # Monte Carlo samples

# Derived thermal penetration depth
delta = np.sqrt(alpha * tau)  # m

# Analytic survival for exponential depth PDF: S = exp(-lambda * delta)
S_analytic = np.exp(-lambda_depth * delta)

# Monte Carlo sampling for verification (exponential and uniform models)
depths_exp = np.random.exponential(scale=1.0/lambda_depth, size=n_mc)
depths_unif = np.random.uniform(0.0, 0.01, size=n_mc)  # uniform to 1 cm

S_exp_mc = np.mean(depths_exp > delta)
S_unif_mc = np.mean(depths_unif > delta)

# Print concise results
print(f"delta = {delta:.3e} m")
print(f"Analytic (exp) S = {S_analytic:.4f}, MC (exp) S = {S_exp_mc:.4f}")
print(f"MC (uniform 0-1cm) S = {S_unif_mc:.4f}")