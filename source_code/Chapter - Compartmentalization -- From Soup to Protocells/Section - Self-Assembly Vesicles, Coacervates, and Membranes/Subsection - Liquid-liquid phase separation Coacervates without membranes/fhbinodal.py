import numpy as np
from scipy.optimize import brentq, minimize

kBT = 1.0  # natural units
def f_phi(phi, N, chi):
    # Flory-Huggins free energy per site (dimensionless)
    phi = np.clip(phi, 1e-12, 1-1e-12)
    return (phi/N)*np.log(phi) + (1-phi)*np.log(1-phi) + chi*phi*(1-phi)

def mu_phi(phi, N, chi):
    # chemical potential d(f)/d(phi)
    phi = np.clip(phi, 1e-12, 1-1e-12)
    return (1/N)*(np.log(phi)+1) - (np.log(1-phi)+1) + chi*(1-2*phi)

def common_tangent(N, chi):
    # find two-phase coexistence phi1 < phi2
    def objective(vars):
        phi1, phi2 = vars
        f1, f2 = f_phi(phi1, N, chi), f_phi(phi2, N, chi)
        mu1, mu2 = mu_phi(phi1, N, chi), mu_phi(phi2, N, chi)
        # conditions: equal chemical potentials and equal tangent intercepts
        return mu1-mu2, (f1 - mu1*phi1) - (f2 - mu2*phi2)
    # initial guesses
    x0 = np.array([1e-3, 1-1e-3])
    res = minimize(lambda v: np.sum(np.square(objective(v))), x0,
                   bounds=[(1e-6,0.2),(0.8,1-1e-6)])
    if not res.success:
        raise RuntimeError("Binodal solve failed")
    phi1, phi2 = res.x
    return phi1, phi2

# Example usage
N = 100
chi = 0.5
phi1, phi2 = common_tangent(N, chi)
print(f"Coexistence: phi_low={phi1:.4g}, phi_high={phi2:.4g}")