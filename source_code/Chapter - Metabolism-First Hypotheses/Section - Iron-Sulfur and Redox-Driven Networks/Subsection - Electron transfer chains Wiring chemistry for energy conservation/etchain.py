import numpy as np
from scipy.linalg import null_space

# Physical constants
hbar = 1.054571817e-34  # J s
kB = 1.380649e-23       # J K^-1
F = 96485.33212         # C mol^-1
e_charge = 1.602176634e-19  # C

def marcus_rate(V, lam, deltaG, T):
    """Nonadiabatic Marcus rate (s^-1). V in J, lam in J, deltaG in J, T in K."""
    prefactor = (2*np.pi/hbar) * V**2 / np.sqrt(4*np.pi*lam*kB*T)
    exponent = -((deltaG + lam)**2) / (4*lam*kB*T)
    return prefactor * np.exp(exponent)

def build_rate_matrix(E_mV, V_coup_cm1, lam_eV, T=298.15):
    """Construct rate matrix K for linear chain from potentials E_mV list."""
    n = len(E_mV)
    # unit conversions
    E = np.array(E_mV) * 1e-3  # V
    lam = lam_eV * 1.602176634e-19 * 1e0  # J
    V_coup = (V_coup_cm1 * 100.0) * (hbar*1e10)  # rough conversion to J (approx)
    K = np.zeros((n, n))
    for i in range(n-1):
        deltaE = E[i+1] - E[i]                # V
        deltaG = -1.0 * F * deltaE           # J per mol electron -> J
        # single-electron transfer: use V_coup as coupling energy scale
        k_forward = marcus_rate(V_coup, lam, deltaG, T)
        k_backward = marcus_rate(V_coup, lam, -deltaG, T)
        K[i, i] -= k_forward
        K[i+1, i] += k_forward
        K[i+1, i+1] -= k_backward
        K[i, i+1] += k_backward
    return K

def steady_state_flux(E_mV, V_coup_cm1=5.0, lam_eV=0.7, T=298.15):
    """Compute steady-state populations and flux through first bond."""
    K = build_rate_matrix(E_mV, V_coup_cm1, lam_eV, T)
    # null_space finds steady-state vector (kernel of K)
    ns = null_space(K)
    p_ss = ns[:, 0]
    p_ss = np.real(p_ss / np.sum(p_ss))  # normalized steady-state probabilities
    # compute flux across first bond
    # forward and backward rates for bond 0
    n = len(E_mV)
    # recompute rates for bond 0
    E = np.array(E_mV) * 1e-3
    deltaG = -F * (E[1] - E[0])
    lam = lam_eV * 1.602176634e-19
    V_coup = (V_coup_cm1 * 100.0) * (hbar*1e10)
    kf = marcus_rate(V_coup, lam, deltaG, T)
    kb = marcus_rate(V_coup, lam, -deltaG, T)
    J = kf * p_ss[0] - kb * p_ss[1]
    I = J * e_charge  # current in A per chain
    return p_ss, J, I

# Example potentials in mV across a 6-site Fe-S wire stepping toward a more positive sink
E_mV = [-420, -360, -300, -240, -180, -120]
p_ss, J, I = steady_state_flux(E_mV)
# p_ss: steady-state site occupancies, J: electrons s^-1, I: amperes per chain
print("Steady-state occupancies:", p_ss)
print("Flux (e s^-1):", J, "Current (A):", I)