import numpy as np
from scipy.optimize import root

# Solve mass-balance + equilibrium for M + L <-> ML and Fe2+ / Fe3+ Nernst constraint
def speciation_solver(M_tot, L_tot, K_ML, E0=None, Eh=None, n=1, T=298.15):
    R, F = 8.314462618, 96485.33212
    # Unknowns: [M], [L], [ML], optionally [Fe2+],[Fe3+]
    def residuals(x):
        M, L, ML = x
        eq1 = ML - K_ML*M*L                         # binding equilibrium (Eq. 1)
        eq2 = M + ML - M_tot                       # metal mass balance
        eq3 = L + ML - L_tot                       # ligand mass balance
        return [eq1, eq2, eq3]

    # initial guess
    x0 = np.array([max(M_tot*0.5,1e-12), max(L_tot*0.5,1e-12), max(min(M_tot,L_tot)*0.5,1e-12)])
    sol = root(residuals, x0, tol=1e-12)
    if not sol.success:
        raise RuntimeError("Speciation solver failed: " + sol.message)
    M, L, ML = sol.x
    # if redox info provided, compute ratio via Nernst (optional)
    fe_ratio = None
    if E0 is not None and Eh is not None:
        fe_ratio = np.exp(-n*F*(Eh - E0)/(R*T))  # [Fe2+]/[Fe3+] from Nernst
    return {"M": M, "L": L, "ML": ML, "f_bound": ML/M_tot, "fe_ratio": fe_ratio}

# Example usage: 1e-3 M Mg total, 1e-3 M phosphate, K=100
if __name__ == "__main__":
    out = speciation_solver(1e-3, 1e-3, K_ML=1e2)
    print(out)  # production-ready numeric output for design decisions