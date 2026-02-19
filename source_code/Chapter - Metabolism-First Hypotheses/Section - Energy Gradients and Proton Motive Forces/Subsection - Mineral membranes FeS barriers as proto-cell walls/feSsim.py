import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Physical parameters
L = 1e-4                     # membrane thickness (m)
N = 200                      # grid points
D_H = 9.3e-9                 # proton diffusivity (m2/s)
k_cat = 1e-3                 # surface catalytic rate (s^-1)
pH_ext = 6.0                 # exterior seawater pH
pH_int = 10.0                # alkaline fluid pH
cH_ext = 10**(-pH_ext)       # mol/L -> mol/m3 later conversion
cH_int = 10**(-pH_int)

# Grid setup
x = np.linspace(0, L, N)
dx = x[1] - x[0]
# Concentration in mol/m3 (convert from mol/L)
cH_left = cH_int * 1e3
cH_right = cH_ext * 1e3

# Build finite-difference Laplacian for steady diffusion
main = -2.0 * np.ones(N)
off = 1.0 * np.ones(N-1)
lap = diags([off, main, off], [-1, 0, 1]) / dx**2

# Boundary conditions and source term (surface catalysis at center)
b = np.zeros(N)
b[0] = cH_left
b[-1] = cH_right
# Impose Dirichlet by modifying matrix rows
A = -D_H * lap.tocsr()
A[0, :] = 0; A[0, 0] = 1
A[-1, :] = 0; A[-1, -1] = 1

# Optional: include first-order sink term uniformly to model proton production/consumption
sink = k_cat * np.ones(N)    # s^-1
A += diags(sink, 0)

# Solve for steady-state concentration (mol/m3)
cH = spsolve(A, b)

# Compute proton flux at membrane interior (Fick's law)
J = -D_H * (cH[1] - cH[0]) / dx   # mol/m2/s
print(f"Proton flux J = {J:.3e} mol m^-2 s^-1")