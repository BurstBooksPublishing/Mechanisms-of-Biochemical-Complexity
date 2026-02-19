import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve

# Physical parameters
L = 1.0                    # domain length (m)
N = 200                    # grid points
x = np.linspace(0, L, N)
dx = x[1]-x[0]
D_d, D_a = 1e-9, 1e-9      # diffusion coefficients (m^2/s)
T = 298.15                 # temperature (K)
R = 8.314                  # J/(mol K)

# Boundary concentrations (mol/m^3): donor at left, acceptor at right
c_d_left, c_d_right = 1.0, 1e-6
c_a_left, c_a_right = 1e-6, 1.0

# Assemble Laplacian for steady diffusion (Dirichlet BCs applied)
diag = np.ones(N)*(-2.0)
off = np.ones(N-1)
Lmat = diags([off, diag, off], [-1,0,1]).toarray() / dx**2
Lmat[0,:] = Lmat[-1,:] = 0.0  # enforce BC rows

# Solve steady profiles: D * L * c = 0 with Dirichlet BCs
b_d = np.zeros(N); b_a = np.zeros(N)
b_d[0], b_d[-1] = c_d_left, c_d_right
b_a[0], b_a[-1] = c_a_left, c_a_right
A = np.eye(N); A[1:-1,1:-1] = Lmat[1:-1,1:-1]
c_d = solve(A, b_d); c_a = solve(A, b_a)

# Chemical potentials using ideal activities (can replace with speciation)
mu0_d, mu0_a = 0.0, 0.0
mu_d = mu0_d + R*T*np.log(np.maximum(c_d,1e-12))
mu_a = mu0_a + R*T*np.log(np.maximum(c_a,1e-12))

# Fluxes (Fick) and entropy production density sigma = sum(J_i * -grad(mu_i))/T
J_d = -D_d * np.gradient(c_d, x)
J_a = -D_a * np.gradient(c_a, x)
grad_mu_d = np.gradient(mu_d, x)
grad_mu_a = np.gradient(mu_a, x)
sigma = (J_d * (-grad_mu_d) + J_a * (-grad_mu_a)) / T

# Output integrated available power per unit area (W/m^2)
power_density = np.trapz(sigma, x)
print(f"Integrated chemical power density: {power_density:.3e} W/m^2")