import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters (chosen to show pattern formation)
A = 1.0          # feed concentration
B = 3.2          # control parameter
Du, Dv = 1e-2, 5e-3  # diffusion coefficients
L = 100.0        # physical length
N = 400          # spatial grid points
dx = L/N
x = np.linspace(0, L, N, endpoint=False)

# Initial condition: homogeneous plus small noise
u0 = A + 0.01*np.random.randn(N)
v0 = B/A + 0.01*np.random.randn(N)

def laplacian(arr):
    # Periodic second-difference
    return (np.roll(arr, -1) - 2*arr + np.roll(arr, 1)) / dx**2

def brusselator_rhs(t, y):
    u = y[:N]
    v = y[N:]
    # Local reaction rates
    ru = A - (B+1)*u + u*u*v
    rv = B*u - u*u*v
    # Reaction-diffusion PDE discretized to ODEs
    du_dt = ru + Du * laplacian(u)
    dv_dt = rv + Dv * laplacian(v)
    return np.concatenate([du_dt, dv_dt])

# Integrate in time
y0 = np.concatenate([u0, v0])
t_span = (0.0, 200.0)
sol = solve_ivp(brusselator_rhs, t_span, y0, method='RK45', atol=1e-6, rtol=1e-6)

# Extract final state
u_final = sol.y[:N, -1]
v_final = sol.y[N:, -1]

# Compute a simple proxy for entropy production:
# use local reaction flux J = u^2 v - B u (net autocatalytic flux proxy)
J_local = u_final**2 * v_final - B * u_final
# Use an idealized affinity proportional to log ratio (placeholder)
A_local = np.log(np.maximum(u_final, 1e-12) / (A))
sigma_local = J_local * A_local  # local production density (arbitrary units)
Sigma = np.trapz(sigma_local, x) # integrated entropy production proxy

# Plot results
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(x, u_final, label='u')
plt.plot(x, v_final, label='v')
plt.xlabel('x'); plt.title('Final concentrations')
plt.legend()
plt.subplot(1,2,2)
plt.plot(x, sigma_local); plt.xlabel('x'); plt.title('Local entropy prod. proxy')
plt.suptitle(f'Integrated sigma (proxy) = {Sigma:.3f}')
plt.tight_layout(); plt.show()