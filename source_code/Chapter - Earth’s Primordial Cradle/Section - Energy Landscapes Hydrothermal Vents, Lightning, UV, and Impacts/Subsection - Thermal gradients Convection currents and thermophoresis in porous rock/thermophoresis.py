import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def run_simulation(nx=200, L=0.02, dt=0.1, tmax=100.0,
                   D=1e-9, S_T=1e-3, k=1e-13, mu=1e-3,
                   rho=1000.0, beta=4e-4, g=9.81, dT=100.0):
    # grid and initial condition
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    c = np.zeros(nx); c[0] = 1.0  # feed at left boundary

    # temperature field (linear gradient)
    T = 300.0 + dT * (1 - x / L)  # hot at x=0, cold at x=L
    dTdx = np.gradient(T, dx)
    D_T = D * S_T

    # Darcy velocity from buoyancy scaling (assumed uniform)
    v = (k / mu) * rho * beta * g * dT / L

    # implicit diffusion matrix (Dirichlet at boundaries)
    alpha = D * dt / dx**2
    main = (1 + 2*alpha) * np.ones(nx)
    off = -alpha * np.ones(nx-1)
    A = diags([off, main, off], [-1, 0, 1], format='csc')
    # enforce Dirichlet: left and right fixed
    A[0,:] = 0; A[0,0] = 1
    A[-1,:] = 0; A[-1,-1] = 1

    times = np.arange(0, tmax, dt)
    for _ in times:
        # explicit advection + thermophoresis (upwind for advection)
        # advective flux J_adv = v*c
        J_adv = np.empty_like(c)
        J_adv[1:] = v * c[:-1]  # upwind approximation
        J_adv[0] = v * c[0]
        adv_term = -(J_adv[1:] - J_adv[:-1]) / dx
        # thermophoretic flux J_th = -D_T * c * dTdx
        J_th = -D_T * c * dTdx
        th_term = np.empty_like(c)
        th_term[1:-1] = -(J_th[2:] - J_th[1:-1]) / dx
        th_term[0] = 0; th_term[-1] = 0

        rhs = c + dt * (np.concatenate(([0.0], adv_term)) + dt * 0*0) + dt * th_term
        # enforce Dirichlet boundaries (left fixed source, right zero)
        rhs[0] = 1.0
        rhs[-1] = 0.0
        c = spsolve(A, rhs)

    return x, T, c

# Example run
if __name__ == "__main__":
    x, T, c = run_simulation()
    # plotting omitted; callers can inspect arrays for concentration profiles