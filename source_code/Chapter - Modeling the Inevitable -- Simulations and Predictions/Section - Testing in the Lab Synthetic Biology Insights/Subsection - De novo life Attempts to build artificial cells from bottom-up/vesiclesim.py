import numpy as np
from scipy.integrate import solve_ivp

# Parameters (tune to experimental values)
r = 0.5        # replicator max rate (1/min)
K = 100.0      # saturation constant
delta = 0.01   # degradation rate
k_upt = 1e-3   # amphiphile uptake rate (mM^-1 min^-1)
L_ext = 1.0    # external amphiphile concentration (mM)
beta = 1e-2    # coupling: replicator -> area (um^2 per molecule per min)
mu = 1e-2      # relaxation rate of area
A_eq = 10.0    # equilibrium area (um^2)
V0 = 4.18879   # initial volume of sphere (um^3) ~ (4/3)pi R^3 with R=1

s_threshold = 4.0  # surface-to-volume threshold for division (um^-1)

def odes(t, y):
    N, A = y
    dNdt = r * N / (1 + N / K) - delta * N
    dAdt = k_upt * L_ext * A + beta * N - mu * (A - A_eq)
    return [dNdt, dAdt]

def simulate(tmax=500.0, dt=0.1):
    t = 0.0
    state = np.array([10.0, A_eq])  # initial N, A
    times, Ns, As = [t], [state[0]], [state[1]]
    while t < tmax:
        sol = solve_ivp(odes, [t, t+dt], state, method='RK45', rtol=1e-6)
        state = sol.y[:, -1]
        # compute current volume from area assuming sphere: A = 4*pi*R^2 => R = sqrt(A/(4*pi))
        R = np.sqrt(state[1] / (4*np.pi))
        V = 4/3 * np.pi * R**3
        s = state[1] / V
        # division event
        if s > s_threshold and state[0] >= 2:
            # stochastic partitioning of replicators (binomial)
            n_left = np.random.binomial(int(round(state[0])), 0.5)
            n_right = int(round(state[0])) - n_left
            # split into two daughters; continue simulation with one daughter chosen at random
            state = np.array([n_left, state[1]/2.0]) if np.random.rand() < 0.5 else np.array([n_right, state[1]/2.0])
        t += dt
        times.append(t); Ns.append(state[0]); As.append(state[1])
    return np.array(times), np.array(Ns), np.array(As)

# Example run (plotting omitted for brevity)
if __name__ == "__main__":
    times, Ns, As = simulate()
    # save results for experimental comparison
    np.savez("vesicle_sim_results.npz", t=times, N=Ns, A=As)