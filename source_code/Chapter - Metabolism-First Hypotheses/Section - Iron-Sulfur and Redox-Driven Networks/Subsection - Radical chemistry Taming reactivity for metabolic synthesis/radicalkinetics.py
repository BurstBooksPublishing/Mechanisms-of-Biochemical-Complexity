import numpy as np
from scipy.integrate import solve_ivp

# Parameters (typical orders of magnitude)
k_i = 1e-6      # M s^-1, radical initiation rate (low to avoid damage)
k_p = 1e6       # M^-1 s^-1, propagation (HAT) on surface
k_t = 1e8       # M^-1 s^-1, termination (radical-radical recombination)
k_pcet = 1e3    # M^-1 s^-1, PCET quenching by buffered donor
M0 = 1e-3       # M, substrate concentration
D0 = 1e-4       # M, PCET donor concentration

def odes(t, y):
    R, M, D = y
    # Radical balance: initiation, propagation consumes M producing product P (implicit),
    # termination removes two radicals, PCET consumes radicals and donor.
    dR_dt = k_i - k_t*R*R - k_pcet*R*D - k_p*R*M
    dM_dt = -k_p*R*M                         # substrate consumption by propagation
    dD_dt = -k_pcet*R*D                      # PCET donor consumption
    return [dR_dt, dM_dt, dD_dt]

y0 = [0.0, M0, D0]
t_span = (0, 1000)       # seconds
sol = solve_ivp(odes, t_span, y0, dense_output=True, max_step=1.0)

# Compute steady-state radical concentration estimate from Eq. (1)
R_ss_est = np.sqrt(k_i / k_t)
print(f"Estimated steady-state [R•] = {R_ss_est:.3e} M")
# Export time course (production code would store or plot results)
time = np.linspace(0, 1000, 201)
R_t, M_t, D_t = sol.sol(time)