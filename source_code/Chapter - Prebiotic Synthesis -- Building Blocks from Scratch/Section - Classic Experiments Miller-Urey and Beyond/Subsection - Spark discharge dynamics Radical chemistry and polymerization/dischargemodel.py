import numpy as np
from scipy.integrate import solve_ivp

# Parameters (SI units)
G = 1e20         # radicals generated per m^3 s^-1 (adjust for discharge power)
kp = 1e-16       # m^3 s^-1 propagation rate constant
kt = 1e-12       # m^3 s^-1 termination rate constant
M0 = 1e24        # monomer number density (m^-3)

def odes(t, y):
    R, M, P = y                          # radicals, monomer, polymer mass proxy
    dR_dt = G - 2*kt*R*R - kp*R*M       # generation - termination - propagation
    dM_dt = -kp*R*M                      # monomer consumption by propagation
    dP_dt =  kp*R*M                      # polymer mass accrual (proportional)
    return [dR_dt, dM_dt, dP_dt]

# initial conditions: no radicals, initial monomer, no polymer
y0 = [0.0, M0, 0.0]
t_span = (0.0, 10.0)  # seconds
sol = solve_ivp(odes, t_span, y0, rtol=1e-7, atol=1e-9)

# Output steady-state estimates for diagnostic use
R_ss = np.sqrt(G/(2*kt))
Rp_ss = kp*M0*R_ss
print(f"Estimated steady-state radicals: {R_ss:.3e} m^-3")
print(f"Estimated polymerization rate: {Rp_ss:.3e} m^-3 s^-1")