import numpy as np
from scipy.integrate import solve_ivp

# Parameters (example, units: s^-1 or M^-1 s^-1 where applicable)
k_form = 1e2        # pseudo-first-order formation from surface activation (s^-1)
k_hyd = 1e-3        # hydrolysis rate constant of thioester (s^-1)
k_transfer = 1e1    # acyl transfer to acceptor (s^-1)
A0 = 1e-3           # initial acetate/acceptor (M)
T0 = 1e-4           # initial free thiol (M)
AcS0 = 0.0          # initial acetyl-thioester (M)

def odes(t, y):
    Ac, T, AcS = y
    # Formation: activation flux converts acetate+thiol -> AcS (pseudo-first-order)
    r_form = k_form * Ac * T
    # Hydrolysis: AcS -> Ac + T
    r_hyd = k_hyd * AcS
    # Transfer: AcS + acceptor -> product + T (consumes AcS)
    r_tr = k_transfer * AcS * (A0 - Ac)  # acceptor pool modeled as A0 - Ac for simplicity
    dAc_dt = -r_form + r_hyd + r_tr      # acetate net change
    dT_dt = -r_form + r_hyd + r_tr       # thiol net change (regenerated on transfer/hydrolysis)
    dAcS_dt = r_form - r_hyd - r_tr      # thioester balance
    return [dAc_dt, dT_dt, dAcS_dt]

sol = solve_ivp(odes, [0, 1000], [A0, T0, AcS0], rtol=1e-8, atol=1e-12)
Ac_ss, T_ss, AcS_ss = sol.y[:, -1]
print(f"steady-state acetyl-thioester: {AcS_ss:.3e} M")
print(f"steady-state thiol: {T_ss:.3e} M")
print(f"steady-state acetate: {Ac_ss:.3e} M")