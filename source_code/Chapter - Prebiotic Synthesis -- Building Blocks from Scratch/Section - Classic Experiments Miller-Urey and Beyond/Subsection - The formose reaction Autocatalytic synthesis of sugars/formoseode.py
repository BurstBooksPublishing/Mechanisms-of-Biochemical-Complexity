import numpy as np
from scipy.integrate import solve_ivp

def formose_odes(t, y, params):
    F, G, S = y
    k0, kcat, ks, kc, kd = params['k0'], params['kcat'], params['ks'], params['kc'], params['kd']
    # reaction rates (mass-action)
    r_uncat = k0 * F**2           # 2F -> G
    r_auto  = kcat * G * F        # G + F -> 2G
    r_synth = ks * G * F          # G + F -> S
    r_canna = kc * F**2           # Cannizzaro loss
    dF = -2*r_uncat - r_auto - r_synth - 2*r_canna
    dG = r_uncat + r_auto - r_synth - kd * G
    dS = r_synth
    return [dF, dG, dS]

def simulate_formose(t_span=(0,1000), y0=(0.1,1e-6,0.0), params=None, t_eval=None):
    """Simulate minimal formose network. Returns solution object."""
    if params is None:
        params = {'k0':1e-2, 'kcat':1.0, 'ks':1e-1, 'kc':1e-3, 'kd':1e-2}
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(formose_odes, t_span, y0, args=(params,), t_eval=t_eval, rtol=1e-8, atol=1e-12)
    return sol

# example usage
# sol = simulate_formose()
# np.savez('formose_sim.npz', t=sol.t, y=sol.y)  # save for plotting/analysis