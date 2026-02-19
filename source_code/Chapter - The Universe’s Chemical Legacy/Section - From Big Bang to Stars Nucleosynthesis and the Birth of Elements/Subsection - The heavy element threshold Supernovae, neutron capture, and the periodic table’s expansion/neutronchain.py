import numpy as np
from scipy.integrate import solve_ivp

def integrate_chain(A_min, A_max, Yn0, n_n, sigma_v, decay_rates, t_span):
    """
    Integrate dY_A/dt = -Y_A*n_n*sigma_v[A] + Y_{A-1}*n_n*sigma_v[A-1] - decay_rates[A]*Y_A.
    A_min, A_max : int mass-number bounds inclusive
    Yn0 : array initial abundances length A_max-A_min+1
    n_n : neutron density (cm^-3)
    sigma_v : array of  for each A (cm^3 s^-1)
    decay_rates : array lambda for each A (s^-1)
    t_span : tuple (t0, tf) seconds
    Returns: times, Y(t) array
    """
    A_count = A_max - A_min + 1
    sigma_v = np.asarray(sigma_v, float)
    decay_rates = np.asarray(decay_rates, float)
    def dydt(t, Y):
        dY = np.zeros_like(Y)
        for i in range(A_count):
            cap_out = Y[i] * n_n * sigma_v[i]
            cap_in = Y[i-1] * n_n * sigma_v[i-1] if i>0 else 0.0
            dY[i] = -cap_out + cap_in - decay_rates[i] * Y[i]
        return dY
    sol = solve_ivp(dydt, t_span, Yn0, dense_output=True, atol=1e-12, rtol=1e-9)
    return sol.t, sol.y

# Example usage: s-process-like parameters (low n_n) vs r-process-like (high n_n).