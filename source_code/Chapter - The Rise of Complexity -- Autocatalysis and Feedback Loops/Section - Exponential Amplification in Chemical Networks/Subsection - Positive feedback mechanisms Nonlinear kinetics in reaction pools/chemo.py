import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def chemostat(t, y, params):
    A, B = y
    k, kd, D, A_in = params['k'], params['kd'], params['D'], params['A_in']
    dA = D*(A_in - A) - k*A*B
    dB = k*A*B - (D + kd)*B
    return [dA, dB]

# parameters (example realistic magnitudes)
params = {'k': 1.0, 'kd': 0.1, 'D': 0.05, 'A_in': 1.0}
y0 = [params['A_in'], 1e-6]  # small seeding of B
tspan = (0.0, 200.0)
t_eval = np.linspace(*tspan, 2001)

sol = solve_ivp(chemostat, tspan, y0, t_eval=t_eval, args=(params,), rtol=1e-8, atol=1e-10)

plt.plot(sol.t, sol.y[0], label='A (feedstock)')
plt.plot(sol.t, sol.y[1], label='B (autocatalyst)')
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('concentration (log scale)')
plt.legend()
plt.show()