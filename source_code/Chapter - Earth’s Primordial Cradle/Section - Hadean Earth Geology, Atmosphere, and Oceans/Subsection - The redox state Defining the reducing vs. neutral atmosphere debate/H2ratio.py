import numpy as np
import matplotlib.pyplot as plt

def h2_h2o_ratio(fO2_array, Kp):
    # Eq. (2): f_H2 / f_H2O = (1 / Kp) * fO2^{-1/2}
    return (1.0 / Kp) * fO2_array**(-0.5)

# Example usage: supply Kp(T) from thermochemical source
# Here Kp is a placeholder; replace with value computed from $\Delta$G°(T).
Kp_example = 1.0e-1  # user-supplied equilibrium constant (fugacity units)
logfO2 = np.linspace(-40, -10, 301)  # typical planetary range (log10)
fO2 = 10.0**logfO2

ratio = h2_h2o_ratio(fO2, Kp_example)

plt.semilogx(fO2, ratio)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$f_{\mathrm{O}_2}$ (dimensionless fugacity)')
plt.ylabel(r'$f_{{\rm H_2}}/f_{{\rm H_2O}}$')
plt.title('Equilibrium H2/H2O vs oxygen fugacity (Kp={})'.format(Kp_example))
plt.grid(True, which='both', ls=':')
plt.show()