# Production-ready Python: requires numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

def master_fraction(mu, L, A, a):
    """Return steady-state fraction of master sequence for single-peak landscape.
       Uses approximation x0 = max(0, (q^L * A - a) / (A - a))."""
    q = 1.0 - mu
    qL = q**L
    num = qL * A - a
    denom = A - a
    return np.clip(num / denom, 0.0, 1.0)

# Parameters: adjust to experimental scenarios
L = 100                  # genome length (nucleotides)
A = 10.0                 # master fitness
a = 1.0                  # mutant fitness
mus = np.logspace(-3, -0.0, 200)  # per-site error rates 1e-3 .. 1.0

x0 = master_fraction(mus, L, A, a)

plt.semilogx(mus, x0, lw=2)
plt.xlabel("Per-site error rate $\mu$")
plt.ylabel("Master sequence fraction $x_0$")
plt.title(f"Error threshold: L={L}, A/a={A/a:.1f}")
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()