import numpy as np
import matplotlib.pyplot as plt

def flory_distribution(p, Nmax=50):
    # number fraction of chains of length n: (1-p) * p^(n-1)
    n = np.arange(1, Nmax+1)
    frac = (1.0 - p) * (p ** (n - 1))
    return n, frac

# Example extents of reaction
ps = [0.2, 0.5, 0.8]
plt.figure(figsize=(6,4))
for p in ps:
    n, frac = flory_distribution(p, Nmax=40)
    plt.semilogy(n, frac, marker='o', label=f'p={p:.2f}')
plt.xlabel('Degree of polymerization (n)')
plt.ylabel('Number fraction (log scale)')
plt.title('Flory-type oligomer distribution for HCN polymerization')
plt.legend()
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.show()