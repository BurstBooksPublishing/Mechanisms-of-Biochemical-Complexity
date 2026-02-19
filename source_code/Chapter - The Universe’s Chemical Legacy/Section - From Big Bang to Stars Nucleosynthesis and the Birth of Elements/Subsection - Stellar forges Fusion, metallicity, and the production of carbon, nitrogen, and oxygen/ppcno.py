import numpy as np

# Physical scalings (illustrative): choose normalized constants
eps0_pp = 1.0         # arbitrary units at T6=1
eps0_cno = 8e-3       # scaled to reflect smaller prefactor
rho = 1.5             # g cm^-3, typical core density
X = 0.70              # hydrogen mass fraction

def eps_pp(T, rho=rho, X=X):
    T6 = T / 1e6
    return eps0_pp * rho * X**2 * T6**4

def eps_cno(T, Z, rho=rho, X=X):
    T6 = T / 1e6
    return eps0_cno * rho * X * Z * T6**18

def crossover_temperature(Z, Tmin=5e6, Tmax=2e8):
    Ts = np.logspace(np.log10(Tmin), np.log10(Tmax), 1000)
    ratio = np.array([eps_cno(T, Z)/eps_pp(T) for T in Ts])
    idx = np.argmax(ratio > 1.0)
    return Ts[idx] if ratio.max() > 1.0 else None

# Example: find crossover for solar metallicity and low metallicity
for Z in (0.014, 1e-4):
    Tcross = crossover_temperature(Z)
    print(f"Z={Z:.3g}, crossover T ~ {Tcross:.2e} K")