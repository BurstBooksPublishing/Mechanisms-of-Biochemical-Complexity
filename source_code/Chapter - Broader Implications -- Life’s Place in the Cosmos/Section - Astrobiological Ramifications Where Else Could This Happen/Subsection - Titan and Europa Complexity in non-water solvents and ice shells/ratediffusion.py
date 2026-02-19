import math
R = 8.31446261815324           # J mol^-1 K^-1
Ea = 50e3                      # activation energy J mol^-1
A = 1e12                       # prefactor s^-1 (typical molecular)
T_earth = 298.15               # K
T_titan = 94.0                 # K

def arrhenius_rate(A, Ea, T):
    return A * math.exp(-Ea/(R*T))

k_earth = arrhenius_rate(A, Ea, T_earth)
k_titan = arrhenius_rate(A, Ea, T_titan)
rate_ratio = k_titan / k_earth

# diffusion coefficients (m^2/s) estimates
D_earth = 1e-9                 # typical liquid at room T
D_titan = 1e-10                # reduced mobility in cryoliquid

# diffusion length for t = 1e6 s (~12 days)
t = 1e6
L_earth = math.sqrt(D_earth * t)
L_titan = math.sqrt(D_titan * t)

print(f"k_earth = {k_earth:.3e} s^-1, k_titan = {k_titan:.3e} s^-1")
print(f"rate ratio (Titan/Earth) = {rate_ratio:.3e}")
print(f"diffusion lengths: L_earth = {L_earth:.2e} m, L_titan = {L_titan:.2e} m")