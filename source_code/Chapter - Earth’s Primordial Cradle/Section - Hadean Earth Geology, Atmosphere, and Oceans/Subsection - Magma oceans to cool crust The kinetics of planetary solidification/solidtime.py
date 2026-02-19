import numpy as np

def magma_ocean_solidification_time(H,
                                   deltaT=2000.0,
                                   rho=3300.0,
                                   c=1200.0,
                                   L=4.0e5,
                                   k=3.0,
                                   alpha=3e-5,
                                   g=9.81,
                                   eta=10.0,
                                   a=0.08,
                                   beta=1/3):
    """
    Return solidification time in years.
    H      : layer depth (m)
    deltaT : driving superadiabatic temperature (K)
    rho    : density (kg/m^3)
    c      : heat capacity (J/kg/K)
    L      : latent heat (J/kg)
    k      : thermal conductivity (W/m/K)
    alpha  : thermal expansivity (1/K)
    g      : gravity (m/s^2)
    eta    : dynamic viscosity (Pa s)
    a,beta : Nu = a * Ra^beta
    """
    nu = eta / rho                          # kinematic viscosity (m^2/s)
    kappa = k / (rho * c)                   # thermal diffusivity (m^2/s)
    Ra = alpha * g * deltaT * H**3 / (nu * kappa)
    Nu = a * Ra**beta
    Fs = Nu * k * deltaT / H                # surface heat flux (W/m^2)
    energy_per_area = rho * H * (c * deltaT + L)
    t_seconds = energy_per_area / Fs
    return t_seconds / (3600.0 * 24.0 * 365.25)  # years

# example: 1000 km deep magma ocean
print("t_solid (kyr):", magma_ocean_solidification_time(1e6) * 1e3)