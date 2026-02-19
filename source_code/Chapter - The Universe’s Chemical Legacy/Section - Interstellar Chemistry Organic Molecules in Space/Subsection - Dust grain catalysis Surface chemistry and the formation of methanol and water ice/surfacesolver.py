#!/usr/bin/env python3
"""
ODE solver for simple grain-surface hydrogenation (CO -> CH3OH, O -> H2O).
Units: densities in cm^-3 for gas, molecules per grain for surface.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Physical params
T = 10.0                         # K
kB = 1.380649e-16                # erg/K (cgs)
mH = 1.6735575e-24               # g
nu = 1e12                        # s^-1 attempt frequency
a_barrier = 1.0e-8               # cm (approximate barrier width)

# Grain and surface parameters
r_gr = 0.1e-4                    # grain radius cm (0.1 micron)
sigma_gr = np.pi * r_gr**2
N_s = 1e6                        # sites per grain
n_H = 1e5                        # cm^-3 (example dense core)
n_gr = 1e-12 * n_H               # cm^-3 (typical dust-to-gas by number)

# Gas-phase abundances (initial)
n_CO_g = 1e-5 * n_H
n_O_g = 1e-5 * n_H
v_th = lambda m: np.sqrt(8*kB*T/(np.pi*m))

# adsorption flux per grain (molecules s^-1)
def flux_to_grain(n_g, mass):
    return 0.25 * n_g * v_th(mass) * sigma_gr

# barrier-dependent rates (thermal hopping + simple tunneling)
def hop_rate(E_diff):
    return nu * np.exp(-E_diff / (kB * T))

def tunneling_rate(E_a):
    # WKB-like estimate; prefactor nu
    exponent = -2 * a_barrier * np.sqrt(2 * mH * E_a) / 1.0545718e-27
    return nu * np.exp(exponent)

# Example energy parameters (erg)
E_diff_H = 4.0e-14               # H diffusion barrier
E_diff_CO = 2.0e-13
E_a_CO_H = 6.0e-14               # activation barrier for CO+H (erg)

# Reaction rates per grain (s^-1 per reactive pair)
k_hop_H = hop_rate(E_diff_H)
k_hop_CO = hop_rate(E_diff_CO)
k_CO_H = max(tunneling_rate(E_a_CO_H), (k_hop_H + k_hop_CO)/N_s)

# O + H reaction (assumed barrierless on ice)
k_O_H = (k_hop_H + hop_rate(2.0e-13)) / N_s

# ODE system: [CO_s, H_s, HCO_s, H2CO_s, CH3OH_s, O_s, OH_s, H2O_s]
def dydt(t, y):
    COs, Hs, HCOs, H2COs, CH3OHs, Os, OHs, H2Os = y

    # adsorption source terms (per grain)
    src_CO = flux_to_grain(n_CO_g, 28*mH)
    src_O = flux_to_grain(n_O_g, 16*mH)
    src_H = flux_to_grain(n_H, mH)

    # reactions (per grain)
    r1 = k_CO_H * COs * Hs
    r2 = k_hop_H/N_s * HCOs * Hs
    r3 = k_hop_H/N_s * H2COs * Hs
    rO1 = k_O_H * Os * Hs
    rO2 = k_hop_H/N_s * OHs * Hs

    dCOs = src_CO - r1
    dHs = src_H - r1 - r2 - r3 - rO1 - rO2
    dHCOs = r1 - r2
    dH2COs = r2 - r3
    dCH3OHs = r3
    dOs = src_O - rO1
    dOHs = rO1 - rO2
    dH2Os = rO2

    return [dCOs, dHs, dHCOs, dH2COs, dCH3OHs, dOs, dOHs, dH2Os]

# Initial surface populations (molecules per grain)
y0 = [100.0, 1.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0]

# Integrate
sol = solve_ivp(
    dydt,
    [0, 1e6],
    y0,
    method='BDF',
    rtol=1e-8,
    atol=1e-12
)

# Plot results
labels =
\subsection{Complex detection: Polycyclic aromatic hydrocarbons (PAHs) and cyanopolyynes in molecular clouds}
\subsection{Surface-mediated synthesis and low-temperature ion-molecule chemistry}

The previous subsections established the importance of surface-mediated synthesis and low-temperature ion-molecule chemistry for building complex organics. Those processes provide plausible feedstock and excitation conditions that make PAHs and cyanopolyynes observable in molecular clouds.

Concept. Polycyclic aromatic hydrocarbons (PAHs) are large, predominantly aromatic, carbonaceous molecules that dominate certain mid-infrared emission features. Cyanopolyynes are linear carbon-chain nitriles, $\mathrm{HC}_{2n+1}\mathrm{N}$, with strong dipoles and rich rotational spectra. Both classes trace different facets of chemical complexity: PAHs as reservoirs of condensed aromatic carbon, and cyanopolyynes as signatures of active carbon-chain growth.

Theory. Detection rests on well-understood spectroscopic selection rules and excitation physics. PAH emission arises when UV photons transiently heat small grains or molecules, producing vibrational fluorescence at characteristic bands near 3.3, 6.2, 7.7, 8.6, and 11.2 $\upmu$m. Band intensities depend on the ionization state, size distribution, and local radiation field. Cyanopolyynes emit via rotational transitions at cm--mm wavelengths; their line strengths scale with permanent dipole moments and population of rotational levels under local excitation conditions.

Rotational line observations commonly use the optically thin, local thermodynamic equilibrium (LTE) approximation for column density estimation. For a linear molecule, a practical form is
\begin{equation}[H]\label{eq:col_density}
N = \frac{3k}{8\pi^3\nu S\mu^2}\;Q_{\rm rot}(T_{\rm ex})\;\exp\!\left(\frac{E_u}{kT_{\rm ex}}\right)\;\int T_b\,dv,
\end{equation}
where $N$ is the total column density, $\nu$ is transition frequency, $S$ is the line strength, $\mu$ is the permanent dipole moment, $Q_{\rm rot}$ is the rotational partition function, $E_u$ is the upper-level energy, $T_b$ is the brightness temperature, and $T_{\rm ex}$ is the excitation temperature. For non-LTE conditions, statistical equilibrium codes (e.g., RADEX) are required.

Formation pathways reflect environment. In cold, dark clouds like TMC-1, ion-molecule sequences dominate:
\begin{enumerate}
\item Chain initiation by acetylene-derived ions and radicals, e.g., $\mathrm{C}_2\mathrm{H}_2 + \mathrm{C}^+ \rightarrow \mathrm{C}_3\mathrm{H}^+ + \mathrm{H}$.
\item Growth through successive addition and hydrogen abstraction, often involving CN or CCH: $\mathrm{C}_{2n}\mathrm{H}_2 + \mathrm{CN} \rightarrow \mathrm{HC}_{2n+1}\mathrm{N} + \mathrm{H}$.
\item Recombination and radiative stabilization yield the neutral cyanopolyyne.
\end{enumerate}
These routes are efficient at $T\sim 10$ K when barrierless or near-barrierless. In photon-dominated regions (PDRs) or circumstellar envelopes, neutral-neutral reactions and high-temperature chemistry contribute, and PAHs commonly form in AGB star outflows then survive into the ISM.

Example. TMC-1 exhibits exceptionally strong cyanopolyyne lines: $\mathrm{HC}_3\mathrm{N}$, $\mathrm{HC}_5\mathrm{N}$, and $\mathrm{HC}_7\mathrm{N}$ show column densities from $10^{12}$ to $10^{13}\ \mathrm{cm^{-2}}$. Mid-IR spectra of the Orion Bar display classic PAH bands, with band-to-continuum ratios diagnostic of PAH ionization fractions. Combining IR band ratios with radio rotational surveys constrains both the aromatic fraction of carbon and the steady-state carbon-chain abundances.

Application. Practical detection combines instrumentation, analysis, and modelling:
\begin{itemize}
\item Observational modalities: mid-IR spectroscopy (JWST, Spitzer) for PAH bands; heterodyne radio/mm spectroscopy (GBT, IRAM, ALMA) for cyanopolyynes.
\item Data analysis: Gaussian decomposition, LTE column density via Eq. \eqref{eq:col_density}, and non-LTE radiative transfer when required.
\item Chemical inference: compare observed abundances with kinetic network models that include ion-molecule rates measured in the laboratory or calculated theoretically.
\end{itemize}

The following production-ready Python snippet fits a single Gaussian to a cyanopolyyne line and computes LTE column density using Eq. \eqref{eq:col_density}. Replace the placeholders with measured values and molecular constants.

\begin{lstlisting}[language=Python,caption={Gaussian fit and LTE column density for a cyanopolyyne line.},label={lst:cyano_fit}]
import numpy as np
from scipy.optimize import curve_fit
from astropy import constants as const
from astropy import units as u

# Gaussian model for brightness temperature spectrum
def gauss(v, T0, v0, sigma):
    return T0 * np.exp(-0.5 * ((v - v0) / sigma)**2)

# Example input arrays: velocity (km/s) and T_b (K)
v = np.loadtxt('velocity_kms.txt')            # replace with data file
Tb = np.loadtxt('brightness_temp_K.txt')

# Initial fit
p0 = [Tb.max(), v[np.argmax(Tb)], 0.2]        # initial guesses
popt, pcov = curve_fit(gauss, v, Tb, p0=p0)
T0, v0, sigma = popt

# Integrated intensity (K km/s)
int_Tb_dv = np.sqrt(2*np.pi) * T0 * sigma

# Molecular parameters (example for HC3N J->J-1)
nu = 90979.0e6      # Hz, transition frequency
S =  (3/ (2*1+1))   # placeholder; use real line strength
mu_D = 3.724        # Debye, dipole moment for HC3N
mu = mu_D * 1e-18 * u.C * u.cm                    # convert Debye to SI*cm units

# Partition function and upper-level energy (SI)
Tex = 10.0           # K, excitation temperature
Qrot =  (k:=const.k_B.value) * Tex / (const.h.value * 2 * np.pi * nu)  # approximate; replace with tabulated Q
Eu =  (const.h.value * nu) *  (1) / const.k_B.value  # approx E_u in K; adjust for J

# Column density using Eq. (1) (SI consistent)
prefac = 3*const.k_B.value / (8*np.pi**3 * nu * S * (mu.to(u.C*u.m)).value**2)
N = prefac * Qrot * np.exp(Eu / Tex) * (int_Tb_dv * 1e3)  # K km/s -> K m/s

print(f"Fitted center velocity: {v0:.3f} km/s")
print(f"Integrated intensity: {int_Tb_dv:.3f} K km/s")
print(f"Estimated column density: {N:.3e} m^-2")