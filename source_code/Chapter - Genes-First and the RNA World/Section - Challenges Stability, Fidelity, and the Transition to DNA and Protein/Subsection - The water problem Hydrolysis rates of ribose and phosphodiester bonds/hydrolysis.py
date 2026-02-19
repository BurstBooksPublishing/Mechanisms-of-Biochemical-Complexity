#!/usr/bin/env python3
"""
Compute pseudo-first-order half-life for RNA hydrolysis using additive mechanistic channels.
Defaults are illustrative; replace with measured k or Arrhenius parameters.
"""
import math
import numpy as np

R = 1.9872036e-3  # kcal mol^-1 K^-1

def arrhenius(A, Ea_kcal, T_K):
    return A * math.exp(-Ea_kcal / (R * T_K))

def hydroxide_conc(pH, pKw=14.0):
    # Approximate at 25 C; adjust pKw for T if known.
    return 10 ** (-(pKw - pH))

def k_obs_from_channels(pH, T_K,
                        kw=1e-12,  # s^-1 uncatalyzed water attack (illustrative)
                        kH=0.0,    # M^-1 s^-1 acid-specific (illustrative)
                        kOH=1e6,   # M^-1 s^-1 base-catalyzed (illustrative)
                        metal_channels=None):
    """
    metal_channels: list of tuples (k_M [M^-1 s^-1], [M] in M)
    """
    if metal_channels is None:
        metal_channels = [(1e3, 0.0)]  # default no metal catalysis
    H = 10 ** (-pH)
    OH = hydroxide_conc(pH)
    k = kw + kH * H + kOH * OH
    for kM, conc in metal_channels:
        k += kM * conc
    return k

def half_life_seconds(k_obs):
    if k_obs <= 0:
        return math.inf
    return math.log(2) / k_obs

# Example usage: neutral pH, 298 K, 1 mM Mg2+ with illustrative k_M
if __name__ == "__main__":
    pH = 7.0
    T = 298.15
    metal_channels = [(1e4, 1e-3)]  # k_M (M^-1 s^-1), [M]=1 mM
    k = k_obs_from_channels(pH, T, metal_channels=metal_channels)
    t_half = half_life_seconds(k)
    print(f"k_obs = {k:.3e} s^-1, t1/2 = {t_half/3600:.2f} hours")