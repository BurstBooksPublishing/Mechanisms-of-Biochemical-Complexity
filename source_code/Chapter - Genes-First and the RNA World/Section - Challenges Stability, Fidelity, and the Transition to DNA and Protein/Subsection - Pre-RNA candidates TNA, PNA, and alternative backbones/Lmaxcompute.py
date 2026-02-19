#!/usr/bin/env python3
"""Compute L_max = floor( ln(Q_min) / ln(1-epsilon) ) for multiple epsilons.
Assumes small epsilon approximation is optional.
"""
from typing import Iterable, Tuple
import math

def compute_L_max(epsilons: Iterable[float], Q_min: float,
                  use_approx: bool = False) -> Tuple[Tuple[float,int], ...]:
    """
    epsilons: iterable of per-site error rates (0 < epsilon < 1)
    Q_min: minimal acceptable error-free copy probability (0 < Q_min < 1)
    use_approx: use small-epsilon approximation L_max ~ -ln(Q_min)/epsilon
    Returns tuple of (epsilon, L_max)
    """
    if not (0.0 < Q_min < 1.0):
        raise ValueError("Q_min must lie in (0,1)")
    results = []
    for e in epsilons:
        if not (0.0 < e < 1.0):
            raise ValueError("epsilon values must lie in (0,1)")
        if use_approx and e < 0.01:
            L = int(math.floor(-math.log(Q_min) / e))
        else:
            denom = math.log(1.0 - e)
            L = int(math.floor(math.log(Q_min) / denom))
        results.append((e, L))
    return tuple(results)

# Example usage: compare representative epsilons for RNA, TNA, and PNA.
if __name__ == "__main__":
    eps = [1e-2, 5e-3, 1e-3]    # rough per-site error rates (non-enzymatic)
    Q_min = 1e-2                # require 1% chance of perfect copy
    for e, Lmax in compute_L_max(eps, Q_min):
        print(f"epsilon={e:.1e}, L_max={Lmax}")