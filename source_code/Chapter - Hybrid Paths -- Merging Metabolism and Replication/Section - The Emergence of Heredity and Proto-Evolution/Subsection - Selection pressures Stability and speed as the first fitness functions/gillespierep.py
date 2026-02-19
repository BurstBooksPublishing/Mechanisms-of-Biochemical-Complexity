#!/usr/bin/env python3
"""Stochastic simulation of two replicators A and B.
High-quality, reproducible code with explicit parameterization.
"""
import math
import random
from typing import Tuple, List

def gillespie_two_species(k_rep_A: float, k_deg_A: float,
                          k_rep_B: float, k_deg_B: float,
                          N_A0: int, N_B0: int, T_max: float,
                          seed: int = 0) -> Tuple[List[float], List[int], List[int]]:
    random.seed(seed)
    t = 0.0
    N_A, N_B = N_A0, N_B0
    times, A_traj, B_traj = [t], [N_A], [N_B]

    while t < T_max and (N_A + N_B) > 0:
        # propensities: replication proportional to current count
        a_rep_A = k_rep_A * N_A
        a_deg_A = k_deg_A * N_A
        a_rep_B = k_rep_B * N_B
        a_deg_B = k_deg_B * N_B
        a0 = a_rep_A + a_deg_A + a_rep_B + a_deg_B
        if a0 <= 0.0:
            break
        # time to next reaction
        tau = -math.log(random.random()) / a0
        t += tau
        # choose reaction
        r = random.random() * a0
        if r < a_rep_A:
            N_A += 1            # A replicates
        elif r < a_rep_A + a_deg_A:
            N_A = max(0, N_A - 1)  # A degrades
        elif r < a_rep_A + a_deg_A + a_rep_B:
            N_B += 1            # B replicates
        else:
            N_B = max(0, N_B - 1)  # B degrades
        times.append(t); A_traj.append(N_A); B_traj.append(N_B)

    return times, A_traj, B_traj

# Example usage (can be adapted for parameter sweeps in studies)
if __name__ == "__main__":
    times, A, B = gillespie_two_species(
        k_rep_A=0.05, k_deg_A=0.01,
        k_rep_B=0.08, k_deg_B=0.03,
        N_A0=50, N_B0=50, T_max=1000.0, seed=42
    )
    # downstream analysis: compute fixation probabilities, mean trajectories