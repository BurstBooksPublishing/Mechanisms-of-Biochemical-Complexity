#!/usr/bin/env python3
"""
Compute neutral-network statistics for random RNA sequences.
Requires RNAfold in PATH (ViennaRNA).
"""
import shutil, subprocess, random, statistics, tempfile
from itertools import combinations

RNAFOLD = shutil.which("RNAfold")
if RNAFOLD is None:
    raise RuntimeError("RNAfold not found in PATH")

def rand_seq(L):
    return ''.join(random.choice("ACGU") for _ in range(L))

def rnafold(seq):
    # call RNAfold, parse MFE structure and deltaG
    p = subprocess.run([RNAFOLD, "--noPS"], input=seq.encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    out = p.stdout.decode().strip().splitlines()[-1]
    # output format:  ()
    struct, dG = out.split()
    dG = float(dG.strip('()kcal/mol'))
    return struct, dG

def hamming(a,b):
    return sum(x!=y for x,y in zip(a,b))

def survey(L, n):
    groups = {}
    seqs = [rand_seq(L) for _ in range(n)]
    for s in seqs:
        struct, dG = rnafold(s)
        groups.setdefault(struct, []).append(s)
    stats = {}
    for struct, members in groups.items():
        size = len(members)
        mean_hd = statistics.mean(hamming(a,b) for a,b in combinations(members,2)) if size>1 else 0
        stats[struct] = {"size": size, "mean_hamming": mean_hd}
    return stats

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    results = survey(args.length, args.n)
    # print top groups by size
    for struct, info in sorted(results.items(), key=lambda x:-x[1]["size"])[:10]:
        print(f"{struct}\tsize={info['size']}\tmean_hd={info['mean_hamming']:.2f}")