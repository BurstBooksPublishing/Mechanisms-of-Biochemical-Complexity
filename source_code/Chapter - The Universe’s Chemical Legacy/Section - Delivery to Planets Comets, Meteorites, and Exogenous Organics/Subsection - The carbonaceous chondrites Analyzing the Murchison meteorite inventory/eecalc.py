#!/usr/bin/env python3
"""
Compute enantiomeric excess (EE) for amino acids in a CSV file.
CSV must have columns: name, L_abundance, D_abundance (units arbitrary).
Outputs EE (%) and saves summary CSV.
"""
from pathlib import Path
import csv
from typing import List, Dict

def compute_ee(l: float, d: float) -> float:
    # return enantiomeric excess in percent, handle zero denominator
    denom = l + d
    return 0.0 if denom == 0.0 else 100.0 * (l - d) / denom

def process_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline='') as infile, output_path.open('w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['EE_percent'] if reader.fieldnames else ['name','L_abundance','D_abundance','EE_percent']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            l = float(row.get('L_abundance', 0.0))
            d = float(row.get('D_abundance', 0.0))
            row['EE_percent'] = f"{compute_ee(l,d):.3f}"
            writer.writerow(row)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compute enantiomeric excess from amino-acid CSV")
    p.add_argument('input_csv', type=Path, help='Input CSV path')
    p.add_argument('output_csv', type=Path, help='Output CSV path')
    args = p.parse_args()
    process_csv(args.input_csv, args.output_csv)