import RNA
import sys

def analyze_rna(sequence):
    # Create fold compound with default temperature
    fc = RNA.fold_compound(sequence)
    # Compute MFE structure and energy
    mfe_struct, mfe_energy = fc.mfe()
    # Compute partition function and base-pair probabilities
    fc.pf()  # populates internal probability data
    length = len(sequence)
    bp_prob = [[0.0]*length for _ in range(length)]
    for i in range(1, length+1):
        for j in range(i+1, length+1):
            p = fc.bpp(i, j)  # base-pair probability between positions i,j
            bp_prob[i-1][j-1] = p
            bp_prob[j-1][i-1] = p
    return {'sequence': sequence,
            'mfe_structure': mfe_struct,
            'mfe_energy': mfe_energy,
            'bp_prob': bp_prob}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_rna.py SEQUENCE", file=sys.stderr)
        sys.exit(2)
    result = analyze_rna(sys.argv[1].strip())
    print("MFE structure:", result['mfe_structure'])
    print("MFE energy (kcal/mol):", result['mfe_energy'])
    # show pair probability for first ten positions as example
    for i, row in enumerate(result['bp_prob'][:10], start=1):
        print(f"pos {i} probs (first10):", ['{:.3f}'.format(x) for x in row[:10]])