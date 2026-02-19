import random, math, collections
# Parameters
M = 4                    # alphabet size (A,C,G,U)
L0 = 6                   # target polymer length
q = 0.92                 # per-site fidelity in templated copying
k_s = 1e-4               # spontaneous ligation rate
k_t = 1e-2               # templated extension rate (effective)
steps = 100000

# Pools and templates
pool = collections.Counter({'A':1000,'C':1000,'G':1000,'U':1000})
templates = ['A'*L0]     # initial seed template

seq_counts = collections.Counter()

for t in range(steps):
    # spontaneous formation attempt
    if random.random() < k_s:
        seq = ''.join(random.choices(list(pool.keys()), k=L0))
        seq_counts[seq]+=1
    # templated extension attempt
    if random.random() < k_t and templates:
        templ = random.choice(templates)
        # build copy with fidelity q per site
        copy = []
        for base in templ:
            if random.random() < q:
                # correct complement (simple A<->U, C<->G mapping)
                comp = {'A':'U','U':'A','C':'G','G':'C'}[base]
                copy.append(comp)
            else:
                copy.append(random.choice(list(pool.keys())))
        seq = ''.join(copy)
        seq_counts[seq]+=1
        # allow successful copies to act as new templates probabilistically
        if random.random() < 0.01:
            templates.append(seq)

# Report top sequences and site entropies
top = seq_counts.most_common(10)
print("Top sequences:", top)
# compute site entropies
pos_counts = [collections.Counter() for _ in range(L0)]
for s,c in seq_counts.items():
    for i,ch in enumerate(s):
        pos_counts[i][ch]+=c
ents = []
for pc in pos_counts:
    total = sum(pc.values())
    H = -sum((v/total)*math.log(v/total,2) for v in pc.values())
    ents.append(H)
print("Site entropies (bits):", ents)