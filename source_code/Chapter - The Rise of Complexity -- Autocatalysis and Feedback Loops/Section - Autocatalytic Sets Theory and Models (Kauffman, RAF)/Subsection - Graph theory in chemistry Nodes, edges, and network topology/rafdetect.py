import networkx as nx

# species list and reaction definitions (reactants->products, catalysts list)
species = ["HCHO","GA","G3P","Tetroses"]  # replace with full inventory
reactions = {
    "r1": (["HCHO","HCHO"], ["GA"]),       # HCHO + HCHO -> GA
    "r2": (["GA","HCHO"], ["G3P"]),        # GA + HCHO -> G3P
    "r3": (["G3P","HCHO"], ["Tetroses"])  # etc.
}
catalysts = {"r1":["GA"], "r2":["GA"], "r3":["G3P"]}  # catalysis assignments
food = {"HCHO"}  # externally supplied set

# build bipartite graph: species nodes prefixed 's:' and reaction nodes 'r:'
B = nx.DiGraph()
for s in species: B.add_node("s:"+s, bipartite=0)
for r, (react, prod) in reactions.items():
    B.add_node("r:"+r, bipartite=1)
    for a in react: B.add_edge("s:"+a, "r:"+r)  # reactant -> reaction
    for b in prod:  B.add_edge("r:"+r, "s:"+b)  # reaction -> product
    for cat in catalysts.get(r, []): B.add_edge("s:"+cat, "r:"+r)  # catalyst -> reaction

# species-projection: edge u->v if u catalyzes some reaction producing v
Sproj = nx.DiGraph()
for u in species:
    for r, (_, prod) in reactions.items():
        if u in catalysts.get(r, []):
            for v in prod: Sproj.add_edge(u, v)

# find SCCs that are candidate RAF modules (exclude trivial food-only SCCs)
sccs = [c for c in nx.strongly_connected_components(Sproj)]
candidates = [c for c in sccs if not c.issubset(food)]

# iterative RAF pruning: remove reactions lacking catalysts in food+products
available = set(food)
reactions_active = set(reactions.keys())
changed = True
while changed:
    changed = False
    for r in list(reactions_active):
        cats = set(catalysts.get(r, []))
        if cats and not (cats & (available | set().union(*[set(reactions[p][1]) for p in reactions_active]))):
            reactions_active.remove(r); changed = True
    # update available with products of remaining reactions
    available = set(food) | set().union(*[set(reactions[r][1]) for r in reactions_active])
# output candidates and active reactions
print("SCC candidates:", candidates)
print("Remaining reactions after pruning:", reactions_active)