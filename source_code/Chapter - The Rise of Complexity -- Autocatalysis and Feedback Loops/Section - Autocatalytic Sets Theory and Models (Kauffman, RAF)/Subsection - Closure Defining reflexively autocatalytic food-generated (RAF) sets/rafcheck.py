from typing import Dict, FrozenSet, Iterable, Set, Tuple, List
Reaction = Tuple[FrozenSet[str], FrozenSet[str]]  # (reactants, products)
Catalysis = Dict[str, Set[int]]  # species -> set of reaction indices

def closure(F: Set[str], reactions: List[Reaction], R_idx: Iterable[int]) -> Set[str]:
    """Compute cl_{R'}(F) as least fixed point (iterative expansion)."""
    cl = set(F)
    changed = True
    R_idx = set(R_idx)
    while changed:
        changed = False
        for i in R_idx:
            reactants, products = reactions[i]
            if reactants.issubset(cl) and not products.issubset(cl):
                cl.update(products)
                changed = True
    return cl

def is_raf(F: Set[str], reactions: List[Reaction], catalysis: Catalysis, R_idx: Iterable[int]) -> bool:
    """Return True if reaction subset R' (by indices) is a RAF."""
    cl = closure(F, reactions, R_idx)
    # F-generated: every reactant of each r in R' must be in closure
    for i in R_idx:
        reactants, _ = reactions[i]
        if not reactants.issubset(cl):
            return False
    # Reflexive autocatalysis: each r has at least one catalyst in closure
    for i in R_idx:
        if not any((i in catalysis.get(x, set())) for x in cl):
            return False
    return True

# Example data corresponding to the text example
reactions = [
    (frozenset({"a","b"}), frozenset({"c"})),        # r1
    (frozenset({"c","Pi"}), frozenset({"d"})),      # r2
    (frozenset({"b","Pi"}), frozenset({"e"})),      # r3
]
cat: Catalysis = {"c": {0,2}, "e": {1}}             # c catalyses r1,r3; e catalyses r2
F = {"a","b","Pi"}
assert is_raf(F, reactions, cat, {0,1,2})          # should be True for the example RAF