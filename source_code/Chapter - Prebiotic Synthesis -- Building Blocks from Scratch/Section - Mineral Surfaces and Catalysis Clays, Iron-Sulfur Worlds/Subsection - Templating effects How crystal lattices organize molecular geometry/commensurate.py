import numpy as np

def best_registry(a1: np.ndarray, a2: np.ndarray, b: np.ndarray, max_index: int = 10):
    """
    Find integers (n,m) minimizing distance |n*a1 + m*a2 - b|.
    a1,a2,b are 2D vectors (numpy arrays).
    Returns (n,m,delta,vec).
    """
    best = (0, 0, np.inf, np.zeros(2))
    for n in range(-max_index, max_index + 1):
        for m in range(-max_index, max_index + 1):
            vec = n * a1 + m * a2
            d = np.linalg.norm(vec - b)
            if d < best[2]:
                best = (n, m, d, vec)
    # normalized mismatch relative to |a1|
    delta = best[2] / np.linalg.norm(a1)
    return best[0], best[1], delta, best[3]

# Example usage: substrate square lattice a=5.0 A, adsorbate spacing b=(7.1,0)
if __name__ == "__main__":
    a = 5.0
    a1 = np.array([a, 0.0])
    a2 = np.array([0.0, a])
    b = np.array([7.1, 0.0])
    n, m, delta, vec = best_registry(a1, a2, b)
    print(f"best (n,m)=({n},{m}), delta={delta:.3f}, vec={vec}")