import numpy as np
from scipy.stats import gaussian_kde

def estimate_pmf(x_samples, kT=2.5e-21, grid=None, bw_method='scott'):
    """
    Estimate potential of mean force F(x) = -kT * ln p(x) from samples.
    x_samples: 1D numpy array of sampled collective coordinate values.
    kT: thermal energy (J); default corresponds to ~298 K for kB~1.38e-23 J/K.
    grid: optional grid for evaluation; if None, uses linspace over samples.
    bw_method: bandwidth method for gaussian_kde.
    Returns: grid (numpy array), F (numpy array, same shape).
    """
    x = np.asarray(x_samples).ravel()
    if grid is None:
        xmin, xmax = x.min(), x.max()
        grid = np.linspace(xmin, xmax, 512)
    kde = gaussian_kde(x, bw_method=bw_method)          # smooth density
    p = np.clip(kde(grid), 1e-300, None)               # avoid log(0)
    F = -kT * np.log(p)                                # PMF in energy units
    F -= F.min()                                       # set global min to zero
    return grid, F

# Example usage:
# samples = run_simulation_collective_coordinate()
# xs, Fxs = estimate_pmf(samples, kT=4.11e-21)  # kT at 300 K ~ 4.11e-21 J