import numpy as np
from scipy.stats import zscore

def kl_divergence(p, q, eps=1e-12):
    # p, q: 1D arrays of mole fractions, sum to 1
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def bootstrap_pvalue(obs_p, abiotic_ensemble, n_boot=10000, random_state=None):
    rng = np.random.default_rng(random_state)
    obs_kl = kl_divergence(obs_p, np.mean(abiotic_ensemble, axis=0))
    # compute KLs for each abiotic realization against ensemble mean
    kl_vals = np.array([kl_divergence(a, np.mean(abiotic_ensemble, axis=0))
                        for a in abiotic_ensemble])
    # p-value as fraction of abiotic KL >= observed KL
    pval = (np.sum(kl_vals >= obs_kl) + 1) / (len(kl_vals) + 1)
    return obs_kl, kl_vals, pval

def composite_score(phi, z_iso, eea, d_kl, weights=(1.0,1.0,1.0,1.0)):
    w = np.asarray(weights)
    vals = np.asarray([phi, z_iso, eea, d_kl])
    return float(np.dot(w, vals)/w.sum())

# Example usage (replace with real spectral or mass-spec data)
# obs, abiotic_ensemble should be arrays of same length with mole fractions.