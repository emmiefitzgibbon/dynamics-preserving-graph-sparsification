from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

DATA = Path(__file__).resolve().parent / 'data'
BASE_SEED = 12345


def _load(name, symmetrize=False):
    path = DATA / name
    W = np.loadtxt(path) if path.suffix == '.txt' else np.load(path)
    A = (W + W.T) / 2.0 if symmetrize else W.astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def _symmetric(n, ei, ej, w):
    A = np.zeros((n, n))
    A[ei, ej] = w
    A[ej, ei] = w
    return A


def scale_free(rng, n=100, d=5, hub_frac=0.15, hub_weight_mu=1.5, leaf_weight_mu=-1.0, weight_sigma=0.5):
    ei, ej = [list(x) for x in np.triu_indices(d, 1)]
    degree = np.zeros(n, dtype=int)
    degree[:d] = d - 1
    for new in range(d, n):
        p = degree[:new] + 1.0
        for t in rng.choice(new, size=min(d, new), replace=False, p=p / p.sum()):
            ei.append(new)
            ej.append(int(t))
            degree[new] += 1
            degree[t] += 1
    ei, ej = np.array(ei), np.array(ej)
    hub = degree >= np.quantile(degree, 1 - hub_frac)
    w = np.array([rng.lognormal(hub_weight_mu if hub[a] or hub[b] else leaf_weight_mu, weight_sigma)
                  for a, b in zip(ei, ej)])
    return _symmetric(n, ei, ej, w)


def erdos_renyi(rng, n=100, mean_deg=20.0, weight_mu=(1.2, -0.8), weight_sigma=0.5):
    p = min(0.85, mean_deg / (n - 1))
    for _ in range(200):
        i, j = np.triu_indices(n, k=1)
        exists = rng.random(len(i)) < p
        ei, ej = i[exists], j[exists]
        if len(ei) < n - 1:
            continue
        A = _symmetric(n, ei, ej, rng.lognormal(rng.choice(weight_mu, size=len(ei)), weight_sigma))
        if connected_components(A, directed=False)[0] == 1:
            return A
    raise RuntimeError('no connected graph')


def complete(rng, n=66, weight_mu=0.0, weight_sigma=0.5):
    ei, ej = np.triu_indices(n, k=1)
    return _symmetric(n, ei, ej, rng.lognormal(weight_mu, weight_sigma, len(ei)))


def core_periphery(rng=None, n=60, lam_hi=10.0, lam_lo=1.0):
    A = np.full((n, n), lam_lo / n)
    A[:n // 2, :n // 2] = (2 * lam_hi - lam_lo) / n
    np.fill_diagonal(A, 0.0)
    return A


def sbm(rng, n=90, k=3, p_in=0.35, p_out=0.05, weight_mu_in=1.0, weight_mu_out=-1.5, weight_sigma=0.5):
    sizes = [n // k + (i < n % k) for i in range(k)]
    off = np.cumsum([0] + sizes[:-1])
    A = np.zeros((n, n))
    for b in range(k):
        ii, jj = np.triu_indices(sizes[b], k=1)
        keep = rng.random(len(ii)) < p_in
        w = rng.lognormal(weight_mu_in, weight_sigma, keep.sum())
        A[off[b] + ii[keep], off[b] + jj[keep]] += w
        A[off[b] + jj[keep], off[b] + ii[keep]] += w
    for a in range(k):
        for b in range(a + 1, k):
            ii, jj = np.repeat(np.arange(sizes[a]), sizes[b]), np.tile(np.arange(sizes[b]), sizes[a])
            keep = rng.random(len(ii)) < p_out
            w = rng.lognormal(weight_mu_out, weight_sigma, keep.sum())
            A[off[a] + ii[keep], off[b] + jj[keep]] += w
            A[off[b] + jj[keep], off[a] + ii[keep]] += w
    for b in range(k):
        perm = rng.permutation(sizes[b])
        for i in range(1, sizes[b]):
            u, v = off[b] + perm[i], off[b] + perm[rng.integers(0, i)]
            if A[u, v] == 0:
                A[u, v] = A[v, u] = rng.lognormal(weight_mu_in, weight_sigma)
    for b in range(k - 1):
        u, v = off[b] + rng.integers(0, sizes[b]), off[b + 1] + rng.integers(0, sizes[b + 1])
        if A[u, v] == 0:
            A[u, v] = A[v, u] = rng.lognormal(weight_mu_out, weight_sigma)
    return A


def p_out_for_mixing(n, k, p_in, mixing):
    size = n // k
    return float(mixing * (size - 1) * p_in / (1.0 - mixing) / (n - size))


NETWORKS = {
    'macaque_cortex': lambda rng: _load('macaque_neural_adjacency.npy'),
    'human_connectome': lambda rng: _load('connectome_66_weights.txt', symmetrize=True),
    'c_intestinalis': lambda rng: _load('cintestinalis_adjacency.npy'),
    'c_elegans': lambda rng: _load('celegans_adjacency.npy'),
    'sbm_pout_0.042': lambda rng: sbm(rng, p_in=0.35, p_out=p_out_for_mixing(90, 3, 0.35, 0.2), weight_mu_out=0.0),
    'sbm_pout_0.085': lambda rng: sbm(rng, p_in=0.7, p_out=p_out_for_mixing(90, 3, 0.7, 0.2), weight_mu_out=0.0),
    'sbm_pout_0.113': lambda rng: sbm(rng, p_in=0.35, p_out=p_out_for_mixing(90, 3, 0.35, 0.4)),
    'scale_free_d8': lambda rng: scale_free(rng, d=8),
    'scale_free_d15': lambda rng: scale_free(rng, d=15),
    'erdos_renyi_p0.2': lambda rng: erdos_renyi(rng),
    'erdos_renyi_p1': lambda rng: complete(rng),
    'core_periphery': core_periphery,
}


def build(name, realization=0):
    seed = BASE_SEED if realization == 0 else BASE_SEED + 104_729 * realization
    return NETWORKS[name](np.random.default_rng(seed))
