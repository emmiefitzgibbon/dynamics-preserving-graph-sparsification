import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components

N_SWAPS = 50
POOL = 20
CHECK = 5
LBFGS_ITERS = 60
WEIGHT_CAP = 20.0
USE_PS = (8, 32, 128)


# ----------------------------------------------------------------------------
# Graph basics
# ----------------------------------------------------------------------------

def laplacian(A):
    return np.diag(A.sum(axis=1)) - A


def is_connected(A):
    return connected_components(A, directed=False)[0] <= 1


def edge_list(A):
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float)


def n_target(A, q):
    m = int(np.count_nonzero(np.triu(A, 1)))
    return max(A.shape[0] - 1, int(round(float(q) * m)))


def graph_from_coeffs(A, c, edges):
    ei, ej, we = edges
    B = np.zeros(A.shape)
    keep = c > 0
    B[ei[keep], ej[keep]] = c[keep] * we[keep]
    return B + B.T


def effective_resistance_scores(A):
    ei, ej, we = edge_list(A)
    Lp = np.linalg.pinv(laplacian(A))
    d = np.diag(Lp)
    return we * (d[ei] + d[ej] - 2 * Lp[ei, ej])


# ----------------------------------------------------------------------------
# Diffusion Spectral Error (DSE) sparsification
# ----------------------------------------------------------------------------

def diffusion_error(lam, V, lam_s, V_s):
    lam, mu = np.clip(lam[1:], 1e-12, None), np.clip(lam_s[1:], 1e-12, None)
    O = V[:, 1:].T @ V_s[:, 1:]
    return float(np.sum(O ** 2 * (lam[:, None] - mu[None, :]) ** 2
                        / (2.0 * lam[:, None] * mu[None, :] * (lam[:, None] + mu[None, :]))))


class DiffusionError:
    def __init__(self, A, edges):
        self.A, self.edges = A, edges
        lam, V = np.linalg.eigh(laplacian(A))
        self.lam, self.V = np.clip(lam, 0.0, None), V

    def state(self, c):
        lam_s, V_s = np.linalg.eigh(laplacian(graph_from_coeffs(self.A, c, self.edges)))
        lam_s = np.clip(lam_s, 0.0, None)
        return dict(value=diffusion_error(self.lam, self.V, lam_s, V_s), lam=lam_s, V=V_s)

    def grad(self, st, idx):
        ei, ej, we = (x[idx] for x in self.edges)
        lam, mu = np.clip(self.lam[1:], 1e-12, None), np.clip(st['lam'][1:], 1e-12, None)
        Vf = self.V[:, 1:]
        O = Vf.T @ st['V'][:, 1:]
        d = lam[:, None] - mu[None, :]
        den = 2.0 * lam[:, None] * mu[None, :] * (lam[:, None] + mu[None, :])
        f = d ** 2 / den
        df_dmu = (-2.0 * d * den - d ** 2 * 2.0 * lam[:, None] * (lam[:, None] + 2.0 * mu[None, :])) / den ** 2
        gaps = st['lam'][None, 1:] - st['lam'][:, None]
        gaps[1:][np.diag_indices(len(mu))] = np.inf
        with np.errstate(divide='ignore'):
            inv_gaps = np.where(np.abs(gaps) < 1e-10, 0.0, 1.0 / gaps)
        dp = st['V'][ei] - st['V'][ej]
        C = dp[:, 1:] @ (inv_gaps * (st['V'].T @ Vf @ (O * f))).T
        return 2.0 * we * np.sum(dp * C, axis=1) + (we[:, None] * dp[:, 1:] ** 2) @ np.sum(O ** 2 * df_dmu, axis=0)


def dse_sparsify(A, q):
    edges = edge_list(A)
    err = DiffusionError(A, edges)
    return _optimize(A, q, edges, edge_vectors(A, edges), [err, err])


# ----------------------------------------------------------------------------
# Uniform Spectral Error (USE) sparsification
# ----------------------------------------------------------------------------

def edge_vectors(A, edges):
    lam, U = np.linalg.eigh(laplacian(A))
    if np.count_nonzero(lam > 1e-10) != A.shape[0] - 1:
        raise ValueError('graph must be connected')
    U, lam = U[:, 1:], lam[1:]
    ei, ej, we = edges
    return (U[ei] - U[ej]) * np.sqrt(we)[:, None] / np.sqrt(lam)[None, :]


class SpectralError:
    def __init__(self, A, edges, Z, p):
        self.A, self.edges, self.Z, self.p = A, edges, Z, p

    def state(self, c):
        idx = c > 0
        mu, U = np.linalg.eigh((self.Z[idx].T * c[idx]) @ self.Z[idx])
        d = mu - 1.0
        top = np.abs(d).max()
        if top == 0:
            return dict(value=0.0, G=np.zeros((len(mu), len(mu))))
        value = top * float(np.sum((np.abs(d) / top) ** self.p)) ** (1.0 / self.p)
        g = np.sign(d) * (np.abs(d) / value) ** (self.p - 1)
        return dict(value=value, G=(U * g) @ U.T)

    def grad(self, st, idx):
        Z = self.Z[idx]
        return np.sum((Z @ st['G']) * Z, axis=1)


def use_sparsify(A, q):
    edges = edge_list(A)
    Z = edge_vectors(A, edges)
    errs = [SpectralError(A, edges, Z, p) for p in USE_PS]
    return _optimize(A, q, edges, Z, [errs[0]] + errs)


# ----------------------------------------------------------------------------
# Optimization shared by DSE and USE (Algorithm 1)
# ----------------------------------------------------------------------------

def _spanning_tree(A, edges):
    ei, ej, we = edges
    parent = np.arange(A.shape[0])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    c = np.zeros(len(we))
    for k in np.argsort(-we, kind='stable'):
        a, b = find(ei[k]), find(ej[k])
        if a != b:
            parent[a] = b
            c[k] = 1.0
    return c


def _fill(err, c, target):
    c = c.copy()
    while np.count_nonzero(c) < target:
        st = err.state(c)
        cand = np.where(c == 0)[0]
        best = None
        for e in cand[np.argsort(err.grad(st, cand))[:CHECK]]:
            c[e] = 1.0
            v = err.state(c)['value']
            c[e] = 0.0
            if best is None or v < best[0]:
                best = (v, e)
        c[best[1]] = 1.0
    return c


def _swaps(err, c):
    c = c.copy()
    st = err.state(c)
    for _ in range(N_SWAPS):
        kept, unused = np.where(c > 0)[0], np.where(c == 0)[0]
        if len(unused) == 0:
            break
        g_kept, g_unused = err.grad(st, kept), err.grad(st, unused)
        cut = kept[np.argsort(-g_kept * c[kept])[:POOL]]
        add = unused[np.argsort(g_unused)[:POOL]]
        g = dict(zip(kept, g_kept)) | dict(zip(unused, g_unused))
        pred = [(c[a] * (g[b] - g[a]), a, b) for a in cut for b in add]
        best, checked = None, 0
        for p, a, b in sorted(pred):
            if p >= 0 or (best is not None and checked >= CHECK):
                break
            trial = c.copy()
            trial[b], trial[a] = c[a], 0.0
            if not is_connected(graph_from_coeffs(err.A, trial, err.edges)):
                continue
            checked += 1
            st_try = err.state(trial)
            if st_try['value'] < st['value'] - 1e-12 and (best is None or st_try['value'] < best[1]['value']):
                best = (trial, st_try)
        if best is None:
            break
        c, st = best
    return c


def _weights(err, c):
    idx = np.where(c > 0)[0]
    we = err.edges[2]
    cap = np.log(WEIGHT_CAP * we.max() / we[idx])
    c_try = c.copy()

    def f(theta):
        c_try[idx] = np.exp(theta)
        st = err.state(c_try)
        return st['value'], err.grad(st, idx) * c_try[idx]

    res = minimize(f, np.minimum(np.log(c[idx]), cap), jac=True, method='L-BFGS-B',
                   bounds=[(None, b) for b in cap], options={'maxiter': LBFGS_ITERS, 'ftol': 1e-12})
    out = c.copy()
    out[idx] = np.exp(np.minimum(res.x, cap))
    return out


def _optimize(A, q, edges, Z, errors):
    c = _fill(errors[0], _spanning_tree(A, edges), n_target(A, q))
    c *= Z.shape[1] / np.sum(c * np.sum(Z ** 2, axis=1))
    for err in errors:
        c = _weights(err, _swaps(err, c))
    return graph_from_coeffs(A, c, edges)


# ----------------------------------------------------------------------------
# Effective Resistance and Random sampling
# ----------------------------------------------------------------------------

def inclusion_probs(scores, n_keep):
    s = np.asarray(scores, dtype=float)
    if n_keep >= len(s):
        return np.ones(len(s))
    lo, hi = 0.0, 1.0 / s[s > 0].min()
    while np.minimum(1.0, hi * s).sum() < n_keep:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.minimum(1.0, mid * s).sum() < n_keep else (lo, mid)
    return np.minimum(1.0, 0.5 * (lo + hi) * s)


def _sample(A, q, scores, rng, connected, max_draws):
    ei, ej, we = edge_list(A)
    p = inclusion_probs(scores, float(q) * len(we))
    for _ in range(max_draws if connected else 1):
        keep = rng.random(len(p)) < p
        B = np.zeros(A.shape)
        B[ei[keep], ej[keep]] = we[keep] / p[keep]
        B = B + B.T
        if not connected or is_connected(B):
            return B
    return None


def er_sparsify(A, q, rng, connected=True, max_draws=100_000):
    return _sample(A, q, effective_resistance_scores(A), rng, connected, max_draws)


def random_sparsify(A, q, rng, connected=True, max_draws=100_000):
    return _sample(A, q, np.ones(int(np.count_nonzero(np.triu(A, 1)))), rng, connected, max_draws)
