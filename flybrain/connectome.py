import numpy as np
import scipy.sparse as sp

# Neurotransmitters treated as inhibitory (glutamate is context dependent in flies; this is a simple default).
INHIBITORY = {"GABA", "GLUT"}


def normalize(W):
    # Scale every neuron's incoming weights so the total absolute input is 1.
    W = sp.csr_matrix(W, dtype=np.float32)
    col = np.asarray(abs(W).sum(axis=0)).ravel()
    col[col == 0] = 1.0
    return (W @ sp.diags(1.0 / col)).tocsr().astype(np.float32)


def synthetic_fly(n=2000, mean_degree=30, inhib_frac=0.2, seed=0):
    # Fly-like stand-in: sparse, heavy-tailed hubs, log-normal synapse strengths, ~20% inhibitory.
    rng = np.random.default_rng(seed)
    nnz = n * mean_degree
    pre = rng.integers(0, n, nnz)
    popularity = rng.pareto(2.5, n) + 1.0
    popularity /= popularity.sum()
    post = rng.choice(n, nnz, p=popularity)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    w = rng.lognormal(0.0, 1.0, pre.size).astype(np.float32)
    inhib = rng.random(n) < inhib_frac
    w[inhib[pre]] *= -1
    return normalize(sp.coo_matrix((w, (pre, post)), shape=(n, n)))


def load_flywire(path, max_neurons=None, min_syn=5):
    # Loads a FlyWire-style connections table (pre id, post id, synapse count, optional neurotransmitter).
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for name in names:
            for lc, c in cols.items():
                if lc.startswith(name):
                    return c
        raise KeyError("none of %s in %s" % (names, list(df.columns)))

    pre_c = pick("pre_root_id", "pre_id", "pre")
    post_c = pick("post_root_id", "post_id", "post")
    syn_c = pick("syn_count", "weight", "syn")
    nt_c = next((c for lc, c in cols.items() if lc.startswith("nt")), None)

    df = df[df[syn_c] >= min_syn]
    ids = pd.unique(pd.concat([df[pre_c], df[post_c]]))
    if max_neurons and len(ids) > max_neurons:
        deg = pd.concat([df[pre_c], df[post_c]]).value_counts()
        ids = deg.index[:max_neurons].to_numpy()
        df = df[df[pre_c].isin(ids) & df[post_c].isin(ids)]
    index = {v: i for i, v in enumerate(ids)}
    pre = df[pre_c].map(index).to_numpy()
    post = df[post_c].map(index).to_numpy()
    w = df[syn_c].to_numpy(np.float32)
    if nt_c is not None:
        neg = df[nt_c].astype(str).str.upper().isin(INHIBITORY).to_numpy()
        w = np.where(neg, -w, w).astype(np.float32)
    n = len(ids)
    return normalize(sp.coo_matrix((w, (pre, post)), shape=(n, n)))
