"""Synthetic tabular data: per-client CTGAN generators + fidelity / utility / privacy-proxy evaluation.
(Phase-1 scope: CTGAN. CTAB-GAN+ and federated generator training are planned for Review 2.)"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from .data import NUMERIC, CATEGORICAL, TARGET

warnings.filterwarnings("ignore")


def fit_generator(df: pd.DataFrame, epochs=15, seed=0, batch_size=500):
    from ctgan import CTGAN
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    g = CTGAN(epochs=epochs, batch_size=batch_size, verbose=False, pac=10)
    g.fit(df, discrete_columns=CATEGORICAL + [TARGET])
    return g


def sample_balanced(gens, n: int, pos_frac: float, seed=0) -> pd.DataFrame:
    """Pool samples from one or more generators and rejection-sample on the target column so the
    result has the requested class balance (CTGAN's built-in conditioning is only approximate)."""
    gens = gens if isinstance(gens, (list, tuple)) else [gens]
    pool = pd.concat([g.sample(max(4 * n // len(gens), 2000)) for g in gens], ignore_index=True)
    pos, neg = pool[pool[TARGET] == ">50K"], pool[pool[TARGET] != ">50K"]
    npos = min(int(round(n * pos_frac)), len(pos)); nneg = min(n - int(round(n * pos_frac)), len(neg))
    out = pd.concat([pos.sample(npos, random_state=seed), neg.sample(nneg, random_state=seed)], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """Column-wise statistical fidelity: 1-KS for numerics, 1-TVD for categoricals (1.0 = identical)."""
    cols = {}
    for c in NUMERIC:
        cols[c] = 1 - ks_2samp(real[c], syn[c]).statistic
    for c in CATEGORICAL + [TARGET]:
        pr = real[c].value_counts(normalize=True); ps = syn[c].value_counts(normalize=True)
        idx = pr.index.union(ps.index)
        cols[c] = 1 - 0.5 * np.abs(pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).sum()
    return {"overall": float(np.mean(list(cols.values()))), "per_column": {k: float(v) for k, v in cols.items()}}


def dcr(real_train: pd.DataFrame, syn: pd.DataFrame, holdout: pd.DataFrame, n=1500, seed=0) -> dict:
    """Distance-to-closest-record privacy proxy. If synthetic rows are closer to TRAIN than to HOLDOUT
    rows, the generator may be memorising. Reports median DCR to each (Gower-like, mixed types)."""
    rng = np.random.default_rng(seed)
    s = syn.sample(min(n, len(syn)), random_state=seed)
    tr = real_train.sample(min(4000, len(real_train)), random_state=seed)
    ho = holdout.sample(min(4000, len(holdout)), random_state=seed)
    rngs = {c: max(real_train[c].max() - real_train[c].min(), 1) for c in NUMERIC}

    def dist(A: pd.DataFrame, B: pd.DataFrame):
        out = np.zeros(len(A))
        for i, (_, r) in enumerate(A.iterrows()):
            d = np.zeros(len(B))
            for c in NUMERIC:
                d += np.abs(B[c].to_numpy() - r[c]) / rngs[c]
            for c in CATEGORICAL:
                d += (B[c].to_numpy() != r[c])
            out[i] = (d / (len(NUMERIC) + len(CATEGORICAL))).min()
        return out
    dt, dh = dist(s, tr), dist(s, ho)
    return {"median_dcr_train": float(np.median(dt)), "median_dcr_holdout": float(np.median(dh)),
            "ratio": float(np.median(dt) / max(np.median(dh), 1e-9))}
