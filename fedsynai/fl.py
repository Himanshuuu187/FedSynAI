"""Federated training engine: FedAvg / FedProx (+ optional client-level DP), with communication accounting."""
from __future__ import annotations
import copy, time
import numpy as np
import torch
from torch import nn
from .model import MLP, get_flat, set_flat
from .metrics import evaluate
from .privacy import clip_and_noise, epsilon


def local_train(model, X, y, epochs=2, lr=0.05, bs=128, mu=0.0, global_vec=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(X[b]), y[b])
            if mu > 0 and global_vec is not None:       # FedProx proximal term
                loss = loss + 0.5 * mu * ((get_flat_grad(model) - global_vec) ** 2).sum()
            loss.backward()
            opt.step()
    return model


def get_flat_grad(model):  # flat params WITH autograd graph (needed for proximal term)
    return torch.cat([p.view(-1) for p in model.parameters()])


def train_centralized(prep, rounds=20, lr=0.05, seed=0):
    """E1: all data pooled (upper-bound baseline). One 'round' = one epoch."""
    torch.manual_seed(seed)
    m = MLP(prep.n_features)
    hist = []
    for r in range(rounds):
        local_train(m, prep.X_train, prep.y_train, epochs=1, lr=lr, seed=seed + r)
        hist.append({"round": r + 1, **evaluate(m, prep.X_test, prep.y_test)})
    return m, hist


def train_federated(prep, parts, rounds=20, local_epochs=2, lr=0.05, algo="fedavg", mu=0.0,
                    dp_clip=None, dp_noise=0.0, extra=None, seed=0, delta=1e-5):
    """Federated training. `extra` = optional list of (X_syn, y_syn) per client (synthetic augmentation).
    Returns (global_model, history, comm_bytes)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    glob = MLP(prep.n_features)
    n_params = get_flat(glob).numel()
    hist, comm = [], 0
    for r in range(rounds):
        gvec = get_flat(glob)
        deltas, sizes = [], []
        for k, idx in enumerate(parts):
            Xk, yk = prep.X_train[idx], prep.y_train[idx]
            if extra is not None and extra[k] is not None:
                Xk = np.vstack([Xk, extra[k][0]]); yk = np.concatenate([yk, extra[k][1]])
            local = copy.deepcopy(glob)
            local_train(local, Xk, yk, epochs=local_epochs, lr=lr,
                        mu=mu if algo == "fedprox" else 0.0,
                        global_vec=gvec.detach() if algo == "fedprox" else None, seed=seed * 1000 + r * 10 + k)
            deltas.append(get_flat(local) - gvec)
            sizes.append(len(Xk))
        comm += 2 * len(parts) * n_params * 4            # up + down, float32
        clip_frac = None
        if dp_clip is not None:
            upd, clip_frac = clip_and_noise(deltas, dp_clip, dp_noise, gen)
        else:                                            # FedAvg: size-weighted mean of updates
            w = torch.tensor(sizes, dtype=torch.float32); w = w / w.sum()
            upd = sum(wi * d for wi, d in zip(w, deltas))
        set_flat(glob, gvec + upd)
        row = {"round": r + 1, **evaluate(glob, prep.X_test, prep.y_test), "comm_mb": comm / 1e6}
        if dp_clip is not None:
            row["epsilon"] = epsilon(dp_noise, r + 1, delta); row["clip_frac"] = clip_frac
        hist.append(row)
    return glob, hist, comm
