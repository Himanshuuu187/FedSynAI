"""Experiment suite: E1 centralized, E2 FedAvg (IID), E3 FedAvg/FedProx (non-IID), E4 FL+DP, E5-lite FedSynAI."""
from __future__ import annotations
import json, os, time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from .data import *
from .fl import train_centralized, train_federated
from .attacks import membership_auc
from .metrics import evaluate
from . import synth

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def summarize(name, desc, hist, model, prep, extra=None):
    last = hist[-1]
    mia = membership_auc(model, prep.X_train, prep.y_train, prep.X_test, prep.y_test)
    return {"id": name, "desc": desc, "accuracy": last["accuracy"], "f1": last["f1"], "auc": last["auc"],
            "comm_mb": last.get("comm_mb", 0.0), "epsilon": last.get("epsilon"), "mia_auc": mia,
            "history": hist, **(extra or {})}


def run_all(rounds=20, alpha=1.0, seed=42, synth_epochs=40, log=print):
    t0 = time.time()
    prep = prepare(seed)
    iid = partition(prep.y_train, 4, None, seed)
    non = partition(prep.y_train, 4, alpha, seed)
    out = {"config": {"rounds": rounds, "alpha": alpha, "seed": seed, "clients": 4, "dataset": "UCI Adult (income)",
                      "train_rows": len(prep.y_train), "test_rows": len(prep.y_test)},
           "partition_iid": describe_partition(iid, prep.y_train).to_dict("records"),
           "partition_noniid": describe_partition(non, prep.y_train).to_dict("records"), "experiments": []}
    E = out["experiments"]
    log("E1 centralized");  m, h = train_centralized(prep, rounds, seed=seed)
    E.append(summarize("E1", "Centralized (upper bound)", h, m, prep))
    log("E2 FedAvg IID");   m, h, _ = train_federated(prep, iid, rounds, seed=seed)
    E.append(summarize("E2", "FedAvg, IID clients", h, m, prep))
    log("E3 FedAvg non-IID"); m, h, _ = train_federated(prep, non, rounds, seed=seed)
    E.append(summarize("E3a", f"FedAvg, non-IID (Dirichlet a={alpha})", h, m, prep))
    log("E3 FedProx non-IID"); m, h, _ = train_federated(prep, non, rounds, algo="fedprox", mu=0.1, seed=seed)
    E.append(summarize("E3b", f"FedProx mu=0.1, non-IID (a={alpha})", h, m, prep))
    for z in (0.05, 0.1, 0.25, 0.5, 1.0):
        log(f"E4 FedAvg+DP z={z}")
        m, h, _ = train_federated(prep, non, rounds, dp_clip=1.0, dp_noise=z, seed=seed)
        E.append(summarize(f"E4-z{z}", f"FedAvg + client-level DP (clip=1, sigma={z}), non-IID", h, m, prep))

    # ---- synthetic data: one CTGAN per client (Phase-1 prototype) ----
    log("Training per-client CTGAN generators")
    gens, synth_rows = [], []
    for k, idx in enumerate(non):
        g = synth.fit_generator(prep.df_train.iloc[idx], epochs=synth_epochs, seed=seed + k)
        gens.append(g)
        s = g.sample(3000)
        fid = synth.fidelity(prep.df_train.iloc[idx], s)   # vs the client's OWN real data
        d = synth.dcr(prep.df_train.iloc[idx], s, prep.df_test, n=300)
        synth_rows.append({"client": f"Client {k+1}", "fidelity": fid["overall"], **d,
                           "synthetic_positive_rate": float((s[TARGET] == ">50K").mean()),
                           "real_positive_rate": float(prep.y_train[idx].mean())})
        if k == 0:
            os.makedirs(RES, exist_ok=True); s.head(500).to_csv(os.path.join(RES, "synthetic_sample_client1.csv"), index=False)
            out["fidelity_per_column_client1"] = fid["per_column"]
    out["synthetic"] = synth_rows

    # TSTR: train-on-synthetic, test-on-real (utility) vs train-on-real
    pooled = synth.sample_balanced(gens, 12000, float(prep.y_train.mean()), seed)
    ct = ColumnTransformer([("n", StandardScaler(), NUMERIC), ("c", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    Xs = ct.fit_transform(pooled); ys = (pooled[TARGET] == ">50K").to_numpy()
    Xr = ct.transform(prep.df_test); yr = prep.y_test
    clf = LogisticRegression(max_iter=500).fit(Xs, ys)
    base = LogisticRegression(max_iter=500).fit(ct.transform(prep.df_train), prep.y_train)
    out["tstr"] = {"train_real_auc": float(roc_auc_score(yr, base.predict_proba(Xr)[:, 1])),
                   "train_synthetic_auc": float(roc_auc_score(yr, clf.predict_proba(Xr)[:, 1])),
                   "train_synthetic_f1": float(f1_score(yr, clf.predict(Xr))),
                   "train_real_f1": float(f1_score(yr, base.predict(Xr)))}

    # ---- E5-lite: FedSynAI = FedAvg + synthetic augmentation to repair label skew ----
    log("E5-lite FedSynAI (synthetic augmentation)")
    from .data import prepare as _p
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)]).fit(prep.df_train)
    extra = []
    for k, idx in enumerate(non):
        syn = synth.sample_balanced(gens, 3000, float(prep.y_train.mean()), seed + k)
        extra.append((pre.transform(syn).astype(np.float32), (syn[TARGET] == ">50K").to_numpy().astype(np.float32)))
    m, h, _ = train_federated(prep, non, rounds, extra=extra, seed=seed)
    E.append(summarize("E5-lite", "FedSynAI-lite: FedAvg + CTGAN synthetic augmentation, non-IID", h, m, prep))
    m, h, _ = train_federated(prep, non, rounds, extra=extra, algo="fedprox", mu=0.1, seed=seed)
    E.append(summarize("E5-lite+Prox", "FedSynAI-lite + FedProx mu=0.1, non-IID", h, m, prep))

    out["runtime_sec"] = time.time() - t0
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
