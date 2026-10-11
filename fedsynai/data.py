"""Data loading, preprocessing and client partitioning (IID / Dirichlet non-IID)."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adult.csv")
TARGET = "income"
NUMERIC = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "gender", "native-country"]


def load_adult(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True).replace("?", np.nan).dropna()
    df = df[NUMERIC + CATEGORICAL + [TARGET]].reset_index(drop=True)
    df[TARGET] = df[TARGET].str.strip().str.rstrip(".")
    return df


@dataclass
class Prepared:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    df_train: pd.DataFrame      # raw rows (needed by the synthetic generators)
    df_test: pd.DataFrame
    n_features: int


def prepare(seed: int = 42, test_size: float = 0.2, max_rows: int | None = None) -> Prepared:
    df = load_adult()
    if max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    ct = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    Xtr = ct.fit_transform(tr).astype(np.float32)
    Xte = ct.transform(te).astype(np.float32)
    ytr = (tr[TARGET] == ">50K").to_numpy().astype(np.float32)
    yte = (te[TARGET] == ">50K").to_numpy().astype(np.float32)
    return Prepared(Xtr, ytr, Xte, yte, tr, te, Xtr.shape[1])


def partition(y: np.ndarray, n_clients: int = 4, alpha: float | None = None, seed: int = 42):
    """Return list of index arrays. alpha=None -> IID; small alpha -> strong label skew (non-IID).
    Dirichlet label-skew partitioning is the standard benchmark protocol for FL heterogeneity."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if alpha is None:
        rng.shuffle(idx)
        return [np.sort(a) for a in np.array_split(idx, n_clients)]
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            ci = idx[y == c]
            rng.shuffle(ci)
            p = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(p) * len(ci)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(ci, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= 200:      # avoid degenerate empty clients
            return [np.sort(np.array(p)) for p in parts]


def describe_partition(parts, y) -> pd.DataFrame:
    return pd.DataFrame([{"client": f"Client {i+1}", "rows": len(p),
                          "positive_rate": float(y[p].mean())} for i, p in enumerate(parts)])
