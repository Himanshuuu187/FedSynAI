import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@torch.no_grad()
def evaluate(model, X, y) -> dict:
    model.eval()
    p = torch.sigmoid(model(torch.as_tensor(X))).numpy()
    pred = (p >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "auc": float(roc_auc_score(y, p))}
