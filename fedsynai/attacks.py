"""Membership-inference evaluation (loss-threshold attack, Yeom et al.). AUC 0.5 = no leakage."""
import numpy as np, torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def _loss(model, X, y):
    model.eval()
    z = model(torch.as_tensor(X))
    return torch.nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(y), reduction="none").numpy()


def membership_auc(model, X_members, y_members, X_non, y_non, n=3000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    a = rng.choice(len(X_members), min(n, len(X_members), len(X_non)), replace=False)
    b = rng.choice(len(X_non), len(a), replace=False)
    lm, ln = _loss(model, X_members[a], y_members[a]), _loss(model, X_non[b], y_non[b])
    scores = -np.concatenate([lm, ln])                  # lower loss => predicted "member"
    labels = np.concatenate([np.ones(len(lm)), np.zeros(len(ln))])
    return float(roc_auc_score(labels, scores))
