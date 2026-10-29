"""Evaluation metrics with confidence intervals and significance tests.

Dependency-light on purpose: numpy + scipy + sklearn only, so the DGX enroot
image (nvcr.io#nvidia/pytorch:23.01-py3) needs nothing extra installed.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef)


def operating_point(y, p, mode="youden"):
    """Pick a threshold on the given scores. 'youden' maximises sens+spec-1."""
    ts = np.unique(np.round(p, 4))
    best, bt = -np.inf, 0.5
    for t in ts:
        yh = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        s = sens + spec - 1 if mode == "youden" else f1_score(y, yh, zero_division=0)
        if s > best:
            best, bt = s, t
    return float(bt)


def full_metrics(y, p, thr=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if thr is None:
        thr = operating_point(y, p)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "auc":   roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "auprc": average_precision_score(y, p),
        "acc":   (tp + tn) / len(y),
        "sens":  tp / max(tp + fn, 1),
        "spec":  tn / max(tn + fp, 1),
        "ppv":   tp / max(tp + fp, 1),
        "npv":   tn / max(tn + fn, 1),
        "f1":    f1_score(y, yh, zero_division=0),
        "mcc":   matthews_corrcoef(y, yh) if len(np.unique(yh)) > 1 else 0.0,
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "thr":   thr, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y, p, fn=roc_auc_score, n=2000, alpha=0.05, seed=0):
    """Stratified percentile bootstrap CI for a score function."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        try:
            vals.append(fn(y[idx], p[idx]))
        except ValueError:
            pass
    if not vals:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# DeLong test for two correlated ROC curves (Sun & Xu 2014, O(n log n)).
# ---------------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, float); out[J] = T
    return out


def _structural(preds, m):
    k, n = preds.shape[0], preds.shape[1] - m
    tx = np.array([_midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([_midrank(preds[r, :])  for r in range(k)])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return auc, np.atleast_2d(s)


def delong_test(y, p1, p2):
    """Two-sided p-value for H0: AUC(p1) == AUC(p2) on the same samples."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)                       # positives first
    yy = y[order]; m = int(yy.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    auc, s = _structural(preds, m)
    L = np.array([[1.0, -1.0]])
    var = float(L @ s @ L.T)
    if var <= 0:
        return float(auc[0]), float(auc[1]), 1.0
    z = float((auc[0] - auc[1]) / np.sqrt(var))
    return float(auc[0]), float(auc[1]), float(2 * (1 - stats.norm.cdf(abs(z))))


def permutation_test(y, p, n=5000, seed=0):
    """H0: scores carry no information about y (label permutation on AUC)."""
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, p)
    null = np.array([roc_auc_score(rng.permutation(y), p) for _ in range(n)])
    return float(obs), float((np.sum(null >= obs) + 1) / (n + 1))


def calibration_curve(y, p, bins=5):
    """Equal-count bins -> (mean predicted, observed fraction, count) per bin."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0] -= 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out
