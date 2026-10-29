"""Why does fusion (0.647) underperform imaging alone (0.694)?

Two candidate explanations, tested directly:

  A. The entropy regulariser. It pushes attention toward uniform, which is
     actively wrong when modality quality is unequal (imaging 0.694 vs geometry
     0.518). Test: lambda_ent = 0 lets attention concentrate.
  B. The weak branch. Attention is a convex combination of embeddings, so a
     near-chance modality injects noise it cannot fully suppress. Test: drop
     geometry, drop clinical.

Same protocol as everything else: 5x5 repeated stratified patient-grouped CV,
fixed hyperparameters.
"""
import sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cv as T
import metrics as M

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

RUNS = [
    ("img only        (reference)", ("img",),                 0.01),
    ("fusion  lam=0.01 (current)",  ("geom", "img", "tab"),   0.01),
    ("fusion  lam=0    (A)",        ("geom", "img", "tab"),   0.00),
    ("img+tab lam=0    (B: -geom)", ("img", "tab"),           0.00),
    ("geom+img lam=0   (B: -tab)",  ("geom", "img"),          0.00),
]


def main():
    d = np.load(PROC / "tabular.npz", allow_pickle=True)
    mesh = np.load(PROC / "meshes.npz", allow_pickle=True)
    vol = np.load(PROC / "volumes.npy")
    data = {
        "tab": np.hstack([d["X_clin"], d["X_morph"]]).astype(np.float32),
        "y": d["y"].astype(np.int64),
        "groups": d["groups"].astype(str),
        "mesh": {k: mesh[k] for k in ("x", "Ahat", "mask", "nmask")},
        "vol": vol.astype(np.float32),
    }
    y, g = data["y"], data["groups"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("n=%d ruptured=%d device=%s\n" % (len(y), y.sum(), dev), flush=True)

    out = {}
    for label, mods, lam in RUNS:
        T.LAM_ENT = lam
        t0 = time.time()
        oof_all, aucs, attn = [], [], []
        for rep in range(T.N_REPEATS):
            skf = StratifiedGroupKFold(T.N_FOLDS, shuffle=True, random_state=rep)
            oof = np.zeros(len(y))
            for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, g)):
                p, a = T.run_fold(data, tr, te, mods, dev, seed=1000 * rep + k)
                oof[te] = p
                attn.append(a.mean(0))
            oof_all.append(oof)
            aucs.append(M.full_metrics(y, oof)["auc"])
        cis = [M.bootstrap_ci(y, o, n=1000, seed=i) for i, o in enumerate(oof_all)]
        lo = float(np.mean([c[0] for c in cis])); hi = float(np.mean([c[1] for c in cis]))
        fm = M.full_metrics(y, np.mean(oof_all, 0))
        a = np.mean(attn, 0) if len(mods) > 1 else None
        out[label] = (np.mean(aucs), np.std(aucs), lo, hi, fm, np.mean(oof_all, 0), a, mods)
        ent = (-(a * np.log(a + 1e-9)).sum() / np.log(len(a))) if a is not None else float("nan")
        print("  %-28s AUC %.3f +- %.3f  CI [%.3f,%.3f]  MCC %.3f  attn %s  H/Hmax %.2f  (%.0fs)"
              % (label, np.mean(aucs), np.std(aucs), lo, hi, fm["mcc"],
                 np.round(a, 3) if a is not None else "-", ent, time.time() - t0), flush=True)

    ref = out["img only        (reference)"]
    print("\n%-28s %7s %7s %7s %7s" % ("", "AUC", "AUPRC", "spec", "MCC"))
    for k, v in out.items():
        print("%-28s %7.3f %7.3f %7.3f %7.3f" % (k, v[0], v[4]["auprc"], v[4]["spec"], v[4]["mcc"]))
    print("\nDeLong vs imaging-only (repeat-averaged predictions):")
    for k, v in out.items():
        if k == "img only        (reference)":
            continue
        a1, a2, p = M.delong_test(y, v[5], ref[5])
        flag = "  <== beats imaging" if (a1 > a2 and p < 0.05) else ""
        print("  %-28s %.3f vs %.3f  p=%.4f%s" % (k, a1, a2, p, flag))


if __name__ == "__main__":
    main()
