"""Stacked generalisation: the fusion operator that CAN zero out a dead branch.

Job 22329 tested five fusion operators. None beat imaging alone (0.709), and
none could down-weight a provably uninformative modality -- given a pure-noise
branch they all assigned it about the uniform share:

    attention 0.261   late_prec 0.258   evidential 0.243   (uniform = 0.250)

Every one of those operators decides the weights from the SAME forward pass that
produces the prediction, using representations fitted on the training data. They
have no held-out evidence about whether a branch actually generalises.

Stacking does. Each branch is trained separately; inner out-of-fold predictions
on the training split give an honest estimate of what each branch is worth; a
meta-learner is fitted on those. With an L1 penalty it can drive a useless
branch's coefficient to exactly zero -- the capability every operator above
lacked. It is also the textbook method for this problem, which is the point: the
finding would be that at n~100 the standard approach beats the learned one.

Protocol is unchanged: 5x5 repeated stratified PATIENT-GROUPED CV. Grouping is
enforced in the inner loop too, otherwise the meta-features leak.

    python src/stacking.py --all
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model2 import MultiModalNet, fusion_loss
from fusion_study import load_data, make_batch, NOISE_DIM
import metrics as M

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_fusion"
RES.mkdir(exist_ok=True)

import os
# Stacking is the most expensive stage: each outer fold trains every branch
# (N_INNER + 1) times. On a slower GPU set CMHA_N_REPEATS=3 and
# CMHA_EPOCHS_BRANCH=60 to cut it roughly in half.
EPOCHS_BRANCH = int(os.environ.get("CMHA_EPOCHS_BRANCH", 80))
BATCH, LR, WD = 16, 1e-3, 1e-4
N_FOLDS = int(os.environ.get("CMHA_N_FOLDS", 5))
N_REPEATS = int(os.environ.get("CMHA_N_REPEATS", 5))
N_INNER = int(os.environ.get("CMHA_N_INNER", 3))

R3 = ("geom", "img", "tab")
R4 = ("geom", "img", "tab", "noise")

RUNS = {
    "stack_l1_3mod":   dict(mods=R3, meta="l1"),
    "stack_l1_4mod":   dict(mods=R4, meta="l1"),      # <- the key test
    "stack_l2_4mod":   dict(mods=R4, meta="l2"),
    "stack_mean_4mod": dict(mods=R4, meta="mean"),    # fixed-weight reference
}


def train_branch(data, tr_idx, mod, device, seed):
    """Train a single-modality model on tr_idx. Fixed epochs, no early stopping:
    the inner OOF structure is what protects the meta-learner, and a fixed budget
    keeps every branch comparable."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = data["y"]
    scaler = StandardScaler().fit(data["tab"][tr_idx])
    pw = torch.tensor([(y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)],
                      dtype=torch.float32, device=device)
    net = MultiModalNet(tab_dim=data["tab"].shape[1], modalities=(mod,),
                        fusion="concat", geom_arch="point",
                        noise_dim=NOISE_DIM).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    net.train()
    for _ in range(EPOCHS_BRANCH):
        perm = np.random.permutation(tr_idx)
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 2:
                continue
            out = net(make_batch(data, idx, scaler, device, (mod,)))
            loss = fusion_loss(out, torch.tensor(y[idx], device=device), pw, "concat")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
    return net, scaler


def branch_logits(net, scaler, data, idx, mod, device):
    net.eval()
    with torch.no_grad():
        logit, _, _ = net(make_batch(data, idx, scaler, device, (mod,)))
    return logit.cpu().numpy()


def outer_fold(data, tr, te, mods, meta_kind, device, seed):
    y = data["y"]
    M_ = len(mods)

    # ---- inner OOF meta-features on the training split (patient-grouped) ----
    Z_tr = np.zeros((len(tr), M_), dtype=np.float64)
    inner = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed)
    for j, (itr, iva) in enumerate(inner.split(np.zeros(len(tr)), y[tr],
                                               data["groups"][tr])):
        for k, mod in enumerate(mods):
            net, sc = train_branch(data, tr[itr], mod, device, seed + 17 * j + k)
            Z_tr[iva, k] = branch_logits(net, sc, data, tr[iva], mod, device)

    # ---- branches refit on the full training split, applied to test ----------
    Z_te = np.zeros((len(te), M_), dtype=np.float64)
    for k, mod in enumerate(mods):
        net, sc = train_branch(data, tr, mod, device, seed + 101 + k)
        Z_te[:, k] = branch_logits(net, sc, data, te, mod, device)

    if meta_kind == "mean":
        p = 1.0 / (1.0 + np.exp(-Z_te.mean(1)))
        return p, np.full(M_, 1.0 / M_)

    ms = StandardScaler().fit(Z_tr)
    meta = LogisticRegression(
        penalty="l1" if meta_kind == "l1" else "l2",
        C=0.5, solver="liblinear", class_weight="balanced", max_iter=2000)
    meta.fit(ms.transform(Z_tr), y[tr])
    p = meta.predict_proba(ms.transform(Z_te))[:, 1]
    return p, meta.coef_.ravel().copy()


def run(name, data, device):
    cfg = RUNS[name]
    mods, y, g = cfg["mods"], data["y"], data["groups"]
    t0 = time.time()
    oof_all, aucs, coefs = [], [], []
    for rep in range(N_REPEATS):
        skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=rep)
        oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, g)):
            p, c = outer_fold(data, tr, te, mods, cfg["meta"], device,
                              seed=1000 * rep + k)
            oof[te] = p
            coefs.append(c)
        oof_all.append(oof)
        aucs.append(M.full_metrics(y, oof)["auc"])
        print("    [%s] repeat %d/%d AUC=%.3f" % (name, rep + 1, N_REPEATS,
                                                  aucs[-1]), flush=True)

    C = np.array(coefs)
    absw = np.abs(C).mean(0)
    absw = absw / max(absw.sum(), 1e-9)
    cis = [M.bootstrap_ci(y, o, n=1000, seed=i) for i, o in enumerate(oof_all)]
    oof_m = np.mean(oof_all, 0)
    fm = M.full_metrics(y, oof_m)

    res = {
        "run": name, "modalities": list(mods), "meta": cfg["meta"],
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "auc_ci95": [float(np.mean([c[0] for c in cis])),
                     float(np.mean([c[1] for c in cis]))],
        "pooled": fm,
        "meta_coef_mean": {m: float(v) for m, v in zip(mods, C.mean(0))},
        "weights": {m: float(v) for m, v in zip(mods, absw)},
        "zero_coef_rate": {m: float((np.abs(C[:, i]) < 1e-8).mean())
                           for i, m in enumerate(mods)},
        "runtime_s": round(time.time() - t0, 1),
    }
    if "noise" in mods:
        i = list(mods).index("noise")
        res["noise_weight"] = float(absw[i])
        res["noise_zero_rate"] = float((np.abs(C[:, i]) < 1e-8).mean())
        res["uniform_weight"] = 1.0 / len(mods)

    np.savez_compressed(RES / ("oof_" + name + ".npz"), oof=oof_m, y=y)
    (RES / (name + ".json")).write_text(json.dumps(res, indent=2))

    extra = ""
    if "noise_weight" in res:
        extra = ("  NOISE w=%.3f (uniform %.3f, zeroed in %.0f%% of folds)"
                 % (res["noise_weight"], res["uniform_weight"],
                    100 * res["noise_zero_rate"]))
    print("  %-18s AUC %.3f +- %.3f  CI [%.3f,%.3f]  MCC %.3f  w=%s%s  (%.0fs)"
          % (name, res["auc_mean"], res["auc_std"], res["auc_ci95"][0],
             res["auc_ci95"][1], fm["mcc"], np.round(absw, 3), extra,
             res["runtime_s"]), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s torch=%s" % (dev, torch.__version__), flush=True)
    data = load_data()
    print("n=%d ruptured=%d patients=%d\n"
          % (len(data["y"]), data["y"].sum(), len(set(data["groups"]))), flush=True)

    out = {}
    for n in (list(RUNS) if a.all else [a.run]):
        if (RES / (n + ".json")).exists():          # RESUME (see train_cv.py)
            out[n] = json.loads((RES / (n + ".json")).read_text())
            print("  = already done, skipping %s" % n, flush=True)
            continue
        try:
            out[n] = run(n, data, dev)
        except Exception as e:
            print("  !! FAILED %s -> %r" % (n, e), flush=True)
            import traceback; traceback.print_exc()

    print("\n=============== STACKING vs THE FIVE OPERATORS ===============")
    print("  Can the operator zero out a modality that carries no information?")
    print("  %-18s %8s %10s %14s" % ("run", "AUC", "w(noise)", "zeroed folds"))
    for n, r in out.items():
        if "noise_weight" in r:
            print("  %-18s %8.3f %10.3f %13.0f%%"
                  % (n, r["auc_mean"], r["noise_weight"],
                     100 * r["noise_zero_rate"]))
    print("\n  reference from the reference run (all near the 0.250 uniform share):")
    print("    attention 0.261   late_prec 0.258   evidential 0.243")

    print("\n  mean meta-coefficients (signed):")
    for n, r in out.items():
        print("    %-18s %s" % (n, {k: round(v, 3)
                                    for k, v in r["meta_coef_mean"].items()}))

    ref = RES / "oof_img_only.npz"
    if ref.exists() and out:
        y = load_data()["y"]
        a2 = np.load(ref)["oof"]
        print("\n  DeLong vs img_only:")
        for n in out:
            f = RES / ("oof_" + n + ".npz")
            if f.exists():
                x, b, p = M.delong_test(y, np.load(f)["oof"], a2)
                flag = "  <== beats img" if (x > b and p < 0.05) else ""
                print("    %-18s %.3f vs %.3f  p=%.4f%s" % (n, x, b, p, flag))

    (RES / "stacking_summary.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results_fusion/")
