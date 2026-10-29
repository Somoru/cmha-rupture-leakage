"""Fusion-operator study on the corrected CMHA rupture task.

Same protocol as everything else: 5x5 repeated stratified PATIENT-GROUPED CV,
hyperparameters fixed a priori.

Three questions, in order of importance to the paper:

  Q1  Is the geometry branch salvageable? The GNN sits at 0.473 (chance) while
      logistic regression on descriptors of the SAME point clouds reaches 0.597.
      Compare geom_arch = gnn vs point.

  Q2  THE CONTROL EXPERIMENT. Add a fourth modality that is pure per-sample
      Gaussian noise -- zero information by construction -- and measure the
      weight each operator assigns it. Prediction: softmax attention gives it
      roughly 1/M; precision weighting drives it toward 0. This tests the
      paper's central claim directly, and its outcome does not depend on
      whether any operator beats imaging-alone on AUC.

  Q3  Which fusion operator is best on the three real modalities?

Run:  python src/fusion_study.py --all
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model2 import MultiModalNet, fusion_loss
import metrics as M

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results_fusion"
RES.mkdir(exist_ok=True)

import os
EPOCHS = int(os.environ.get("CMHA_EPOCHS", 120))
BATCH, LR, WD = 16, 1e-3, 1e-4
PATIENCE = int(os.environ.get("CMHA_PATIENCE", 30))
N_FOLDS = int(os.environ.get("CMHA_N_FOLDS", 5))
N_REPEATS = int(os.environ.get("CMHA_N_REPEATS", 5))
NOISE_DIM = 32

R3 = ("geom", "img", "tab")
R4 = ("geom", "img", "tab", "noise")

RUNS = {
    # -- Q1: geometry encoder ------------------------------------------------
    "geom_gnn":        dict(mods=("geom",), fusion="concat", arch="gnn"),
    "geom_point":      dict(mods=("geom",), fusion="concat", arch="point"),
    "img_only":        dict(mods=("img",),  fusion="concat", arch="point"),
    "tab_only":        dict(mods=("tab",),  fusion="concat", arch="point"),
    # -- Q2: the noise control experiment ------------------------------------
    "noise_attn":      dict(mods=R4, fusion="attn",       arch="point"),
    "noise_concat":    dict(mods=R4, fusion="concat",     arch="point"),
    "noise_late_mean": dict(mods=R4, fusion="late_mean",  arch="point"),
    "noise_late_prec": dict(mods=R4, fusion="late_prec",  arch="point"),
    "noise_evidential":dict(mods=R4, fusion="evidential", arch="point"),
    # -- Q3: operators on the three real modalities --------------------------
    "fuse_attn":       dict(mods=R3, fusion="attn",       arch="point"),
    "fuse_concat":     dict(mods=R3, fusion="concat",     arch="point"),
    "fuse_late_mean":  dict(mods=R3, fusion="late_mean",  arch="point"),
    "fuse_late_prec":  dict(mods=R3, fusion="late_prec",  arch="point"),
    "fuse_evidential": dict(mods=R3, fusion="evidential", arch="point"),
    # -- modality dropout on the two most promising --------------------------
    "fuse_late_prec_md": dict(mods=R3, fusion="late_prec", arch="point", md=0.25),
    # fair re-test: warm up with uniform weights so the log-variance heads have
    # something to discriminate before precision weighting switches on
    "fuse_late_prec2":  dict(mods=R3, fusion="late_prec", arch="point", warmup=40),
    "noise_late_prec2": dict(mods=R4, fusion="late_prec", arch="point", warmup=40),
    "fuse_attn_md":      dict(mods=R3, fusion="attn",      arch="point", md=0.25),
}


def load_data():
    d = np.load(PROC / "tabular.npz", allow_pickle=False)
    mesh = np.load(PROC / "meshes.npz", allow_pickle=False)
    vol = np.load(PROC / "volumes.npy")
    y = d["y"].astype(np.int64)
    rng = np.random.default_rng(12345)
    return {
        "tab": np.hstack([d["X_clin"], d["X_morph"]]).astype(np.float32),
        "y": y, "groups": d["groups"].astype(str),
        "mesh": {k: mesh[k] for k in ("x", "Ahat", "mask", "nmask")},
        "vol": vol.astype(np.float32),
        # fixed per sample: a real-looking modality carrying no information
        "noise": rng.standard_normal((len(y), NOISE_DIM)).astype(np.float32),
    }


def make_batch(data, idx, scaler, device, mods):
    b = {}
    if "tab" in mods:
        b["tab"] = torch.tensor(scaler.transform(data["tab"][idx]),
                                dtype=torch.float32, device=device)
    if "geom" in mods:
        for k in ("x", "Ahat", "mask", "nmask"):
            b[k] = torch.tensor(data["mesh"][k][idx], dtype=torch.float32, device=device)
    if "img" in mods:
        b["vol"] = torch.tensor(data["vol"][idx], dtype=torch.float32,
                                device=device).unsqueeze(1)
    if "noise" in mods:
        b["noise"] = torch.tensor(data["noise"][idx], dtype=torch.float32, device=device)
    return b


def run_fold(data, tr, te, cfg, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    y, mods = data["y"], cfg["mods"]
    scaler = StandardScaler().fit(data["tab"][tr])

    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    itr, iva = next(inner.split(np.zeros(len(tr)), y[tr], data["groups"][tr]))
    tr_i, va_i = tr[itr], tr[iva]

    pw = torch.tensor([(y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1)],
                      dtype=torch.float32, device=device)
    net = MultiModalNet(tab_dim=data["tab"].shape[1], modalities=mods,
                        fusion=cfg["fusion"], geom_arch=cfg.get("arch", "point"),
                        noise_dim=NOISE_DIM,
                        modality_dropout=cfg.get("md", 0.0)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)

    warmup = cfg.get("warmup", 0)
    net.use_prec = (warmup == 0)
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        if warmup and ep == warmup:
            net.use_prec = True           # branches now predict; weight them
            bad = 0                       # give the new objective a fresh budget
        net.train()
        perm = np.random.permutation(tr_i)
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 2:                      # BatchNorm needs >1 sample
                continue
            out = net(make_batch(data, idx, scaler, device, mods))
            loss = fusion_loss(out, torch.tensor(y[idx], device=device), pw, cfg["fusion"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            out = net(make_batch(data, va_i, scaler, device, mods))
            vl = fusion_loss(out, torch.tensor(y[va_i], device=device), pw, cfg["fusion"])
        if vl.item() < best - 1e-4:
            best, bad = vl.item(), 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        logit, w, _ = net(make_batch(data, te, scaler, device, mods))
        p = torch.sigmoid(logit).cpu().numpy()
    return p, w.cpu().numpy()


def run(name, data, device):
    cfg = RUNS[name]
    y, g = data["y"], data["groups"]
    t0 = time.time()
    oof_all, aucs, W = [], [], []
    for rep in range(N_REPEATS):
        skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=rep)
        oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, g)):
            p, w = run_fold(data, tr, te, cfg, device, seed=1000 * rep + k)
            oof[te] = p
            W.append(w.mean(0))
        oof_all.append(oof)
        aucs.append(M.full_metrics(y, oof)["auc"])

    cis = [M.bootstrap_ci(y, o, n=1000, seed=i) for i, o in enumerate(oof_all)]
    oof_m = np.mean(oof_all, 0)
    fm = M.full_metrics(y, oof_m)
    w = np.mean(W, 0)
    res = {
        "run": name, "modalities": list(cfg["mods"]), "fusion": cfg["fusion"],
        "geom_arch": cfg.get("arch", "point"), "modality_dropout": cfg.get("md", 0.0),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "auc_ci95": [float(np.mean([c[0] for c in cis])),
                     float(np.mean([c[1] for c in cis]))],
        "pooled": fm,
        "weights": {m: float(v) for m, v in zip(cfg["mods"], w)},
        "runtime_s": round(time.time() - t0, 1),
    }
    if "noise" in cfg["mods"]:
        res["noise_weight"] = float(w[list(cfg["mods"]).index("noise")])
        res["uniform_weight"] = 1.0 / len(cfg["mods"])
    np.savez_compressed(RES / ("oof_" + name + ".npz"), oof=oof_m, y=y)
    (RES / (name + ".json")).write_text(json.dumps(res, indent=2))

    extra = ""
    if "noise_weight" in res:
        extra = "  NOISE w=%.3f (uniform=%.3f)" % (res["noise_weight"], res["uniform_weight"])
    print("  %-20s AUC %.3f +- %.3f  CI [%.3f,%.3f]  MCC %.3f  w=%s%s  (%.0fs)"
          % (name, res["auc_mean"], res["auc_std"], res["auc_ci95"][0],
             res["auc_ci95"][1], fm["mcc"],
             np.round(w, 3), extra, res["runtime_s"]), flush=True)
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

    names = list(RUNS) if a.all else [a.run]
    out, failed = {}, []
    for n in names:
        if (RES / (n + ".json")).exists():          # RESUME (see train_cv.py)
            import json as _j
            out[n] = _j.loads((RES / (n + ".json")).read_text())
            print("  = already done, skipping %s" % n, flush=True)
            continue
        try:
            out[n] = run(n, data, dev)
        except Exception as e:
            failed.append((n, repr(e)))
            print("  !! FAILED %s -> %r" % (n, e), flush=True)
            import traceback; traceback.print_exc()

    # ---------------- reports -------------------------------------------------
    print("\n=============== Q1: geometry encoder ===============")
    for n in ("geom_gnn", "geom_point", "tab_only", "img_only"):
        if n in out:
            print("  %-16s AUC %.3f +- %.3f" % (n, out[n]["auc_mean"], out[n]["auc_std"]))

    print("\n=============== Q2: NOISE CONTROL (the key result) ===============")
    print("  a modality carrying zero information; uniform share would be 0.250")
    print("  %-18s %8s %10s %10s" % ("operator", "AUC", "w(noise)", "vs uniform"))
    for n in ("noise_attn", "noise_concat", "noise_late_mean",
              "noise_late_prec", "noise_evidential"):
        if n in out:
            r = out[n]
            print("  %-18s %8.3f %10.3f %9.2fx"
                  % (r["fusion"], r["auc_mean"], r["noise_weight"],
                     r["noise_weight"] / r["uniform_weight"]))

    print("\n=============== Q3: operators, 3 real modalities ===============")
    ref = out.get("img_only", {}).get("auc_mean")
    print("  %-22s %8s %8s %8s   weights" % ("run", "AUC", "AUPRC", "MCC"))
    for n in ("fuse_attn", "fuse_concat", "fuse_late_mean", "fuse_late_prec",
              "fuse_evidential", "fuse_late_prec_md", "fuse_attn_md"):
        if n in out:
            r = out[n]
            flag = ""
            if ref and r["auc_mean"] > ref:
                flag = "  <== beats img-only"
            print("  %-22s %8.3f %8.3f %8.3f   %s%s"
                  % (n, r["auc_mean"], r["pooled"]["auprc"], r["pooled"]["mcc"],
                     {k: round(v, 3) for k, v in r["weights"].items()}, flag))
    if ref:
        print("\n  reference img_only AUC %.3f" % ref)
        y = load_data()["y"]
        for n in ("fuse_late_prec", "fuse_attn", "fuse_evidential"):
            if n in out and (RES / ("oof_" + n + ".npz")).exists():
                a1 = np.load(RES / ("oof_" + n + ".npz"))["oof"]
                a2 = np.load(RES / "oof_img_only.npz")["oof"]
                a, b, p = M.delong_test(y, a1, a2)
                print("  DeLong %-18s %.3f vs img %.3f  p=%.4f" % (n, a, b, p))

    (RES / "study_summary.json").write_text(json.dumps(out, indent=2))
    if failed:
        print("\nfailed: %s" % failed)
    print("\nwrote results_fusion/")
