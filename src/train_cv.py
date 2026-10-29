"""Repeated stratified PATIENT-GROUPED cross-validation for CMHA rupture prediction.

Every configuration is evaluated with the identical protocol so results are
comparable. Hyperparameters are FIXED a priori (never tuned on the evaluation
folds) -- with n=105 any test-set tuning would invalidate the whole exercise.

Run one config:      python train_cv.py --config rupture_clean_fusion
Run everything:      python train_cv.py --all
"""
import argparse, json, time, sys, traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FusionNet, loss_fn
import metrics as M

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

import os
def _env(name, default, cast=int):
    return cast(os.environ.get(name, default))

# Tunable so a slower box (e.g. a Colab T4) can trade repeats for wall-clock.
# Defaults reproduce the cluster runs exactly.
EPOCHS = _env("CMHA_EPOCHS", 120)
BATCH, LR, WD = 16, 1e-3, 1e-4
PATIENCE = _env("CMHA_PATIENCE", 30)
LAM_ENT = 0.01          # attention-entropy weight; 0 lets attention concentrate
N_FOLDS = _env("CMHA_N_FOLDS", 5)
N_REPEATS = _env("CMHA_N_REPEATS", 5)


# Ordered by importance: the paper's own results run FIRST, so a late failure
# still leaves the headline numbers on disk. The leak_* configs quantify
# artifacts and are the most expendable, so they go last.
CONFIGS = {
    # ---- corrected protocol: unimodal (headline) ----
    "rupture_clean_tab": dict(task="rupture", mods=("tab",), leaky=False),
    "rupture_clean_geom": dict(task="rupture", mods=("geom",), leaky=False),
    "rupture_clean_img": dict(task="rupture", mods=("img",), leaky=False),
    # ---- corrected protocol: full fusion (headline) ----
    "rupture_clean_fusion": dict(task="rupture", mods=("geom", "img", "tab"), leaky=False),
    # ---- corrected protocol: leave-one-modality-out ablation ----
    "rupture_clean_no_geom": dict(task="rupture", mods=("img", "tab"), leaky=False),
    "rupture_clean_no_img": dict(task="rupture", mods=("geom", "tab"), leaky=False),
    "rupture_clean_no_tab": dict(task="rupture", mods=("geom", "img"), leaky=False),
    # ---- leakage quantification (cheap first, imaging-based last) ----
    "leak_rupture_scores": dict(task="rupture", mods=("tab",), leaky=True),
    "leak_detection_tab": dict(task="detection", mods=("tab",), leaky=False),
    "leak_detection_geom": dict(task="detection", mods=("geom",), leaky=False),
    "leak_detection_fusion": dict(task="detection", mods=("geom", "img", "tab"), leaky=False),
}


def load_data(task, leaky):
    d = np.load(PROC / "tabular.npz", allow_pickle=True)
    tab = np.hstack([d["X_clin"], d["X_morph"]]).astype(np.float32)
    if leaky:
        tab = np.hstack([tab, d["baselines"].astype(np.float32)])
    out = dict(tab=tab, y=d["y"].astype(np.int64),
               groups=d["groups"].astype(str), aid=d["aneurysm_id"].astype(str))

    if task == "detection":
        # Patients and controls MUST come from the detection-specific files: they
        # share an 11-clinical + 27-morphology layout (38 cols). tabular.npz is the
        # rupture matrix (16 clinical + 27 morphology = 43 cols) and includes Shape
        # and one-hot location, both of which are undefined for a control -- mixing
        # them both breaks vstack and would add a second leak on top of the one
        # being measured.
        dp = PROC / "tabular_patients_detection.npz"
        dc = PROC / "tabular_controls.npz"
        for f in (dp, dc):
            if not f.exists():
                raise FileNotFoundError(
                    "detection configs need %s (run prep_controls.py)" % f.name)
        p = np.load(dp, allow_pickle=False)
        c = np.load(dc, allow_pickle=False)
        ptab = p["X_clin_padded"].astype(np.float32)
        ctab = c["X_clin_padded"].astype(np.float32)
        assert ptab.shape[1] == ctab.shape[1], (ptab.shape, ctab.shape)
        out["tab"] = np.vstack([ptab, ctab])
        out["y"] = np.concatenate([np.ones(len(ptab), np.int64),
                                   np.zeros(len(ctab), np.int64)])
        out["groups"] = np.concatenate([p["groups"].astype(str),
                                        c["groups"].astype(str)])
        out["aid"] = np.concatenate([p["aneurysm_id"].astype(str),
                                     c["aneurysm_id"].astype(str)])

    suffix = "_detection" if task == "detection" else ""
    for key, stem, ext in (("mesh", "meshes", ".npz"), ("vol", "volumes", ".npy")):
        p = PROC / f"{stem}{suffix}{ext}"
        if p.exists():
            out[key] = np.load(p, allow_pickle=True)
    return out


def make_batch(data, idx, scaler, device, mods):
    b = {}
    if "tab" in mods:
        b["tab"] = torch.tensor(scaler.transform(data["tab"][idx]),
                                dtype=torch.float32, device=device)
    if "geom" in mods:
        mesh = data["mesh"]
        b["x"] = torch.tensor(mesh["x"][idx], dtype=torch.float32, device=device)
        b["Ahat"] = torch.tensor(mesh["Ahat"][idx], dtype=torch.float32, device=device)
        b["mask"] = torch.tensor(mesh["mask"][idx], dtype=torch.float32, device=device)
        b["nmask"] = torch.tensor(mesh["nmask"][idx], dtype=torch.float32, device=device)
    if "img" in mods:
        b["vol"] = torch.tensor(data["vol"][idx], dtype=torch.float32,
                                device=device).unsqueeze(1)
    return b


def run_fold(data, tr, te, mods, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = data["y"]
    scaler = StandardScaler().fit(data["tab"][tr])

    # inner split for early stopping (still patient-grouped)
    g = data["groups"][tr]
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    itr, iva = next(inner.split(np.zeros(len(tr)), y[tr], g))
    tr_i, va_i = tr[itr], tr[iva]

    pw = torch.tensor([(y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1)],
                      dtype=torch.float32, device=device)
    net = FusionNet(tab_dim=data["tab"].shape[1], modalities=mods).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)

    best, best_state, bad = np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        perm = np.random.permutation(tr_i)
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 2:                      # BatchNorm needs >1 sample
                continue
            b = make_batch(data, idx, scaler, device, mods)
            logit, alpha = net(b)
            loss, _ = loss_fn(logit, torch.tensor(y[idx], device=device), alpha, pw,
                              lam_ent=LAM_ENT)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            b = make_batch(data, va_i, scaler, device, mods)
            logit, alpha = net(b)
            vl, _ = loss_fn(logit, torch.tensor(y[va_i], device=device), alpha, pw,
                            lam_ent=LAM_ENT)
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
        b = make_batch(data, te, scaler, device, mods)
        logit, alpha = net(b)
        p = torch.sigmoid(logit).cpu().numpy()
    return p, alpha.cpu().numpy()


def run_config(name, device):
    cfg = CONFIGS[name]
    data = load_data(cfg["task"], cfg["leaky"])
    y, groups = data["y"], data["groups"]
    mods = tuple(m for m in cfg["mods"]
                 if m == "tab"
                 or (m == "geom" and "mesh" in data)
                 or (m == "img" and "vol" in data))
    if not mods:
        raise RuntimeError(name + ": no usable modality present")
    if mods != tuple(cfg["mods"]):
        print("  ! " + name + ": requested " + str(cfg["mods"])
              + ", running " + str(mods) + " (missing inputs)")

    t0 = time.time()
    oof_all, per_repeat, attn = [], [], []
    for rep in range(N_REPEATS):
        skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=rep)
        oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
            p, a = run_fold(data, tr, te, mods, device, seed=1000 * rep + k)
            oof[te] = p
            attn.append(a.mean(0))
        oof_all.append(oof)
        per_repeat.append(M.full_metrics(y, oof))
        print("  [%s] repeat %d/%d  AUC=%.3f"
              % (name, rep + 1, N_REPEATS, per_repeat[-1]["auc"]), flush=True)

    oof_mean = np.mean(oof_all, axis=0)
    scalar_keys = [k for k, v in per_repeat[0].items() if isinstance(v, (int, float))]

    # Two DIFFERENT estimators; never mix them in one "value +- CI" claim.
    #   mean/std   : average over repeats of each repeat's own OOF metric.
    #                This is the headline number -- it reflects a single model.
    #   pooled_*   : metrics of the repeat-AVERAGED predictions. Averaging across
    #                repeats is an ensemble and scores noticeably higher, so its
    #                CI must never be attached to the per-repeat point estimate.
    # Each repeat also gets its own bootstrap CI; averaging those bounds gives an
    # interval that actually brackets the headline number.
    ci_per_repeat = [M.bootstrap_ci(y, o, n=1000, seed=i) for i, o in enumerate(oof_all)]
    auc_ci95 = (float(np.mean([c[0] for c in ci_per_repeat])),
                float(np.mean([c[1] for c in ci_per_repeat])))

    res = {
        "config": name, "task": cfg["task"], "modalities": list(mods), "leaky": cfg["leaky"],
        "n": int(len(y)), "n_pos": int(y.sum()),
        "per_repeat": per_repeat,
        "mean": {k: float(np.mean([r[k] for r in per_repeat])) for k in scalar_keys},
        "std": {k: float(np.std([r[k] for r in per_repeat])) for k in scalar_keys},
        "auc_ci95": auc_ci95,
        "auc_ci95_note": "mean of per-repeat bootstrap CIs; matches mean.auc",
        "pooled": M.full_metrics(y, oof_mean),
        "pooled_auc_ci95": M.bootstrap_ci(y, oof_mean),
        "pooled_note": "repeat-averaged predictions (ensemble); scores above mean.auc",
        "calibration": M.calibration_curve(y, oof_mean),
        "attention_mean": (np.mean(attn, 0).tolist() if len(mods) > 1 else None),
        "attention_std": (np.std(attn, 0).tolist() if len(mods) > 1 else None),
        "runtime_s": round(time.time() - t0, 1),
    }
    _, pval = M.permutation_test(y, oof_mean, n=2000)
    res["permutation_p"] = pval
    np.savez_compressed(RES / ("oof_" + name + ".npz"),
                        oof=oof_mean, all=np.array(oof_all), y=y, groups=groups)
    (RES / (name + ".json")).write_text(json.dumps(res, indent=2))
    print("  -> %s: AUC %.3f+-%.3f CI95 [%.3f,%.3f] p=%.4f  (%.1fs)"
          % (name, res["mean"]["auc"], res["std"]["auc"],
             res["auc_ci95"][0], res["auc_ci95"][1], pval, res["runtime_s"]), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute configs that already have a results JSON")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s  torch=%s" % (dev, torch.__version__), flush=True)
    names = list(CONFIGS) if a.all else [a.config]
    done, failed = [], []
    for n in names:
        # RESUME: a finished config is never redone. Colab disconnects, so each
        # config's JSON is the checkpoint -- just re-run the cell to continue.
        if (RES / (n + ".json")).exists() and not a.force:
            print("  = already done, skipping %s (use --force to redo)" % n, flush=True)
            done.append(n)
            continue
        try:
            run_config(n, dev)
            done.append(n)
        except FileNotFoundError as e:
            failed.append((n, "missing input: %s" % e))
            print("  ! skipped %s: %s" % (n, e), flush=True)
            if not a.skip_missing:
                raise
        except Exception as e:
            # One bad configuration must not destroy the whole suite -- every
            # other config still gets written to results/.
            failed.append((n, "%s: %s" % (type(e).__name__, e)))
            print("  !! FAILED %s -> %s: %s" % (n, type(e).__name__, e), flush=True)
            traceback.print_exc()

    print("\n================ SUITE SUMMARY ================")
    print("completed %d/%d" % (len(done), len(names)))
    for n in done:
        print("  ok     %s" % n)
    for n, why in failed:
        print("  FAILED %s  (%s)" % (n, why))
    if not done:
        sys.exit(1)
