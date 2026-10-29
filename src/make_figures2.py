"""Publication figures + LaTeX tables for the manuscript.

Reads results/ and results_fusion/, writes figs/ as both PDF (vector, for
LaTeX) and PNG (preview). Every figure degrades gracefully if its inputs are
missing, so this is safe to run at any stage.

Design decisions, held deliberately:

* AUC values are INTERVAL ESTIMATES, so they are drawn as dot-and-whisker
  (forest) plots, not bars. A bar implies a count with a meaningful zero; an AUC
  of 0.5 is the null, not 0. The chance line is drawn at 0.5 for that reason.
* Categorical hues come from a colorblind-safe palette, validated with the
  dataviz validator (worst adjacent pair Delta E 9.1 protan / 22.9 normal, all
  four inside the lightness band). Hues are assigned in FIXED order and never
  cycled.
* Two of the four hues fall below 3:1 contrast on a light surface, so every mark
  carries a visible numeric label -- identity is never colour-alone.
* One axis per figure. No dual axes anywhere.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RESF = ROOT / "results_fusion"
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

# validated categorical palette, fixed order
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_RED, C_GREY = "#d03b3b", "#8a8a86"
INK, MUTED = "#1a1a19", "#5c5c58"

plt.rcParams.update({
    "font.size": 8.5, "figure.dpi": 200, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def load(d, name):
    p = d / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote figs/%s.pdf/.png" % stem)


def forest(ax, rows, chance=0.5):
    """rows: (label, mean, lo, hi, colour). Dot + CI whisker, newest at top."""
    ys = np.arange(len(rows))[::-1]
    for y, (lab, m, lo, hi, c) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", alpha=.85)
        ax.plot([m], [y], "o", color=c, ms=6.5, mec="white", mew=1.2, zorder=3)
        ax.text(hi + .012, y, "%.3f" % m, va="center", fontsize=7.8, color=INK)
    ax.axvline(chance, ls=":", lw=1, color=C_GREY, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("AUC (5×5 patient-grouped CV, mean and 95% CI)")
    ax.set_xlim(0.30, 1.10)
    ax.grid(axis="y", visible=False)


# --------------------------------------------------------------- Figure 1
def fig_protocol():
    spec = [("Detection: fusion", "leak_detection_fusion", C_RED),
            ("Detection: mesh only", "leak_detection_geom", C_RED),
            ("Detection: clinical only", "leak_detection_tab", C_RED),
            ("Rupture + PHASES/ELAPSS", "leak_rupture_scores", C_ORANGE),
            ("Rupture, corrected protocol", "rupture_clean_img", C_AQUA)]
    rows = []
    for lab, key, c in spec:
        r = load(RES, key)
        if r:
            rows.append((lab, r["mean"]["auc"], r["auc_ci95"][0], r["auc_ci95"][1], c))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 2.15))
    forest(ax, rows)
    ax.set_title("Evaluation protocol determines the result", fontsize=9.5,
                 loc="left", color=INK, pad=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=C_RED, marker="o", ls="-", lw=2, ms=6,
               label="leaking protocol"),
        Line2D([], [], color=C_ORANGE, marker="o", ls="-", lw=2, ms=6,
               label="score leak"),
        Line2D([], [], color=C_AQUA, marker="o", ls="-", lw=2, ms=6,
               label="corrected"),
        Line2D([], [], color=C_GREY, ls=":", lw=1, label="chance")],
        fontsize=7, loc="lower left", ncol=2, handlelength=1.6)
    save(fig, "fig1_protocol")


# --------------------------------------------------------------- Figure 2
def fig_ablation():
    spec = [("Geometry (GNN)", "rupture_clean_geom", C_BLUE),
            ("Clinical + morphometry", "rupture_clean_tab", C_ORANGE),
            ("Imaging (3D CNN)", "rupture_clean_img", C_AQUA),
            ("Geometry + imaging", "rupture_clean_no_tab", C_GREY),
            ("Imaging + clinical", "rupture_clean_no_geom", C_GREY),
            ("Geometry + clinical", "rupture_clean_no_img", C_GREY),
            ("Attention fusion (all)", "rupture_clean_fusion", C_YELLOW)]
    rows = []
    for lab, key, c in spec:
        r = load(RES, key)
        if r:
            rows.append((lab, r["mean"]["auc"], r["auc_ci95"][0], r["auc_ci95"][1], c))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    forest(ax, rows)
    best = max(r[1] for r in rows)
    ax.axvline(best, ls="--", lw=1, color=C_AQUA, alpha=.55, zorder=0)
    ax.set_xlim(0.30, 1.02)
    ax.set_title("Corrected protocol: no fusion beats the best single modality",
                 fontsize=9.5, loc="left", color=INK, pad=8)
    save(fig, "fig2_ablation")


# --------------------------------------------------------------- Figure 3
def roc_from(path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    y, s = d["y"].astype(int), d["oof"]
    o = np.argsort(-s)
    y = y[o]
    tpr = np.r_[0, np.cumsum(y) / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - y) / max((1 - y).sum(), 1)]
    return fpr, tpr


def fig_roc_cal():
    series = [("Imaging", "rupture_clean_img", C_AQUA),
              ("Attention fusion", "rupture_clean_fusion", C_YELLOW),
              ("Clinical + morph.", "rupture_clean_tab", C_ORANGE),
              ("Geometry", "rupture_clean_geom", C_BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    any_roc = False
    for lab, key, c in series:
        cur = roc_from(RES / ("oof_" + key + ".npz"))
        r = load(RES, key)
        if cur is None or r is None:
            continue
        axes[0].plot(cur[0], cur[1], lw=2, color=c,
                     label="%s (%.3f)" % (lab, r["mean"]["auc"]))
        any_roc = True
    if not any_roc:
        plt.close(fig)
        return
    axes[0].plot([0, 1], [0, 1], ":", lw=1, color=C_GREY)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC, corrected protocol", fontsize=9, loc="left")
    axes[0].legend(fontsize=7, loc="lower right")

    r = load(RES, "rupture_clean_img")
    cal = (r or {}).get("calibration") or []
    if cal:
        axes[1].plot([0, 1], [0, 1], ":", lw=1, color=C_GREY, label="ideal")
        axes[1].plot([c[0] for c in cal], [c[1] for c in cal], "o-",
                     color=C_AQUA, lw=2, ms=6, mec="white", mew=1.2,
                     label="imaging")
        axes[1].set_xlabel("Mean predicted probability")
        axes[1].set_ylabel("Observed rupture fraction")
        axes[1].set_title("Calibration (Brier %.3f)" % r["mean"]["brier"],
                          fontsize=9, loc="left")
        axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_roc_calibration")


# --------------------------------------------------------------- Figure 4
def fig_noise_weight():
    """THE fusion result: can an operator down-weight a modality that carries
    no information at all?"""
    spec = [("Attention", "noise_attn"), ("Concat*", "noise_concat"),
            ("Late mean*", "noise_late_mean"), ("Precision", "noise_late_prec"),
            ("Precision +warmup", "noise_late_prec2"),
            ("Evidential", "noise_evidential")]
    rows = []
    for lab, key in spec:
        r = load(RESF, key)
        if r and "noise_weight" in r:
            rows.append((lab, r["noise_weight"], r.get("uniform_weight", .25)))
    st = load(RESF, "stack_l1_4mod")
    if st and "noise_weight" in st:
        rows.append(("Stacking (L1)", st["noise_weight"],
                     st.get("uniform_weight", .25)))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.0, 2.35))
    xs = np.arange(len(rows))
    unif = rows[0][2]
    cols = [C_GREY if "*" in r[0] else
            (C_AQUA if "Stacking" in r[0] else C_BLUE) for r in rows]
    ax.bar(xs, [r[1] for r in rows], width=.62, color=cols)
    for x, r in zip(xs, rows):
        ax.text(x, r[1] + .006, "%.3f" % r[1], ha="center", fontsize=7.8, color=INK)
    ax.axhline(unif, ls="--", lw=1.2, color=C_RED)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.6, rotation=18, ha="right")
    ax.set_ylabel("weight given to a pure-noise modality")
    # Fix the limit BEFORE placing the annotation: reading get_ylim() first gives
    # the autoscaled value, which put the label straight through the bar tops.
    ymax = max(max(r[1] for r in rows), unif) * 1.42
    ax.set_ylim(0, ymax)
    ax.annotate("uniform share (%.2f)" % unif, xy=(len(rows) - 0.5, unif),
                xytext=(len(rows) - 0.52, ymax * 0.93),
                fontsize=7.2, color=C_RED, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=C_RED, lw=0.8,
                                shrinkA=0, shrinkB=2))
    ax.grid(axis="x", visible=False)
    ax.set_title("Can the operator ignore an uninformative modality?",
                 fontsize=9.5, loc="left", pad=8)
    fig.tight_layout()
    fig.text(0.01, -0.02, "* weights fixed by construction, shown for reference",
             fontsize=7, color=MUTED)
    save(fig, "fig4_noise_weight")


# --------------------------------------------------------------- Figure 5
def fig_operators():
    spec = [("Imaging only", "img_only", C_AQUA),
            ("Attention", "fuse_attn", C_BLUE),
            ("Concat", "fuse_concat", C_BLUE),
            ("Late mean", "fuse_late_mean", C_BLUE),
            ("Precision", "fuse_late_prec", C_BLUE),
            ("Precision +warmup", "fuse_late_prec2", C_BLUE),
            ("Evidential", "fuse_evidential", C_BLUE),
            ("Stacking (L1)", "stack_l1_3mod", C_ORANGE)]
    rows = []
    for lab, key, c in spec:
        r = load(RESF, key)
        if r:
            rows.append((lab, r["auc_mean"], r["auc_ci95"][0], r["auc_ci95"][1], c))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    forest(ax, rows)
    ref = next((r[1] for r in rows if r[0] == "Imaging only"), None)
    if ref:
        ax.axvline(ref, ls="--", lw=1, color=C_AQUA, alpha=.55, zorder=0)
    ax.set_xlim(0.30, 1.02)
    ax.set_title("Fusion operators vs the best single modality",
                 fontsize=9.5, loc="left", pad=8)
    save(fig, "fig5_operators")


# --------------------------------------------------------------- Figure 6
def fig_stacking_coef():
    r = load(RESF, "stack_l1_4mod")
    if not r:
        return
    mods = list(r["meta_coef_mean"].keys())
    vals = [r["meta_coef_mean"][m] for m in mods]
    zero = [r.get("zero_coef_rate", {}).get(m, 0.0) for m in mods]
    fig, ax = plt.subplots(figsize=(5.4, 2.05))
    ys = np.arange(len(mods))[::-1]
    cols = [C_RED if m == "noise" else C_BLUE for m in mods]
    ax.barh(ys, vals, height=.6, color=cols)
    for y, v, z, m in zip(ys, vals, zero, mods):
        ax.text(v + (0.02 if v >= 0 else -0.02), y,
                "%.3f   (zero in %.0f%% of folds)" % (v, 100 * z),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7.4, color=INK)
    ax.axvline(0, lw=1, color="#c9c9c4")
    ax.set_yticks(ys)
    ax.set_yticklabels([{"geom": "Geometry", "img": "Imaging", "tab": "Clinical",
                         "noise": "Pure noise"}.get(m, m) for m in mods], fontsize=8)
    ax.set_xlabel("mean L1 meta-learner coefficient")
    ax.grid(axis="y", visible=False)
    lim = max(abs(min(vals)), abs(max(vals))) * 2.4 + 1e-6
    ax.set_xlim(-lim, lim)
    ax.set_title("Stacking assigns weight from held-out evidence",
                 fontsize=9.5, loc="left", pad=8)
    save(fig, "fig6_stacking_coef")


# --------------------------------------------------------------- tables
def tables():
    def fmt(r, key="mean"):
        m, s = r[key], r.get("std", {})
        ci = r.get("auc_ci95", [float("nan")] * 2)
        return (m["auc"], s.get("auc", float("nan")), ci[0], ci[1],
                m["auprc"], m["sens"], m["spec"], m["f1"], m["mcc"], m["brier"])

    lines_md = ["| Model | n | AUC | 95% CI | AUPRC | Sens | Spec | F1 | MCC | Brier | p |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines_tex = [r"\begin{tabular}{lrrrrrrr}", r"\hline",
                 r"Model & $n$ & AUC & 95\% CI & AUPRC & Sens & Spec & MCC \\", r"\hline"]
    order = [("Geometry (GNN)", "rupture_clean_geom"),
             ("Clinical + morphometry", "rupture_clean_tab"),
             ("Imaging (3D CNN)", "rupture_clean_img"),
             ("Geometry + imaging", "rupture_clean_no_tab"),
             ("Imaging + clinical", "rupture_clean_no_geom"),
             ("Geometry + clinical", "rupture_clean_no_img"),
             ("Attention fusion", "rupture_clean_fusion"),
             ("Detection (fusion)", "leak_detection_fusion"),
             ("Detection (mesh only)", "leak_detection_geom"),
             ("Rupture + PHASES/ELAPSS", "leak_rupture_scores")]
    for lab, key in order:
        r = load(RES, key)
        if not r:
            continue
        a, sd, lo, hi, pr, se, sp, f1, mcc, br = fmt(r)
        p = r.get("permutation_p", float("nan"))
        lines_md.append("| %s | %d | %.3f ± %.3f | [%.3f, %.3f] | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.4f |"
                        % (lab, r["n"], a, sd, lo, hi, pr, se, sp, f1, mcc, br, p))
        lines_tex.append(r"%s & %d & %.3f $\pm$ %.3f & [%.3f, %.3f] & %.3f & %.3f & %.3f & %.3f \\"
                         % (lab, r["n"], a, sd, lo, hi, pr, se, sp, mcc))
    lines_tex += [r"\hline", r"\end{tabular}"]
    (FIG / "table_main.md").write_text("\n".join(lines_md))
    (FIG / "table_main.tex").write_text("\n".join(lines_tex))
    print("  wrote figs/table_main.md and .tex")

    # fusion-study table
    fl = ["| Operator | AUC | 95% CI | AUPRC | MCC | w(noise) |", "|---|---|---|---|---|---|"]
    for lab, key, nkey in [("Attention", "fuse_attn", "noise_attn"),
                           ("Concat", "fuse_concat", "noise_concat"),
                           ("Late mean", "fuse_late_mean", "noise_late_mean"),
                           ("Precision", "fuse_late_prec", "noise_late_prec"),
                           ("Precision +warmup", "fuse_late_prec2", "noise_late_prec2"),
                           ("Evidential", "fuse_evidential", "noise_evidential"),
                           ("Stacking (L1)", "stack_l1_3mod", "stack_l1_4mod")]:
        r, rn = load(RESF, key), load(RESF, nkey)
        if not r:
            continue
        w = ("%.3f" % rn["noise_weight"]) if rn and "noise_weight" in rn else "--"
        fl.append("| %s | %.3f ± %.3f | [%.3f, %.3f] | %.3f | %.3f | %s |"
                  % (lab, r["auc_mean"], r["auc_std"], r["auc_ci95"][0],
                     r["auc_ci95"][1], r["pooled"]["auprc"], r["pooled"]["mcc"], w))
    if len(fl) > 2:
        (FIG / "table_fusion.md").write_text("\n".join(fl))
        print("  wrote figs/table_fusion.md")


if __name__ == "__main__":
    fig_protocol(); fig_ablation(); fig_roc_cal()
    fig_noise_weight(); fig_operators(); fig_stacking_coef()
    tables()
    print("\ndone -> figs/")
