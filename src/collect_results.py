"""Bundle everything worth bringing back from the cluster into one zip.

Run this on the DGX after the job finishes:

    cd ~/cmha && python src/collect_results.py

-> ~/cmha_RESULTS.zip   (small: JSON, figures, tables, logs -- no tensors)

Then from your side:
    scp <user>@<login-node>:~/cmha_RESULTS.zip .
"""
import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "cmha_RESULTS.zip"

# (directory, glob, keep?) -- oof_*.npz are a few KB each and let us recompute
# ROC curves and DeLong tests locally, so they are worth carrying.
SPEC = [
    ("results", "*.json", True),
    ("results", "*.md", True),
    ("results", "oof_*.npz", True),
    ("results_fusion", "*.json", True),
    ("results_fusion", "oof_*.npz", True),
    ("figs", "*.pdf", True),
    ("figs", "*.png", True),
    ("figs", "*.md", True),
    ("figs", "*.tex", True),
    ("logs", "*.out", True),
    ("logs", "*.err", True),
    ("data/processed", "*_qc.json", True),
    ("data/processed", "*_meta.json", True),
]


def main():
    added, total = [], 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for sub, pat, _ in SPEC:
            d = ROOT / sub
            if not d.exists():
                continue
            for f in sorted(d.glob(pat)):
                if f.is_file():
                    z.write(f, str(Path(sub) / f.name))
                    added.append(str(Path(sub) / f.name))
                    total += f.stat().st_size

        manifest = {
            "collected_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_files": len(added),
            "uncompressed_bytes": total,
            "files": added,
        }
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        z.writestr("README.txt",
                   "CMHA results bundle\n"
                   "===================\n\n"
                   "results/          main 11-config suite (JSON + OOF predictions)\n"
                   "results_fusion/   fusion-operator study and stacking\n"
                   "figs/             figures (PDF for LaTeX, PNG preview) + tables\n"
                   "logs/             SLURM stdout/stderr, incl. the GPU probe table\n"
                   "data/processed/   QC summaries only -- no tensors\n\n"
                   "oof_*.npz hold out-of-fold predictions, so ROC curves,\n"
                   "DeLong tests and calibration can all be recomputed locally\n"
                   "without the cluster.\n")

    print("wrote %s" % OUT)
    print("  %d files, %.1f KB compressed (%.1f KB raw)"
          % (len(added), OUT.stat().st_size / 1024, total / 1024))
    missing = [s for s, _, _ in SPEC if not (ROOT / s).exists()]
    if missing:
        print("  note: absent directories skipped: %s" % sorted(set(missing)))
    print("\nfetch with:\n  scp <user>@<login-node>:~/cmha_RESULTS.zip .")


if __name__ == "__main__":
    main()
