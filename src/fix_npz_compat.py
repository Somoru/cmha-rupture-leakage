"""Make the .npz files loadable by numpy 1.x (the NGC container ships 1.x).

We saved object-dtype arrays ('groups', 'aneurysm_id'). Object arrays are
pickled, and a pickle written by numpy 2.x references numpy._core, which does
not exist in 1.x:

    ModuleNotFoundError: No module named 'numpy._core'

Fixed-width unicode ('<U..') arrays are stored as plain data with no pickle at
all, so they load under either major version. Re-saves in place; numeric arrays
are untouched.
"""
import numpy as np
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
TARGETS = ["tabular.npz", "tabular_controls.npz",
           "tabular_patients_detection.npz", "meshes.npz",
           "meshes_detection.npz"]


def main():
    for name in TARGETS:
        p = PROC / name
        if not p.exists():
            print("  ! missing", name)
            continue
        with np.load(p, allow_pickle=True) as z:
            out, converted = {}, []
            for k in z.files:
                a = z[k]
                if a.dtype == object:
                    out[k] = a.astype(str)          # object -> '<U..', no pickle
                    converted.append("%s (%s)" % (k, out[k].dtype))
                else:
                    out[k] = a
        np.savez_compressed(p, **out)
        print("  %-34s converted: %s" % (name, ", ".join(converted) or "none"))

    print("\nverifying every file loads WITHOUT allow_pickle:")
    for name in TARGETS:
        p = PROC / name
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            shapes = {k: z[k].shape for k in z.files}
        print("  OK %-31s %s" % (name, shapes))


if __name__ == "__main__":
    main()
