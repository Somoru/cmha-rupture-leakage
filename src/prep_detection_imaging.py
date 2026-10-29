"""Build mesh/volume tensors for the DETECTION configs (patients + controls).

Exists only to quantify the artifact, not as a proposed model.

The structural point, now visible at the file level:
    patients/<ID>/3D_aneurysm_<ID>.stl        <- an aneurysm dome
    controls/<ID>/3D_healthy_artery_<ID>.stl  <- a healthy artery segment
There is NO aneurysm mesh for a control, and no aneurysm centre on which to
centre a 64^3 crop. Whatever object is substituted for the control -- here the
healthy-artery model, the only thing available -- the two classes are then
different anatomical structures. A "detector" trained on this is separating
dome-shaped surfaces from tubular ones. The choice is arbitrary, and the
arbitrariness is precisely the leak.
"""
import json
import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_mesh_vol import (mesh_graph, crop_volume, find_stl, LPS2RAS,
                           N_NODES, GRID, RAWP, PROC)

RAWC = ROOTC = Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted" / "controls"


def encode(stl_path, nii_path, rng):
    mesh = trimesh.load(stl_path, process=False)
    mesh.vertices = mesh.vertices * LPS2RAS
    img = nib.load(nii_path)
    x, ah, mk, c = mesh_graph(mesh, rng)
    vol, cf = crop_volume(img, c, np.asarray(mesh.vertices, np.float32))
    return x, ah, mk, vol, cf


def main():
    rng = np.random.default_rng(0)
    tab = np.load(PROC / "tabular.npz", allow_pickle=True)
    pids = tab["aneurysm_id"].astype(str)
    cids = sorted(p.name for p in RAWC.iterdir() if p.is_dir())
    n = len(pids) + len(cids)

    X = np.zeros((n, N_NODES, 6), np.float32)
    AH = np.zeros((n, N_NODES, N_NODES), np.float32)
    MK = np.zeros((n, N_NODES, N_NODES), np.float32)
    VOL = np.zeros((n, GRID, GRID, GRID), np.float16)
    qc = []

    for i, aid in enumerate(pids):
        base = aid.split("_")[0]
        d = RAWP / base
        x, ah, mk, vol, cf = encode(find_stl(d, aid),
                                    d / f"cta_images_head_{base}.nii.gz", rng)
        X[i], AH[i], MK[i], VOL[i] = x, ah, mk, vol
        qc.append({"id": aid, "group": "patient", "cf": round(cf, 3)})

    for j, cid in enumerate(cids):
        i = len(pids) + j
        d = RAWC / cid
        stls = sorted(d.glob("3D_healthy_artery_*.stl")) or sorted(d.glob("*.stl"))
        x, ah, mk, vol, cf = encode(stls[0], d / f"cta_images_head_{cid}.nii.gz", rng)
        X[i], AH[i], MK[i], VOL[i] = x, ah, mk, vol
        qc.append({"id": cid, "group": "control", "cf": round(cf, 3)})
        if (j + 1) % 10 == 0:
            print(f"  controls {j+1}/{len(cids)}", flush=True)

    np.savez_compressed(PROC / "meshes_detection.npz", x=X, Ahat=AH, mask=MK,
                        nmask=np.ones((n, N_NODES), np.float32),
                        present=np.ones(n, bool))
    np.save(PROC / "volumes_detection.npy", VOL)
    (PROC / "detection_imaging_qc.json").write_text(json.dumps(
        {"n_patients": len(pids), "n_controls": len(cids),
         "patient_mesh": "3D_aneurysm_*.stl", "control_mesh": "3D_healthy_artery_*.stl",
         "note": "different anatomical objects per class -- see module docstring",
         "per_case": qc}, indent=2))
    print(f"\npatients={len(pids)} controls={len(cids)} total={n}")


if __name__ == "__main__":
    main()
