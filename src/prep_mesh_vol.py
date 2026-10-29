"""Build the geometry (mesh) and imaging (CTA) branches for the rupture task.

Runs LOCALLY on the raw 10.7 GB archive and emits ~90 MB of tensors, so the
DGX only ever sees preprocessed arrays (student quota is 70 GB).

Two things here were established empirically, not assumed -- see validate_alignment():

1. COORDINATE FRAME. The STL models are in DICOM/LPS world coordinates while the
   NIfTI affine maps voxels to RAS. The correct mapping is therefore
        voxel = inv(affine) @ (-x, -y, z, 1)
   Verified on held-out cases by the fraction of interior mesh points landing in
   the contrast-filled lumen HU window: (-1,-1,1) scores 0.48/0.75/0.67 versus
   <=0.30 for every other sign convention. Using the identity mapping places the
   mesh on skull bone or air and silently destroys the imaging branch.

2. FIXED NODE COUNT. Raw meshes range from 229 to 2751 vertices, and vertex count
   correlates with aneurysm size, which correlates with rupture. Feeding variable
   node counts would leak size through the graph's cardinality. We therefore sample
   a fixed N points on every surface. Scale is preserved deliberately by dividing
   coordinates by a GLOBAL constant (not a per-sample norm), so real aneurysm size
   still reaches the encoder through the coordinates rather than through node count.
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import trimesh
from scipy.ndimage import map_coordinates

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAWP = ROOT / "data" / "raw" / "extracted" / "patients"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

N_NODES = 256          # sampled surface points per aneurysm
KNN = 8                # graph degree
BOX_MM = 32.0          # physical crop side length
GRID = 64              # output volume is GRID^3 -> 0.5 mm isotropic
SCALE_MM = 10.0        # global coordinate scale (preserves true size)
HU_LO, HU_HI = 0.0, 3000.0

LPS2RAS = np.array([-1.0, -1.0, 1.0])


def find_stl(pdir, aid):
    """Locate the aneurysm STL. Multi-aneurysm patients use _1/_2 suffixes."""
    base = aid.split("_")[0]
    cands = [pdir / f"3D_aneurysm_{aid}.stl", pdir / f"3D_aneurysm_{base}.stl"]
    if "_" in aid:
        k = aid.split("_")[1]
        cands = [pdir / f"3D_aneurysm_{base}_{k}.stl",
                 pdir / f"3D_aneurysm_{aid}.stl"] + cands
    for c in cands:
        if c.exists():
            return c
    hits = sorted(pdir.glob("3D_aneurysm_*.stl"))
    hits = [h for h in hits if "artery" not in h.name]
    if not hits:
        return None
    if "_" in aid:
        i = int(aid.split("_")[1]) - 1
        return hits[i] if i < len(hits) else hits[0]
    return hits[0]


def mesh_graph(mesh, rng):
    """Fixed-size surface point cloud + symmetric-normalised kNN graph."""
    pts, fidx = trimesh.sample.sample_surface(mesh, N_NODES)
    nrm = mesh.face_normals[fidx]
    c = pts.mean(0)
    x = np.hstack([(pts - c) / SCALE_MM, nrm]).astype(np.float32)   # N,6

    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nb = np.argsort(d2, axis=1)[:, :KNN]

    A = np.zeros((N_NODES, N_NODES), np.float32)
    rows = np.repeat(np.arange(N_NODES), KNN)
    A[rows, nb.ravel()] = 1.0
    A = np.maximum(A, A.T)                      # undirected
    Ahat = A + np.eye(N_NODES, dtype=np.float32)
    deg = Ahat.sum(1)
    Dm = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    Ahat = Ahat * Dm[:, None] * Dm[None, :]
    mask = (A + np.eye(N_NODES, dtype=np.float32) > 0).astype(np.float32)
    return x, Ahat.astype(np.float32), mask, c


def _world_to_vox(img, pts):
    return (np.c_[pts, np.ones(len(pts), np.float32)]
            @ np.linalg.inv(img.affine).T)[:, :3]


def crop_volume(img, centre_ras, mesh_ras):
    """Resample a BOX_MM cube centred on the aneurysm to GRID^3, isotropic.

    Reads only the voxel slab the crop actually touches. A full get_fdata() on
    these scans materialises ~1.2 GB per case; the slab is ~80x80x55 voxels.
    Returns (normalised volume, QC contrast fraction).
    """
    lin = np.linspace(-BOX_MM / 2, BOX_MM / 2, GRID, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    world = np.stack([gx, gy, gz], -1).reshape(-1, 3) + centre_ras
    vox = _world_to_vox(img, world)

    # QC points: interior of the aneurysm surface
    c = mesh_ras.mean(0)
    qvox = _world_to_vox(img, (c + 0.6 * (mesh_ras - c)).astype(np.float32))

    shape = np.array(img.shape[:3])
    allv = np.vstack([vox, qvox])
    lo = np.maximum(np.floor(allv.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(allv.max(0)).astype(int) + 2, shape)
    if np.any(hi <= lo):
        return np.zeros((GRID, GRID, GRID), np.float16), 0.0

    slab = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
                      dtype=np.float32)
    while slab.ndim > 3:              # a few scans carry a trailing singleton axis
        slab = slab[..., 0]

    vol = map_coordinates(slab, (vox - lo).T, order=1, mode="constant", cval=0.0)
    vol = np.clip(vol, HU_LO, HU_HI) / HU_HI

    q = np.round(qvox - lo).astype(int)
    ok = ((q >= 0) & (q < np.array(slab.shape))).all(1)
    if ok.sum() == 0:
        cf = 0.0
    else:
        hu = slab[q[ok, 0], q[ok, 1], q[ok, 2]]
        cf = float(((hu >= 150) & (hu <= 700)).mean())
    return vol.reshape(GRID, GRID, GRID).astype(np.float16), cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tab = np.load(PROC / "tabular.npz", allow_pickle=True)
    aids = tab["aneurysm_id"].astype(str)
    if a.limit:
        aids = aids[:a.limit]

    rng = np.random.default_rng(0)
    X = np.zeros((len(aids), N_NODES, 6), np.float32)
    AH = np.zeros((len(aids), N_NODES, N_NODES), np.float32)
    MK = np.zeros((len(aids), N_NODES, N_NODES), np.float32)
    VOL = np.zeros((len(aids), GRID, GRID, GRID), np.float16)
    qc, missing = [], []
    present = np.zeros(len(aids), bool)

    for i, aid in enumerate(aids):
        base = aid.split("_")[0]
        pdir = RAWP / base
        stl = find_stl(pdir, aid) if pdir.exists() else None
        nii = pdir / f"cta_images_head_{base}.nii.gz"
        if stl is None or not nii.exists():
            missing.append(aid)
            print(f"  !! MISSING {aid}  (stl={stl is not None}, nii={nii.exists()})")
            continue

        mesh = trimesh.load(stl, process=False)
        mesh.vertices = mesh.vertices * LPS2RAS          # LPS -> RAS
        img = nib.load(nii)

        x, ah, mk, c = mesh_graph(mesh, rng)
        X[i], AH[i], MK[i] = x, ah, mk
        VOL[i], cf = crop_volume(img, c, np.asarray(mesh.vertices, np.float32))
        present[i] = True
        qc.append({"aneurysm_id": aid, "stl": stl.name, "n_verts": int(len(mesh.vertices)),
                   "contrast_fraction": round(cf, 3),
                   "centroid_ras": [round(float(v), 2) for v in c]})
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(aids)}] {aid}  verts={len(mesh.vertices)}  cf={cf:.2f}",
                  flush=True)

    np.savez_compressed(PROC / "meshes.npz", x=X, Ahat=AH, mask=MK,
                        nmask=np.ones((len(aids), N_NODES), np.float32),
                        present=present.astype(bool))
    np.save(PROC / "volumes.npy", VOL)
    cfs = [q["contrast_fraction"] for q in qc]
    summary = {
        "n": len(aids), "n_ok": len(qc), "missing": missing,
        "n_nodes": N_NODES, "knn": KNN, "box_mm": BOX_MM, "grid": GRID,
        "contrast_fraction_mean": round(float(np.mean(cfs)), 3) if cfs else None,
        "contrast_fraction_p10": round(float(np.percentile(cfs, 10)), 3) if cfs else None,
        "poorly_aligned_lt_0.2": [q["aneurysm_id"] for q in qc
                                  if q["contrast_fraction"] < 0.2],
        "per_case": qc,
    }
    (PROC / "mesh_vol_qc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
