# Preprocessed tensors

Derivatives of CMHA (Song et al., *Sci. Data* 11:1213, 2024), CC BY 4.0.

| file | contents |
|---|---|
| `tabular.npz` | 105 aneurysms x (16 clinical + 27 morphometric); labels; patient groups; PHASES/ELAPSS held separately as baselines |
| `tabular_controls.npz`, `tabular_patients_detection.npz` | detection-task matrices, used only to quantify leakage |
| `meshes.npz`, `meshes_detection.npz` | 256 sampled surface points per case with normals, kNN adjacency |
| `*_meta.json`, `*_qc.json` | cohort composition and per-case preprocessing QC |

**Not included** (127 MB, regenerate locally):

| file | regenerate with |
|---|---|
| `volumes.npy` | `python src/prep_mesh_vol.py` |
| `volumes_detection.npy` | `python src/prep_detection_imaging.py` |

Both require the raw CMHA release in `data/raw/extracted/`. See the main README.
