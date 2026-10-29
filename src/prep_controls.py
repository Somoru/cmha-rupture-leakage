"""Build the DETECTION-task feature matrices (patients vs healthy controls).

These exist only to QUANTIFY the leak that invalidated the original submission --
they are not part of the corrected rupture model.

The structural problem, stated precisely:
  morphological_aneurysm_artery.csv has 27 features per aneurysm.
  morphological_control.csv        has  6 features per control.
  23 of the 27 are aneurysm-specific (neck width, aneurysm volume, AR, SR,
  ellipticity, undulation, ...) and are UNDEFINED for a subject with no aneurysm.
Therefore any joint patient+control design matrix must impute those 23 columns for
every control, and the imputed constant is itself a perfect class indicator. The
leak is structural, not incidental: no imputation choice avoids it.

Three variants are emitted so the paper can separate the effects:
  padded  : 16 clinical + 27 morphology, controls zero-imputed   -> the leak
  clinical: 16 clinical only                                     -> fair-ish
  shared  : 16 clinical + 4 genuinely shared vessel measurements  -> fair
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "statistical results"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CLIN_NUM = ["Age", "Systolic Pressure", "Diastolic Pressure", "Heart rate", "Respiratory Rate"]
CLIN_BIN = ["Smoking history", "Alcohol consumption history", "Diabetes history",
            "Hypertension history", "Family History"]
SHARED = ["The thicker branch diameter (D_1)", "The smaller branch diameter (D_2)"]


def norm(c):
    return re.sub(r"\s+", " ", str(c)).strip()


def clinical_block(df):
    """Clinical features defined identically for patients and controls.

    'Shape' and 'location' are deliberately EXCLUDED here: both describe an
    aneurysm and are undefined for a control, so including them would add a
    second structural leak on top of the one being measured.
    """
    C = pd.DataFrame(index=df.index)
    C["sex_male"] = (df["Gender"].astype(str).str.upper().str[0] == "M").astype(float)
    for c in CLIN_NUM + CLIN_BIN:
        C[c] = pd.to_numeric(df[c], errors="coerce")
    return C


def main():
    clin = pd.read_csv(RAW / "clinical_all.csv").iloc[1:]
    clin = clin[clin["number"].notna()].reset_index(drop=True)
    pat = clin[clin["Has aneurysm"].astype(str) == "1"].reset_index(drop=True)
    ctl = clin[clin["Has aneurysm"].astype(str) != "1"].reset_index(drop=True)

    mp = pd.read_csv(RAW / "morphological_aneurysm_artery.csv")
    mc = pd.read_csv(RAW / "morphological_control.csv")
    mp.columns = [norm(c) for c in mp.columns]
    mc.columns = [norm(c) for c in mc.columns]
    mp["base"] = mp["number"].astype(str).str.replace(r"_\d+$", "", regex=True)
    mp["k"] = mp.groupby("base").cumcount()
    pat["base"] = pat["number"].astype(str)
    pat["k"] = pat.groupby("base").cumcount()
    P = pat.merge(mp, on=["base", "k"], how="inner", suffixes=("", "_m"))
    assert len(P) == 105, len(P)

    mcols = [c for c in mp.columns if c not in ("number", "base", "k")]
    n_ctl = len(ctl)

    Cp, Cc = clinical_block(P), clinical_block(ctl)
    med = pd.concat([Cp, Cc]).median(numeric_only=True)
    Cp, Cc = Cp.fillna(med), Cc.fillna(med)

    Mp = P[mcols].apply(pd.to_numeric, errors="coerce")
    Mp = Mp.fillna(Mp.median(numeric_only=True))

    # ---- variant 1: zero-imputed morphology for controls (the leak) ----------
    Mc_zero = pd.DataFrame(0.0, index=range(n_ctl), columns=mcols)
    padded_p = np.hstack([Cp.to_numpy(np.float32), Mp.to_numpy(np.float32)])
    padded_c = np.hstack([Cc.to_numpy(np.float32), Mc_zero.to_numpy(np.float32)])

    # ---- variant 3: only genuinely shared vessel measurements ----------------
    shared_ok = [c for c in SHARED if c in mp.columns and c in mc.columns]
    Sp = P[shared_ok].apply(pd.to_numeric, errors="coerce")
    Sc = mc[shared_ok].apply(pd.to_numeric, errors="coerce")
    smed = pd.concat([Sp, Sc]).median(numeric_only=True)
    shared_p = np.hstack([Cp.to_numpy(np.float32), Sp.fillna(smed).to_numpy(np.float32)])
    shared_c = np.hstack([Cc.to_numpy(np.float32), Sc.fillna(smed).to_numpy(np.float32)])

    np.savez_compressed(
        OUT / "tabular_controls.npz",
        X_clin_padded=padded_c.astype(np.float32),
        X_clin_only=Cc.to_numpy(np.float32),
        X_shared=shared_c.astype(np.float32),
        groups=ctl["number"].astype(str).to_numpy(dtype=object),
        aneurysm_id=ctl["number"].astype(str).to_numpy(dtype=object),
    )
    np.savez_compressed(
        OUT / "tabular_patients_detection.npz",
        X_clin_padded=padded_p.astype(np.float32),
        X_clin_only=Cp.to_numpy(np.float32),
        X_shared=shared_p.astype(np.float32),
        groups=P["base"].to_numpy(dtype=object),
        aneurysm_id=P["number_m"].astype(str).to_numpy(dtype=object),
    )

    meta = {
        "n_patients_aneurysms": int(len(P)), "n_controls": int(n_ctl),
        "n_morph_patient": len(mcols), "n_morph_control": int(mc.shape[1] - 1),
        "n_shared_vessel_features": len(shared_ok),
        "aneurysm_specific_undefined_for_controls":
            sorted(set(mcols) - set(mc.columns) - {"number"}),
        "note": "23 of 27 morphology columns are undefined for controls -> "
                "any joint design matrix leaks the label by construction.",
    }
    (OUT / "detection_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:900])
    print("\npadded  patients", padded_p.shape, " controls", padded_c.shape)
    print("clinical patients", Cp.shape, " controls", Cc.shape)
    print("shared  patients", shared_p.shape, " controls", shared_c.shape)


if __name__ == "__main__":
    main()
