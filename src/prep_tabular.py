"""
CMHA rupture-prediction task: build the tabular (clinical + morphometric) branch.

Task definition (corrected after CISCOM review):
    unit of analysis = ANEURYSM (n=105, from 99 patients)
    label            = Rupture (77 ruptured / 28 unruptured)
    healthy controls (n=44) are EXCLUDED -- they belong to a detection task, not this one.

Leakage controls applied here:
    * PHASES / ELAPSS are held out of the feature matrix and kept only as
      clinical comparison baselines (univariate AUC 0.857 / 0.913 -- these
      scores are defined for UNRUPTURED aneurysms and separate the classes
      almost deterministically in this cohort).
    * 'Ealier SAH from another aneurysm' is constant (all zero) -> dropped.
    * 'group' (patient id) is emitted so CV can be grouped: 6 patients each
      contribute one ruptured AND one unruptured aneurysm.
"""
import re, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw" / "statistical results"
OUT  = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LEAKY    = ["PHASE", "ELAPSS"]
CONSTANT = ["Ealier SAH from another aneurysm"]

CLIN_NUM = ["Age", "Systolic Pressure", "Diastolic Pressure",
            "Heart rate", "Respiratory Rate"]
CLIN_BIN = ["Smoking history", "Alcohol consumption history", "Diabetes history",
            "Hypertension history", "Family History", "Shape"]


def main():
    clin = pd.read_csv(RAW / "clinical_all.csv").iloc[1:]          # row 0 = SNOMED codes
    clin = clin[clin["number"].notna()].reset_index(drop=True)
    ane  = clin[clin["Has aneurysm"].astype(str) == "1"].reset_index(drop=True)
    assert len(ane) == 105, f"expected 105 aneurysm rows, got {len(ane)}"

    morph = pd.read_csv(RAW / "morphological_aneurysm_artery.csv")
    assert len(morph) == 105, f"expected 105 morphology rows, got {len(morph)}"

    # multi-aneurysm patients are '<id>_1' / '<id>_2' in morphology, plain '<id>' in clinical.
    # join positionally within each patient (order is consistent in both files).
    ane["base"]   = ane["number"].astype(str)
    morph["base"] = morph["number"].astype(str).str.replace(r"_\d+$", "", regex=True)
    ane["k"]      = ane.groupby("base").cumcount()
    morph["k"]    = morph.groupby("base").cumcount()

    df = ane.merge(morph, on=["base", "k"], how="inner", suffixes=("", "_m"))
    assert len(df) == 105, f"merge lost rows: {len(df)}"

    y      = df["Rupture"].astype(int).to_numpy()
    groups = df["base"].to_numpy()                 # patient id -> grouped CV
    aid    = df["number_m"].astype(str).to_numpy() # per-aneurysm id -> matches STL/NIfTI

    # ---- clinical block -------------------------------------------------
    C = pd.DataFrame(index=df.index)
    C["sex_male"] = (df["Gender"].astype(str).str.upper().str[0] == "M").astype(float)
    for c in CLIN_NUM + CLIN_BIN:
        C[c] = pd.to_numeric(df[c], errors="coerce")
    loc = df["location"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    C = pd.concat([C, pd.get_dummies(loc, prefix="loc").astype(float)], axis=1)
    C = C.fillna(C.median(numeric_only=True))

    # ---- morphometric block ---------------------------------------------
    mcols = [c for c in morph.columns if c not in ("number", "base", "k")]
    M = df[[c if c in df.columns else c + "_m" for c in mcols]].apply(
        pd.to_numeric, errors="coerce")
    M.columns = [re.sub(r"\s+", " ", c).strip() for c in mcols]
    M = M.fillna(M.median(numeric_only=True))

    base = pd.DataFrame({
        "phases": pd.to_numeric(df["PHASE"],  errors="coerce"),
        "elapss": pd.to_numeric(df["ELAPSS"], errors="coerce"),
    }).fillna(0.0)

    np.savez_compressed(
        OUT / "tabular.npz",
        X_clin=C.to_numpy(np.float32), clin_names=np.array(C.columns, dtype=object),
        X_morph=M.to_numpy(np.float32), morph_names=np.array(M.columns, dtype=object),
        baselines=base.to_numpy(np.float32),
        baseline_names=np.array(["phases", "elapss"], dtype=object),
        y=y.astype(np.int64), groups=groups.astype(object), aneurysm_id=aid.astype(object),
    )

    meta = {
        "n_aneurysms": int(len(df)), "n_patients": int(pd.unique(groups).size),
        "n_ruptured": int(y.sum()), "n_unruptured": int((1 - y).sum()),
        "n_clin_features": int(C.shape[1]), "n_morph_features": int(M.shape[1]),
        "excluded_leaky": LEAKY, "excluded_constant": CONSTANT,
        "multi_aneurysm_patients": sorted(
            pd.Series(groups).value_counts().loc[lambda s: s > 1].index.tolist()),
    }
    (OUT / "tabular_meta.json").write_text(json.dumps(meta, indent=2))
    df.assign(y=y, group=groups, aneurysm_id=aid).to_csv(OUT / "cohort.csv", index=False)

    print(json.dumps(meta, indent=2))
    print(f"\nclinical block  : {C.shape}")
    print(f"morphometry block: {M.shape}")
    print(f"-> {OUT/'tabular.npz'}")


if __name__ == "__main__":
    main()
