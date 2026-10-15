"""Data audit: proves the label inversion, missing-value placeholders and duplicates claimed in data.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrisk.data import RAW, ROOT, load_heart, load_pima  # noqa: E402

raw = pd.read_csv(RAW / "cleveland_kb22.csv", encoding="utf-8-sig")
alt = pd.read_csv(RAW / "cleveland_sharma.csv", encoding="utf-8-sig")
audit = {
    "raw_shape": list(raw.shape),
    "two_independent_mirrors_identical": bool(raw.equals(alt)),
    "duplicates": int(raw.duplicated().sum()),
    "ca_eq_4": int((raw.ca == 4).sum()),
    "thal_eq_0": int((raw.thal == 0).sum()),
    "target_counts_raw": raw.target.value_counts().to_dict(),
}
# Evidence that raw target==1 is the HEALTHY class: clinical risk markers are lower in target==1
g = raw.groupby("target")[["age", "thalach", "exang", "oldpeak", "ca"]].mean().round(3)
audit["group_means_by_raw_target"] = g.to_dict("index")
X, y = load_heart()
audit["clean_shape"] = list(X.shape)
audit["prevalence_after_flip"] = float(y.mean())
audit["missing_after_cleaning"] = X.isna().sum()[lambda s: s > 0].to_dict()
Xp, yp = load_pima()
audit["pima_shape"] = list(Xp.shape)
audit["pima_prevalence"] = float(yp.mean())
audit["pima_missing_after_zero_fix"] = Xp.isna().sum()[lambda s: s > 0].to_dict()

# descriptive statistics tables for the report
desc = X.describe().T.round(2)
desc.to_csv(ROOT / "results" / "heart_descriptives.csv")
pd.concat([X, y], axis=1).groupby("disease").mean(numeric_only=True).T.round(2).to_csv(ROOT / "results" / "heart_by_class.csv")
json.dump(audit, open(ROOT / "results" / "data_audit.json", "w"), indent=2, default=str)
print(json.dumps(audit, indent=2, default=str))
