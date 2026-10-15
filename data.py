"""Dataset loading, cleaning and the shared preprocessing pipeline.

Primary dataset : Cleveland Heart Disease (UCI), 303 patients, 13 clinical features.
Secondary       : Pima Indians Diabetes (NIDDK), 768 patients, 8 features (generalisation check).

Data-integrity notes (verified empirically in scripts/00_data_audit.py):
  * In the widely circulated CSV mirror the `target` column is INVERTED w.r.t. the clinical
    meaning (target=1 means NO disease). We define  disease = 1 - target.
  * `ca == 4` (5 rows) and `thal == 0` (2 rows) are placeholder codes for missing values.
  * One exact duplicate row exists and is removed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

HEART_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
HEART_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
HEART_FEATURES = HEART_NUMERIC + HEART_CATEGORICAL

# Human-readable descriptions (used by the app and the report). Codes follow the CSV mirror.
HEART_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest-pain type (0 = asymptomatic, 1 = atypical angina, 2 = non-anginal, 3 = typical angina)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = yes)",
    "restecg": "Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise vs rest (mm)",
    "slope": "Slope of peak-exercise ST segment (0 = downsloping, 1 = flat, 2 = upsloping)",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium stress test (1 = fixed defect, 2 = normal, 3 = reversible defect)",
}

PIMA_COLUMNS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
                "bmi", "pedigree", "age", "diabetes"]
PIMA_ZERO_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_heart(clean: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for Cleveland. y = 1 means heart disease present."""
    df = pd.read_csv(RAW / "cleveland_kb22.csv", encoding="utf-8-sig")
    if clean:
        df = df.drop_duplicates().reset_index(drop=True)
        df.loc[df["ca"] == 4, "ca"] = np.nan
        df.loc[df["thal"] == 0, "thal"] = np.nan
    y = (1 - df["target"]).astype(int).rename("disease")
    return df[HEART_FEATURES].copy(), y


def load_pima() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW / "pima.csv", header=None, names=PIMA_COLUMNS)
    df[PIMA_ZERO_MISSING] = df[PIMA_ZERO_MISSING].replace(0, np.nan)
    return df.drop(columns="diabetes"), df["diabetes"].astype(int)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median-impute + standardise numerics; mode-impute + one-hot categoricals.

    Fitted inside every CV fold (via Pipeline), so no information leaks from test folds.
    """
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    transformers = [("num", num, numeric)]
    if categorical:
        cat = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers)


def get_dataset(name: str):
    """Return (X, y, numeric_cols, categorical_cols) for 'heart' or 'pima'."""
    if name == "heart":
        X, y = load_heart()
        return X, y, HEART_NUMERIC, HEART_CATEGORICAL
    if name == "pima":
        X, y = load_pima()
        return X, y, list(X.columns), []
    raise ValueError(name)
