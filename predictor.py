"""Inference wrapper: probability + uncertainty (conformal set) + exact per-feature explanation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import HEART_CATEGORICAL, HEART_FEATURES
from .trust import aggregate_to_original

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "heart_model.joblib"


@dataclass
class Prediction:
    probability: float
    risk_band: str
    conformal_set: dict          # alpha -> list of classes in the prediction set
    contributions: pd.Series     # log-odds contribution per original feature (sums with intercept to logit)
    intercept: float
    out_of_range: list


def risk_band(p: float) -> str:
    return "Low" if p < 0.30 else "Moderate" if p < 0.60 else "High"


class HeartRiskModel:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle = joblib.load(path)
        self.pipe = self.bundle["pipeline"]
        self.meta = self.bundle["meta"]

    def predict(self, row: dict) -> Prediction:
        X = pd.DataFrame([row])[HEART_FEATURES]
        p = float(self.pipe.predict_proba(X)[0, 1])
        # conformal sets (LAC score): class c kept if 1 - p(c) <= q_alpha
        sets = {}
        for alpha, q in self.bundle["conformal_q"].items():
            keep = [c for c, pc in ((0, 1 - p), (1, p)) if 1 - pc <= q]
            sets[alpha] = keep
        # exact additive explanation for logistic regression: coef_j * z_j per transformed column
        prep, clf = self.pipe.named_steps["prep"], self.pipe.named_steps["clf"]
        z = prep.transform(X)[0]
        names = prep.get_feature_names_out()
        contrib = aggregate_to_original(names, clf.coef_[0] * z, HEART_CATEGORICAL)
        # flag inputs outside the range seen in training
        rng = self.meta["train_range"]
        oor = [f for f in rng if not (rng[f][0] <= row[f] <= rng[f][1])]
        return Prediction(p, risk_band(p), sets, pd.Series(contrib).sort_values(key=abs, ascending=False),
                          float(clf.intercept_[0]), oor)
