"""Train and serialise the deployed model: logistic regression + conformal thresholds.

Conformal thresholds come from 10-fold out-of-fold predictions (cross-conformal style), so every
calibration score is computed by a model that never saw that patient.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sklearn  # noqa: E402

from hrisk.data import HEART_CATEGORICAL, HEART_NUMERIC, ROOT, load_heart  # noqa: E402
from hrisk.trust import make_model, modal_params  # noqa: E402

X, y = load_heart(); y = y.values
folds = pd.read_csv(ROOT / "results" / "benchmark_folds_heart.csv")
summ = pd.read_csv(ROOT / "results" / "summary_heart.csv").set_index("Model")
name = "Logistic Regression"
params = modal_params(folds, name)
pipe = make_model(name, HEART_NUMERIC, HEART_CATEGORICAL, params)

oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), method="predict_proba")[:, 1]
scores = 1.0 - np.where(y == 1, oof, 1.0 - oof)
n = len(scores)
q = {a: float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher")) for a in (0.05, 0.10, 0.20)}

pipe.fit(X, y)
meta = dict(model=f"Logistic Regression (L2, C={params['clf__C']})", n=int(n), prevalence=float(y.mean()),
            cv_auc=summ.loc[name, "auc"], cv_acc=summ.loc[name, "acc"], sklearn=sklearn.__version__,
            train_range={c: [float(X[c].min()), float(X[c].max())] for c in HEART_NUMERIC},
            params=str(params))
joblib.dump(dict(pipeline=pipe, conformal_q=q, meta=meta), ROOT / "models" / "heart_model.joblib")
print(json.dumps(dict(meta=meta, conformal_q=q), indent=2))
