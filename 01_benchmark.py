"""Nested-CV benchmark of 9 models on a dataset, with identical folds for every model.

Outer loop : RepeatedStratifiedKFold (5 folds x R repeats)  -> unbiased performance estimate
Inner loop : 3-fold GridSearchCV on ROC-AUC                  -> hyper-parameter selection
Saves per-fold metrics + out-of-fold probabilities for significance tests, calibration etc.

Usage: python scripts/01_benchmark.py heart [repeats]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrisk.data import ROOT, get_dataset  # noqa: E402
from hrisk.models import SEED, build_pipeline, model_zoo  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "heart"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

X, y, num, cat = get_dataset(name)
y = y.values
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=SEED)
folds = list(splitter.split(X, y))
print(f"{name}: {X.shape}, prevalence={y.mean():.3f}, {len(folds)} outer folds", flush=True)


def run_fold(model_name, est, grid, fold_id, tr, te):
    pipe = build_pipeline(clone(est), num, cat)
    t0 = time.perf_counter()
    if grid:
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
        gs.fit(X.iloc[tr], y[tr])
        fitted, best = gs.best_estimator_, gs.best_params_
    else:
        fitted, best = pipe.fit(X.iloc[tr], y[tr]), {}
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    p = fitted.predict_proba(X.iloc[te])[:, 1]
    lat_ms = (time.perf_counter() - t1) / len(te) * 1000
    yt, yp = y[te], (p >= 0.5).astype(int)
    auc = roc_auc_score(yt, p) if len(np.unique(p)) > 1 else 0.5
    return dict(model=model_name, fold=fold_id, repeat=fold_id // 5, te=te.tolist(), p=p.tolist(),
                acc=accuracy_score(yt, yp), precision=precision_score(yt, yp, zero_division=0),
                recall=recall_score(yt, yp), f1=f1_score(yt, yp, zero_division=0), auc=auc,
                brier=brier_score_loss(yt, p), logloss=log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
                fit_s=fit_s, lat_ms=lat_ms, best=str(best))


jobs = [(m, e, g, i, tr, te) for m, (e, g) in model_zoo().items() for i, (tr, te) in enumerate(folds)]
t0 = time.time()
rows = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(*j) for j in jobs)
print(f"done in {time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.drop(columns=["te", "p"]).to_csv(OUT / f"benchmark_folds_{name}.csv", index=False)

# out-of-fold probability matrices: model -> [repeat, n]
oof = {}
for m, g in df.groupby("model"):
    mat = np.full((R, len(y)), np.nan)
    for _, r in g.iterrows():
        mat[r["repeat"], r["te"]] = r["p"]
    oof[m] = mat
np.savez_compressed(OUT / f"oof_{name}.npz", y=y, **{k.replace(" ", "_"): v for k, v in oof.items()})
json.dump({"n": int(len(y)), "n_train": int(len(folds[0][0])), "n_test": int(len(folds[0][1])), "repeats": R},
          open(OUT / f"benchmark_meta_{name}.json", "w"))

summ = df.groupby("model")[["acc", "precision", "recall", "f1", "auc", "brier", "fit_s", "lat_ms"]].agg(["mean", "std"])
print(summ.round(3).to_string())
