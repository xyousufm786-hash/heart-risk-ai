"""Helpers for the 'trustworthy AI' layer: calibration, conformal prediction, fairness, SHAP."""
from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .models import build_pipeline, model_zoo


def modal_params(folds_csv: pd.DataFrame, model: str) -> dict:
    """Most frequently selected hyper-parameters for `model` across the nested-CV outer folds."""
    cnt = Counter(folds_csv.loc[folds_csv.model == model, "best"])
    return ast.literal_eval(cnt.most_common(1)[0][0])


def make_model(model: str, numeric, categorical, params: dict):
    est, _ = model_zoo()[model]
    pipe = build_pipeline(clone(est), numeric, categorical)
    return pipe.set_params(**params)


def expected_calibration_error(y, p, n_bins: int = 8) -> float:
    """ECE with equal-frequency bins (robust for small n)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def reliability_points(y, p, n_bins: int = 8):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.array_split(np.argsort(p), n_bins)
    return np.array([p[b].mean() for b in bins]), np.array([y[b].mean() for b in bins])


def split_conformal_sets(p_cal, y_cal, p_test, alpha: float = 0.1):
    """Split conformal prediction with the LAC score s = 1 - p(true class).

    Returns a boolean array (n_test, 2): which classes are in each prediction set.
    Coverage >= 1-alpha holds marginally for exchangeable data.
    """
    p_cal = np.asarray(p_cal)
    s = 1.0 - np.where(np.asarray(y_cal) == 1, p_cal, 1.0 - p_cal)
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    p_test = np.asarray(p_test)
    # class c is kept when its score 1 - p(c) <= q, i.e. p(c) >= 1 - q
    return np.column_stack([(1.0 - p_test) >= 1.0 - q, p_test >= 1.0 - q])


def group_metrics(y, p, groups: dict[str, np.ndarray], thr: float = 0.5) -> pd.DataFrame:
    rows = []
    for g, mask in groups.items():
        yt, pp = y[mask], p[mask]
        pred = (pp >= thr).astype(int)
        tp, fn = ((pred == 1) & (yt == 1)).sum(), ((pred == 0) & (yt == 1)).sum()
        fp, tn = ((pred == 1) & (yt == 0)).sum(), ((pred == 0) & (yt == 0)).sum()
        rows.append(dict(group=g, n=int(mask.sum()), prevalence=yt.mean(),
                         auc=roc_auc_score(yt, pp) if len(np.unique(yt)) > 1 else np.nan,
                         TPR=tp / max(tp + fn, 1), FPR=fp / max(fp + tn, 1), selection_rate=pred.mean(),
                         accuracy=(pred == yt).mean()))
    return pd.DataFrame(rows)


def aggregate_to_original(names, values, categorical):
    """Sum per-one-hot-column attributions back to the original clinical feature."""
    out = {}
    for n, v in zip(names, values):
        base = n.split("__", 1)[1]
        for c in categorical:
            if base == c or base.startswith(c + "_"):
                base = c
                break
        out[base] = out.get(base, 0.0) + v
    return out
