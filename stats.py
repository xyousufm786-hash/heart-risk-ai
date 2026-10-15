"""Statistical tests for comparing classifiers (small-sample safe)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(diffs, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold score differences.

    Plain paired t-tests over repeated-CV folds are over-confident because folds share
    training data; the correction inflates the variance by (1/k + n_test/n_train).
    Returns (t, two-sided p).
    """
    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on discordant pairs. Returns (b, c, p)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(y_true, y_score, metric, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for any metric(y_true, y_score)."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, s_a, s_b, metric, n_boot=2000, seed=0):
    """Paired bootstrap of metric(A) - metric(B). Returns (mean diff, lo, hi, p)."""
    rng = np.random.default_rng(seed)
    y_true, s_a, s_b = map(np.asarray, (y_true, s_a, s_b))
    n = len(y_true)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        d.append(metric(y_true[idx], s_a[idx]) - metric(y_true[idx], s_b[idx]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1))
