import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrisk.data import HEART_FEATURES, load_heart, load_pima  # noqa: E402
from hrisk.federated import dirichlet_partition, federated_train, predict_proba  # noqa: E402
from hrisk.predictor import HeartRiskModel  # noqa: E402
from hrisk.stats import corrected_resampled_ttest, holm, mcnemar_exact  # noqa: E402
from hrisk.trust import expected_calibration_error, split_conformal_sets  # noqa: E402

ROW = dict(age=56, sex=1, cp=1, trestbps=135, chol=250, fbs=0, restecg=1, thalach=140,
           exang=0, oldpeak=1.2, slope=1, ca=1, thal=2)


def test_heart_loader_shape_and_label_direction():
    X, y = load_heart()
    assert X.shape == (302, 13) and list(X.columns) == HEART_FEATURES
    # clinically sensible direction after the label fix: exercise angina is MORE common in disease
    assert X.loc[y == 1, "exang"].mean() > X.loc[y == 0, "exang"].mean()
    assert X.loc[y == 1, "thalach"].mean() < X.loc[y == 0, "thalach"].mean()
    assert X["ca"].max() <= 3 and (X["thal"].dropna() > 0).all()


def test_pima_zero_to_nan():
    X, _ = load_pima()
    assert X.shape == (768, 8) and X["glucose"].min() > 0


def test_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(200):
        p = rng.uniform(0, 1, 600)
        y = (rng.uniform(0, 1, 600) < p).astype(int)      # perfectly calibrated scores
        sets = split_conformal_sets(p[:300], y[:300], p[300:], alpha=0.1)
        cover.append(sets[np.arange(300), y[300:]].mean())
    assert 0.88 < np.mean(cover) < 0.94


def test_ece_perfect_calibration_is_small():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_holm_and_mcnemar_and_nb_test():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    b, c, p = mcnemar_exact([1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0])
    assert (b, c) == (2, 1) and p == pytest.approx(1.0)
    t, p = corrected_resampled_ttest(np.full(10, 0.01) + np.random.default_rng(0).normal(0, 1e-3, 10), 240, 60)
    assert p < 0.05


def test_dirichlet_partition_covers_all_samples():
    y = np.r_[np.zeros(120), np.ones(100)]
    parts = dirichlet_partition(y, 5, 0.3, np.random.default_rng(0))
    assert sorted(np.concatenate(parts)) == list(range(len(y)))


def test_federated_learns_separable_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    parts = dirichlet_partition(y, 4, 1.0, rng)
    w = federated_train([(X[p], y[p]) for p in parts], 3, mu=0.1, rounds=30)
    assert ((predict_proba(w, X) > 0.5) == y).mean() > 0.9


def test_deployed_model_contributions_sum_to_logit():
    m = HeartRiskModel()
    pr = m.predict(ROW)
    logit = pr.intercept + pr.contributions.sum()
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(pr.probability, abs=1e-9)
    assert all(len(s) in (0, 1, 2) for s in pr.conformal_set.values())


def test_out_of_range_flag():
    assert "age" in HeartRiskModel().predict({**ROW, "age": 120}).out_of_range
