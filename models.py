"""Model zoo: every classifier with a small, fixed hyper-parameter grid for nested CV."""
from __future__ import annotations

import warnings

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data import make_preprocessor

warnings.filterwarnings("ignore")

SEED = 42


def model_zoo(seed: int = SEED) -> dict[str, tuple[object, dict]]:
    """name -> (estimator, param_grid). Grid keys are prefixed with `clf__`."""
    return {
        "Majority baseline": (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=seed),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]},
        ),
        "k-NN": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 11, 21]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(max_iter=150, random_state=seed),
            {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3]},
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=seed,
                          n_jobs=1, verbosity=0),
            {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]},
        ),
        "LightGBM": (
            LGBMClassifier(n_estimators=150, min_child_samples=10, random_state=seed,
                           n_jobs=1, verbose=-1),
            {"clf__num_leaves": [4, 8], "clf__learning_rate": [0.03, 0.1]},
        ),
        "MLP (ANN)": (
            MLPClassifier(max_iter=2000, random_state=seed),
            {"clf__hidden_layer_sizes": [(8,), (16, 8)], "clf__alpha": [1e-3, 1e-1]},
        ),
    }


def build_pipeline(estimator, numeric, categorical) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("clf", estimator)])
