"""Draw the system architecture diagram used in the report and README."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "architecture.png"
fig, ax = plt.subplots(figsize=(11, 6.9)); ax.set_xlim(0, 110); ax.set_ylim(-5, 66); ax.axis("off")
C = {"data": "#dbeafe", "prep": "#fef3c7", "model": "#dcfce7", "trust": "#fce7f3", "app": "#ede9fe", "fed": "#ffedd5"}

def box(x, y, w, h, text, kind, fs=8.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=C[kind], ec="#334155", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(a, b, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, lw=1.2, color="#334155",
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))

ax.text(55, 64.2, "System architecture - trustworthy heart-disease risk prediction", ha="center", fontsize=12, fontweight="bold")
# row 1: data & prep
box(1, 46, 24, 14, "1  DATA SOURCES\nCleveland heart disease\n(n = 303, 13 features)\nPima diabetes (n = 768)\nsecond dataset", "data", fs=7.6)
box(29, 46, 24, 14, "2  DATA AUDIT\nfix inverted label\nca = 4, thal = 0 -> NaN\nremove duplicate\ncross-check 2 mirrors", "prep", fs=7.6)
box(57, 46, 24, 14, "3  PREPROCESSING\n(fitted inside each fold)\nmedian / mode impute\nstandardise\none-hot encode", "prep", fs=7.6)
box(85, 46, 24, 14, "4  EVALUATION\nnested CV: 5-fold x 5 rep.\ninner 3-fold grid search\nsame folds for all models\n-> paired tests", "data", fs=7.6)
for a_, b_ in [((25, 53), (29, 53)), ((53, 53), (57, 53)), ((81, 53), (85, 53))]:
    arrow(a_, b_)
# row 2: models
box(1, 27, 34, 13, "5  MODEL ZOO (9 models)\nbaseline - Logistic Regression - SVM - k-NN\nRandom Forest - HistGB - XGBoost\nLightGBM - MLP (ANN)", "model", fs=7.6)
box(38, 27, 34, 13, "6  BENCHMARK STATISTICS\nAUC, Acc, Prec, Rec, F1, Brier, latency\nNadeau-Bengio t - McNemar - bootstrap\nHolm correction", "model", fs=7.6)
box(75, 27, 34, 13, "7  FEDERATED SIMULATION\n5 hospitals, Dirichlet label skew\nFedAvg vs FedProx vs local-only\nvs centralised (30 seeds)", "fed", fs=7.6)
arrow((97, 46), (97, 40)); arrow((69, 46), (55, 40)); arrow((41, 46), (18, 40)); arrow((35, 33.5), (38, 33.5))
# row 3: trust layer
box(1, 8, 25, 14, "8a  CALIBRATION\nPlatt / isotonic\nBrier, ECE\nreliability curves", "trust", fs=7.6)
box(29, 8, 25, 14, "8b  UNCERTAINTY\nsplit-conformal sets\ncoverage 80 / 90 / 95 %\nabstain if ambiguous", "trust", fs=7.6)
box(57, 8, 25, 14, "8c  EXPLAINABILITY\nSHAP, |coef|, permutation\nagreement across methods\nbootstrap stability", "trust", fs=7.6)
box(85, 8, 24, 14, "8d  RESPONSIBLE AI\nsex / age subgroups\nnoise + distribution shift\nlearning curves", "trust", fs=7.6)
for x_ in (13.5, 41.5, 69.5):
    arrow((x_, 27), (x_, 22))
arrow((97, 27), (97, 22))
box(1, -3.5, 108, 7, "9  DEPLOYMENT:  Logistic Regression + conformal thresholds (joblib)  ->  Streamlit app\nprobability - risk band - confidence set - per-feature log-odds explanation", "app", fs=8, bold=True)
for x_ in (13.5, 41.5, 69.5, 97):
    arrow((x_, 8), (x_, 3.6))
fig.savefig(OUT, dpi=190, bbox_inches="tight"); print("saved")
