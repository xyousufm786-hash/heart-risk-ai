"""Trustworthy-AI layer on the Cleveland data: calibration, conformal prediction, SHAP + stability,
fairness audit, robustness / distribution shift, learning curves.

Run after 01_benchmark.py and 02_analysis.py:   python scripts/03_trust.py
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrisk.data import HEART_CATEGORICAL, HEART_NUMERIC, ROOT, load_heart  # noqa: E402
from hrisk.trust import (aggregate_to_original, expected_calibration_error, group_metrics, make_model,  # noqa: E402
                         modal_params, reliability_points, split_conformal_sets)
from hrisk.stats import bootstrap_ci  # noqa: E402

warnings.filterwarnings("ignore")
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
SEED = 42
X, y = load_heart()
y = y.values
folds_csv = pd.read_csv(RES / "benchmark_folds_heart.csv")
summ = pd.read_csv(RES / "summary_heart.csv")
ranked = [m for m in summ.Model if m != "Majority baseline"]
TREES = {"Random Forest", "XGBoost", "LightGBM", "HistGradientBoosting"}
best_tree = next(m for m in ranked if m in {"Random Forest", "XGBoost", "LightGBM"})
LR = "Logistic Regression"
P = {m: modal_params(folds_csv, m) for m in ranked}
mk = lambda m: make_model(m, HEART_NUMERIC, HEART_CATEGORICAL, P[m])
report = {"best_overall": ranked[0], "best_tree": best_tree, "params": {m: str(P[m]) for m in [ranked[0], best_tree, LR]}}
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))

# ================= 1. CALIBRATION ==========================================================
cal_models = list(dict.fromkeys([LR, ranked[0], best_tree, "SVM (RBF)"]))
cal_rows, curves = [], {}
for m in cal_models:
    oof = {k: np.zeros((5, len(y))) for k in ["raw", "platt", "isotonic"]}
    for i, (tr, te) in enumerate(cv):
        r = i // 5
        oof["raw"][r, te] = mk(m).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        for kind, method in [("platt", "sigmoid"), ("isotonic", "isotonic")]:
            c = CalibratedClassifierCV(mk(m), method=method, cv=3).fit(X.iloc[tr], y[tr])
            oof[kind][r, te] = c.predict_proba(X.iloc[te])[:, 1]
    for kind in oof:
        b = np.mean([brier_score_loss(y, oof[kind][r]) for r in range(5)])
        e = np.mean([expected_calibration_error(y, oof[kind][r]) for r in range(5)])
        a = np.mean([roc_auc_score(y, oof[kind][r]) for r in range(5)])
        cal_rows.append(dict(model=m, calibration=kind, brier=b, ece=e, auc=a))
        curves[(m, kind)] = reliability_points(y, oof[kind][0])
cal = pd.DataFrame(cal_rows)
cal.to_csv(RES / "calibration_heart.csv", index=False)
print(cal.round(4).to_string(index=False))
fig, axes = plt.subplots(1, len(cal_models), figsize=(3.6 * len(cal_models), 3.6), sharey=True)
for ax, m in zip(axes, cal_models):
    for kind, c in zip(["raw", "platt", "isotonic"], ["#d62728", "#1f77b4", "#2ca02c"]):
        px, py = curves[(m, kind)]
        ax.plot(px, py, "o-", color=c, ms=3, lw=1.2, label=kind)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(title=m, xlabel="Mean predicted probability")
axes[0].set_ylabel("Observed disease rate"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "calibration_heart.png", dpi=170); plt.close(fig)

# ================= 2. CONFORMAL PREDICTION =================================================
conf_rows = []
for m in [LR, ranked[0]] if ranked[0] != LR else [LR]:
    for alpha in (0.05, 0.10, 0.20):
        cov, size, single, acc_single = [], [], [], []
        for i, (tr, te) in enumerate(cv):
            tr_fit, tr_cal = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=i)
            model = mk(m).fit(X.iloc[tr_fit], y[tr_fit])
            sets = split_conformal_sets(model.predict_proba(X.iloc[tr_cal])[:, 1], y[tr_cal],
                                        model.predict_proba(X.iloc[te])[:, 1], alpha)
            covered = sets[np.arange(len(te)), y[te]]
            sz = sets.sum(1)
            cov.append(covered.mean()); size.append(sz.mean()); single.append((sz == 1).mean())
            sm = sz == 1
            if sm.any():
                acc_single.append((sets[sm][:, 1].astype(int) == y[te][sm]).mean())
        conf_rows.append(dict(model=m, alpha=alpha, target_coverage=1 - alpha, coverage=np.mean(cov),
                              coverage_std=np.std(cov), avg_set_size=np.mean(size),
                              singleton_frac=np.mean(single), acc_on_singletons=np.mean(acc_single)))
conf = pd.DataFrame(conf_rows)
conf.to_csv(RES / "conformal_heart.csv", index=False)
print(conf.round(3).to_string(index=False))

# ================= 3. SHAP + explanation validation ========================================
def feature_importance(model_name, Xtr, ytr, kind):
    """Global importance per ORIGINAL feature (mean |SHAP| for trees, |coef| for LR)."""
    pipe = mk(model_name).fit(Xtr, ytr)
    names = pipe.named_steps["prep"].get_feature_names_out()
    Z = pipe.named_steps["prep"].transform(Xtr)
    clf = pipe.named_steps["clf"]
    if kind == "shap":
        sv = shap.TreeExplainer(clf).shap_values(Z)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        vals = np.abs(sv).mean(0)
    else:
        vals = np.abs(clf.coef_[0])
    return pd.Series(aggregate_to_original(names, vals, HEART_CATEGORICAL))

imp = {f"{best_tree} (SHAP)": feature_importance(best_tree, X, y, "shap"),
       f"{LR} (|coef|)": feature_importance(LR, X, y, "coef")}
pipe_lr = mk(LR).fit(X, y)
pi = permutation_importance(pipe_lr, X, y, scoring="roc_auc", n_repeats=30, random_state=SEED)
imp[f"{LR} (permutation)"] = pd.Series(pi.importances_mean, index=X.columns)
imp_df = pd.DataFrame(imp).fillna(0)
imp_df = imp_df.div(imp_df.sum()).sort_values(imp_df.columns[0], ascending=False)
imp_df.to_csv(RES / "feature_importance_heart.csv")
agree = {f"{a} vs {b}": dict(spearman=spearmanr(imp_df[a], imp_df[b])[0], kendall=kendalltau(imp_df[a], imp_df[b])[0])
         for i, a in enumerate(imp_df.columns) for b in imp_df.columns[i + 1:]}
# stability: bootstrap resamples -> rank correlation with the full-data ranking
rng = np.random.default_rng(SEED)
stab = {k: [] for k in ["shap_tree", "coef_lr"]}
full_tree, full_lr = imp[f"{best_tree} (SHAP)"], imp[f"{LR} (|coef|)"]
for _ in range(40):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    Xb, yb = X.iloc[idx].reset_index(drop=True), y[idx]
    stab["shap_tree"].append(spearmanr(full_tree, feature_importance(best_tree, Xb, yb, "shap").reindex(full_tree.index).fillna(0))[0])
    stab["coef_lr"].append(spearmanr(full_lr, feature_importance(LR, Xb, yb, "coef").reindex(full_lr.index).fillna(0))[0])
report["explanation_agreement"] = agree
report["explanation_stability_spearman"] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), min=float(np.min(v))) for k, v in stab.items()}
print(imp_df.round(3)); print(json.dumps(report["explanation_agreement"], indent=1, default=float)); print(report["explanation_stability_spearman"])

# SHAP figures (beeswarm on transformed features)
pipe_t = mk(best_tree).fit(X, y)
Z = pipe_t.named_steps["prep"].transform(X)
names = [n.split("__", 1)[1] for n in pipe_t.named_steps["prep"].get_feature_names_out()]
sv = shap.TreeExplainer(pipe_t.named_steps["clf"]).shap_values(Z)
sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
plt.figure(); shap.summary_plot(sv, Z, feature_names=names, show=False, max_display=12)
plt.title(f"SHAP summary - {best_tree}", fontsize=10); plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm_heart.png", dpi=170, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 4.2))
imp_df.head(10)[::-1].plot.barh(ax=ax, width=0.8)
ax.set(xlabel="Normalised importance", title="Feature importance: three methods"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "importance_agreement_heart.png", dpi=170); plt.close(fig)

# ================= 4. FAIRNESS AUDIT =======================================================
oof = np.load(RES / "oof_heart.npz")
fair_all = []
for m in dict.fromkeys([ranked[0], LR]):
    p = oof[m.replace(" ", "_")][0]
    groups = {"Female": (X.sex == 0).values, "Male": (X.sex == 1).values,
              "Age < 55": (X.age < 55).values, "Age >= 55": (X.age >= 55).values}
    gm = group_metrics(y, p, groups); gm.insert(0, "model", m)
    # bootstrap CIs on the TPR gap and FPR gap between groups
    for a, b in [("Female", "Male"), ("Age < 55", "Age >= 55")]:
        ga, gb = groups[a], groups[b]
        gaps = []
        for _ in range(1000):
            ia, ib = rng.choice(np.where(ga)[0], ga.sum()), rng.choice(np.where(gb)[0], gb.sum())
            def tpr(i): return ((p[i] >= .5) & (y[i] == 1)).sum() / max((y[i] == 1).sum(), 1)
            def fpr(i): return ((p[i] >= .5) & (y[i] == 0)).sum() / max((y[i] == 0).sum(), 1)
            gaps.append((tpr(ia) - tpr(ib), fpr(ia) - fpr(ib)))
        gaps = np.array(gaps)
        report.setdefault("fairness_gaps", {})[f"{m}: {a} - {b}"] = dict(
            TPR_gap=float(np.mean(gaps[:, 0])), TPR_gap_CI=[float(x) for x in np.percentile(gaps[:, 0], [2.5, 97.5])],
            FPR_gap=float(np.mean(gaps[:, 1])), FPR_gap_CI=[float(x) for x in np.percentile(gaps[:, 1], [2.5, 97.5])])
    fair_all.append(gm)
fair = pd.concat(fair_all); fair.to_csv(RES / "fairness_heart.csv", index=False)
print(fair.round(3).to_string(index=False)); print(json.dumps(report["fairness_gaps"], indent=1))

# ================= 5. ROBUSTNESS & DISTRIBUTION SHIFT ======================================
rob_models = list(dict.fromkeys([LR, "SVM (RBF)", "Random Forest", ranked[0], best_tree]))
noise_rows = []
for m in rob_models:
    for sigma in (0.0, 0.25, 0.5, 1.0):
        aucs = []
        for i, (tr, te) in enumerate(cv[:10]):
            fit = mk(m).fit(X.iloc[tr], y[tr])
            Xt = X.iloc[te].copy()
            r = np.random.default_rng(i)
            for c in HEART_NUMERIC:
                Xt[c] = Xt[c] + r.normal(0, sigma * X[c].std(), len(Xt))
            aucs.append(roc_auc_score(y[te], fit.predict_proba(Xt)[:, 1]))
        noise_rows.append(dict(model=m, sigma=sigma, auc=np.mean(aucs), auc_std=np.std(aucs)))
noise = pd.DataFrame(noise_rows); noise.to_csv(RES / "robustness_noise_heart.csv", index=False)
shift_rows = []
splits = {"train Male -> test Female": ((X.sex == 1).values, (X.sex == 0).values),
          "train Female -> test Male": ((X.sex == 0).values, (X.sex == 1).values),
          "train Age<55 -> test Age>=55": ((X.age < 55).values, (X.age >= 55).values),
          "train Age>=55 -> test Age<55": ((X.age >= 55).values, (X.age < 55).values)}
for m in rob_models:
    for nm, (a, b) in splits.items():
        fit = mk(m).fit(X[a], y[a])
        # in-distribution reference: 5-fold CV inside the target group only
        ref = []
        Xb, yb = X[b].reset_index(drop=True), y[b]
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xb, yb):
            ref.append(roc_auc_score(yb[te], mk(m).fit(Xb.iloc[tr], yb[tr]).predict_proba(Xb.iloc[te])[:, 1]))
        shift_rows.append(dict(model=m, shift=nm, shifted_auc=roc_auc_score(yb, fit.predict_proba(Xb)[:, 1]), in_dist_auc=np.mean(ref)))
shift = pd.DataFrame(shift_rows); shift["drop"] = shift.in_dist_auc - shift.shifted_auc
shift.to_csv(RES / "distribution_shift_heart.csv", index=False)
print(noise.round(3).to_string(index=False)); print(shift.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for m in rob_models:
    g = noise[noise.model == m]; ax.errorbar(g.sigma, g.auc, g.auc_std, marker="o", ms=3, capsize=2, label=m, lw=1.2)
ax.set(xlabel="Gaussian noise (x feature std) on numeric inputs", ylabel="ROC-AUC", title="Robustness to measurement noise")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "robustness_noise_heart.png", dpi=170); plt.close(fig)

# ================= 6. LEARNING CURVES (small-data question) =================================
lc_rows = []
for m in dict.fromkeys([LR, "MLP (ANN)", best_tree, "SVM (RBF)"]):
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        aucs = []
        for i, (tr, te) in enumerate(cv[:15]):
            r = np.random.default_rng(i)
            sub = r.choice(tr, max(int(frac * len(tr)), 12), replace=False)
            if len(np.unique(y[sub])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], mk(m).fit(X.iloc[sub], y[sub]).predict_proba(X.iloc[te])[:, 1]))
        lc_rows.append(dict(model=m, frac=frac, n_train=int(frac * len(cv[0][0])), auc=np.mean(aucs), auc_std=np.std(aucs)))
lc = pd.DataFrame(lc_rows); lc.to_csv(RES / "learning_curve_heart.csv", index=False); print(lc.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for m in lc.model.unique():
    g = lc[lc.model == m]; ax.errorbar(g.n_train, g.auc, g.auc_std, marker="o", ms=3, capsize=2, label=m, lw=1.2)
ax.set(xlabel="Training samples", ylabel="ROC-AUC", title="Learning curves"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "learning_curve_heart.png", dpi=170); plt.close(fig)

json.dump(report, open(RES / "trust_report.json", "w"), indent=2, default=float)
print("DONE")
