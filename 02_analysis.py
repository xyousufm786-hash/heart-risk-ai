"""Turn raw benchmark output into tables + figures + significance tests.

Usage: python scripts/02_analysis.py heart
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrisk.data import ROOT  # noqa: E402
from hrisk.stats import bootstrap_ci, corrected_resampled_ttest, holm, mcnemar_exact, paired_bootstrap_diff  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "heart"
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
df = pd.read_csv(RES / f"benchmark_folds_{name}.csv")
meta = json.load(open(RES / f"benchmark_meta_{name}.json"))
oof = np.load(RES / f"oof_{name}.npz")
y = oof["y"]
models = list(df.model.unique())
key = lambda m: m.replace(" ", "_")

# ---- 1. summary table (mean +- std over outer folds) ----------------------------------
metrics = ["acc", "precision", "recall", "f1", "auc", "brier"]
rows = []
for m in models:
    g = df[df.model == m]
    r = {"Model": m}
    for k in metrics:
        r[k] = f"{g[k].mean():.3f} ± {g[k].std():.3f}"
    r["auc_mean"] = g["auc"].mean()
    r["Fit time (s)"] = round(g.fit_s.mean(), 2)
    r["Latency (ms/pt)"] = round(g.lat_ms.mean(), 3)
    # bootstrap CI of AUC on the pooled repeat-0 out-of-fold predictions
    p0 = oof[key(m)][0]
    lo, hi = bootstrap_ci(y, p0, roc_auc_score, 2000)
    r["AUC 95% CI (OOF)"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append(r)
summ = pd.DataFrame(rows).sort_values("auc_mean", ascending=False).drop(columns="auc_mean")
summ.to_csv(RES / f"summary_{name}.csv", index=False)
best = summ.iloc[0]["Model"]
print(summ.to_string(index=False))
print("\nBEST by AUC:", best)

# ---- 2. significance: corrected resampled t-test on per-fold AUC & accuracy ------------
def fold_scores(m, k):
    return df[df.model == m].sort_values("fold")[k].values

sig = []
for k in ["auc", "acc"]:
    base_pairs = [(best, m) for m in models if m != best]
    ps = []
    for a, b in base_pairs:
        d = fold_scores(a, k) - fold_scores(b, k)
        t, p = corrected_resampled_ttest(d, meta["n_train"], meta["n_test"])
        ps.append(p)
        sig.append(dict(metric=k, A=a, B=b, mean_diff=d.mean(), t=t, p=p))
    adj = holm(ps)
    for rec, a_ in zip(sig[-len(ps):], adj):
        rec["p_holm"] = a_
sig = pd.DataFrame(sig)
sig.to_csv(RES / f"significance_{name}.csv", index=False)
print("\nCorrected resampled t-test: best vs each (Holm-adjusted)")
print(sig.round(4).to_string(index=False))

# ---- 3. McNemar (exact) on pooled out-of-fold class predictions, repeat 0 ---------------
mc = []
for a, b in combinations([m for m in models if m != "Majority baseline"], 2):
    pa, pb = (oof[key(a)][0] >= .5).astype(int), (oof[key(b)][0] >= .5).astype(int)
    b_, c_, p = mcnemar_exact(y, pa, pb)
    mc.append(dict(A=a, B=b, A_only_correct=b_, B_only_correct=c_, p=p))
mc = pd.DataFrame(mc)
mc["p_holm"] = holm(mc.p.values)
mc.to_csv(RES / f"mcnemar_{name}.csv", index=False)
print("\nMcNemar pairs with raw p<0.05:", int((mc.p < .05).sum()), "of", len(mc), "| after Holm:", int((mc.p_holm < .05).sum()))

# ---- 4. paired bootstrap AUC difference vs logistic regression -------------------------
pb = []
for m in models:
    if m in ("Logistic Regression", "Majority baseline"):
        continue
    d, lo, hi, p = paired_bootstrap_diff(y, oof[key(m)][0], oof[key("Logistic Regression")][0], roc_auc_score)
    pb.append(dict(model=m, auc_diff_vs_LR=d, ci_lo=lo, ci_hi=hi, p=p))
pd.DataFrame(pb).to_csv(RES / f"bootstrap_vs_lr_{name}.csv", index=False)
print("\nPaired bootstrap dAUC vs LR:\n", pd.DataFrame(pb).round(4).to_string(index=False))

# ---- 5. figures -------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
order = summ.Model.tolist()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
data = [df[df.model == m].auc.values for m in order[::-1]]
ax.boxplot(data, vert=False, tick_labels=order[::-1], showmeans=True)
ax.set(xlabel="ROC-AUC across 25 outer folds", title=f"Model comparison ({name})")
ax.axvline(0.5, color="grey", ls=":")
fig.tight_layout(); fig.savefig(FIG / f"auc_boxplot_{name}.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 4.6))
for m in order:
    if m == "Majority baseline":
        continue
    fpr, tpr, _ = roc_curve(y, oof[key(m)][0])
    ax.plot(fpr, tpr, lw=1.6, label=f"{m} ({roc_auc_score(y, oof[key(m)][0]):.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC (pooled out-of-fold)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / f"roc_{name}.png", dpi=170); plt.close(fig)
