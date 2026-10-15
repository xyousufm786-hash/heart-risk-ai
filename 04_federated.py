"""Federated learning experiment: FedAvg vs FedProx under label-skew non-IID partitions.

5 simulated hospitals; Dirichlet(alpha) label skew; 30 random train/test splits per setting.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrisk.data import HEART_CATEGORICAL, HEART_NUMERIC, ROOT, load_heart, make_preprocessor  # noqa: E402
from hrisk.federated import central_train, dirichlet_partition, federated_train, local_train, predict_proba  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
X, y = load_heart(); y = y.values
SEEDS, ALPHAS, MUS, K = 30, [0.1, 0.3, 1.0, 100.0], [0.01, 0.1, 1.0], 5
rows, hist_store = [], {}
for seed in range(SEEDS):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    prep = make_preprocessor(HEART_NUMERIC, HEART_CATEGORICAL).fit(X.iloc[tr])
    Ztr, Zte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te]); ytr, yte = y[tr], y[te]
    d = Ztr.shape[1]
    ev = lambda w: roc_auc_score(yte, predict_proba(w, Zte))
    def log(method, alpha, w, extra=None):
        p = predict_proba(w, Zte)
        rows.append(dict(seed=seed, alpha=alpha, method=method, auc=roc_auc_score(yte, p), acc=accuracy_score(yte, p >= .5)))
    w_c = central_train(Ztr, ytr, seed=seed)
    for alpha in ALPHAS:
        rng = np.random.default_rng(1000 * seed + int(alpha * 10))
        parts = dirichlet_partition(ytr, K, alpha, rng)
        clients = [(Ztr[p], ytr[p]) for p in parts]
        log("Centralised (upper bound)", alpha, w_c)
        # local-only: each client trains alone for the same total local epochs; report the mean client model
        loc = [local_train(np.zeros(d + 1), X_, y_, 0, np.zeros(d + 1), 150, 0.1, 16, rng) for X_, y_ in clients]
        aucs = [roc_auc_score(yte, predict_proba(w, Zte)) for w in loc]
        accs = [accuracy_score(yte, predict_proba(w, Zte) >= .5) for w in loc]
        rows.append(dict(seed=seed, alpha=alpha, method="Local-only (mean of clients)", auc=np.mean(aucs), acc=np.mean(accs)))
        w, h = federated_train(clients, d, mu=0.0, seed=seed, return_history_on=ev); log("FedAvg", alpha, w)
        if seed == 0 or True: hist_store.setdefault((alpha, "FedAvg"), []).append(h)
        for mu in MUS:
            w, h = federated_train(clients, d, mu=mu, seed=seed, return_history_on=ev); log(f"FedProx mu={mu}", alpha, w)
            hist_store.setdefault((alpha, f"FedProx mu={mu}"), []).append(h)
df = pd.DataFrame(rows); df.to_csv(RES / "federated_runs.csv", index=False)
summ = df.groupby(["alpha", "method"])[["auc", "acc"]].agg(["mean", "std"]).round(4)
summ.to_csv(RES / "federated_summary.csv"); print(summ.to_string())

# paired Wilcoxon: each FedProx vs FedAvg at each alpha
tests = []
for alpha in ALPHAS:
    a = df[(df.alpha == alpha) & (df.method == "FedAvg")].sort_values("seed").auc.values
    for mu in MUS:
        b = df[(df.alpha == alpha) & (df.method == f"FedProx mu={mu}")].sort_values("seed").auc.values
        diff = b - a
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        tests.append(dict(alpha=alpha, mu=mu, mean_dAUC=diff.mean(), p_wilcoxon=p, wins=int((diff > 0).sum()), losses=int((diff < 0).sum())))
tests = pd.DataFrame(tests); tests.to_csv(RES / "federated_tests.csv", index=False); print(tests.round(4).to_string(index=False))

# figures
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for method, st in [("Centralised (upper bound)", "k--"), ("FedAvg", "o-"), ("FedProx mu=0.1", "s-"), ("Local-only (mean of clients)", "x:")]:
    g = df[df.method == method].groupby("alpha").auc
    ax.errorbar(range(len(ALPHAS)), g.mean(), g.std() / np.sqrt(SEEDS), fmt=st, capsize=3, ms=4, label=method)
ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels(["0.1\n(extreme)", "0.3", "1.0", "100\n(≈IID)"])
ax.set(xlabel="Dirichlet α (label skew across hospitals)", ylabel="Test ROC-AUC (mean ± s.e.)", title="Federated learning under non-IID data")
ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "federated_alpha.png", dpi=170); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m in ["FedAvg", "FedProx mu=0.1", "FedProx mu=1.0"]:
    h = np.array(hist_store[(0.1, m)]); ax.plot(h.mean(0), label=m)
ax.set(xlabel="Communication round", ylabel="Test ROC-AUC", title="Convergence at α = 0.1 (mean of 30 runs)")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "federated_convergence.png", dpi=170); plt.close(fig)
print("DONE")
