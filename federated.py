"""Federated learning simulation (FedAvg vs FedProx) with a NumPy logistic-regression model.

Why NumPy: with ~240 training rows a linear model is the realistic federated candidate, and
writing the algorithm out makes every step explainable in the viva.

Local objective at client k (FedProx):  BCE_k(w) + (mu/2) * ||w - w_global||^2
mu = 0 recovers FedAvg.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def dirichlet_partition(y, n_clients: int, alpha: float, rng, min_size: int = 10):
    """Label-skew partition (Hsu et al. 2019). alpha -> inf is IID; small alpha is extreme skew."""
    y = np.asarray(y)
    for _ in range(200):
        parts = [[] for _ in range(n_clients)]
        for cls in np.unique(y):
            idx = rng.permutation(np.where(y == cls)[0])
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]
    raise RuntimeError("could not build a valid partition; raise alpha or lower min_size")


def local_train(w, X, y, mu, w_global, epochs, lr, batch, rng, l2=1e-3):
    """Mini-batch SGD on one client. w includes the bias as the last element."""
    w = w.copy()
    n = len(y)
    Xb = np.hstack([X, np.ones((n, 1))])
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            i = order[s:s + batch]
            p = _sigmoid(Xb[i] @ w)
            g = Xb[i].T @ (p - y[i]) / len(i) + l2 * np.r_[w[:-1], 0.0] + mu * (w - w_global)
            w -= lr * g
    return w


def federated_train(client_data, d, mu=0.0, rounds=40, epochs=5, lr=0.1, batch=16, seed=0,
                    return_history_on=None):
    """Run FedAvg (mu=0) or FedProx (mu>0). client_data = [(X_k, y_k), ...]. Returns weights."""
    rng = np.random.default_rng(seed)
    w = np.zeros(d + 1)
    sizes = np.array([len(c[1]) for c in client_data], float)
    hist = []
    for _ in range(rounds):
        locals_ = [local_train(w, X, y, mu, w, epochs, lr, batch, rng) for X, y in client_data]
        w = np.average(locals_, axis=0, weights=sizes)  # size-weighted aggregation
        if return_history_on is not None:
            hist.append(return_history_on(w))
    return (w, hist) if return_history_on is not None else w


def central_train(X, y, epochs=200, lr=0.1, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    return local_train(np.zeros(X.shape[1] + 1), X, y, 0.0, np.zeros(X.shape[1] + 1), epochs, lr, batch, rng)


def predict_proba(w, X):
    return _sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)
