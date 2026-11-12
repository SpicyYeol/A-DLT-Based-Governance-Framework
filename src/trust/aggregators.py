"""Robust federated-aggregation rules used as C3 baselines.

Each aggregator takes the per-round list of candidate *new global* weight
vectors proposed by the contributing agents and returns the combined
(coef, intercept). Candidates are the agents' post-local-training weights
(g + delta), matching how the FL loop builds them.

    contribs = [ {"coef": ndarray, "int": ndarray, "weight": int, "aid": int}, ... ]

Baselines:
  * fedavg          — sample-weighted mean (no robustness)
  * coordinate_median — element-wise median (Yin et al.)
  * trimmed_mean    — element-wise trimmed mean, trims `beta` each end
  * krum            — Blanchard et al. Krum (single selection)
  * multi_krum      — average of the `m` lowest-score candidates

These defend only against *update-geometry* manipulation; they are blind to the
identity / freshness / integrity threats the governance framework also covers.
"""

from __future__ import annotations

import numpy as np


def _flatten(c):
    return np.concatenate([c["coef"].ravel(), c["int"].ravel()])


def _unflatten(vec, ref):
    n_coef = ref["coef"].size
    coef = vec[:n_coef].reshape(ref["coef"].shape)
    intc = vec[n_coef:].reshape(ref["int"].shape)
    return coef, intc


def fedavg(contribs):
    w = np.asarray([c["weight"] for c in contribs], dtype=float)
    w /= w.sum()
    coef = np.tensordot(w, np.asarray([c["coef"] for c in contribs]), axes=(0, 0))
    intc = np.tensordot(w, np.asarray([c["int"] for c in contribs]), axes=(0, 0))
    return coef, intc


def coordinate_median(contribs):
    M = np.asarray([_flatten(c) for c in contribs])
    med = np.median(M, axis=0)
    return _unflatten(med, contribs[0])


def trimmed_mean(contribs, beta: float = 0.125):
    M = np.asarray([_flatten(c) for c in contribs])
    n = len(M)
    k = int(np.floor(beta * n))
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    Ms = np.sort(M, axis=0)
    trimmed = Ms[k:n - k] if k > 0 else Ms
    return _unflatten(trimmed.mean(axis=0), contribs[0])


def _krum_scores(M, f: int):
    n = len(M)
    # pairwise squared distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = float(np.sum((M[i] - M[j]) ** 2))
            d[i, j] = d[j, i] = s
    m = max(1, n - f - 2)   # neighbours to sum (excluding self)
    scores = np.array([np.sort(d[i])[1:1 + m].sum() for i in range(n)])
    return scores


def krum(contribs, f: int = 2):
    M = np.asarray([_flatten(c) for c in contribs])
    if len(M) <= f + 2:
        return fedavg(contribs)
    scores = _krum_scores(M, f)
    return _unflatten(M[int(np.argmin(scores))], contribs[0])


def multi_krum(contribs, f: int = 2, m: int | None = None):
    M = np.asarray([_flatten(c) for c in contribs])
    n = len(M)
    if n <= f + 2:
        return fedavg(contribs)
    scores = _krum_scores(M, f)
    m = m or max(1, n - f)
    chosen = np.argsort(scores)[:m]
    return _unflatten(M[chosen].mean(axis=0), contribs[0])


# name -> callable, for the comparison table
BASELINES = {
    "FedAvg": fedavg,
    "Coord-Median": coordinate_median,
    "Trimmed-Mean": trimmed_mean,
    "Krum": krum,
    "Multi-Krum": multi_krum,
}
