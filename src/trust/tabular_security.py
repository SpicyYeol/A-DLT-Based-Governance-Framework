"""Tabular network/IoT security branch (second open dataset).

Trains a lightweight anomaly/attack detector on KDDCup'99 (or the synthetic
telemetry fallback) to demonstrate transaction-level trust screening for the
satellite/IoT use case. This is the ML-oriented counterpart to the
DL-oriented federated-learning branch.

Output feeds the report as supplementary evidence and could gate MEDIUM /
Transaction Traceability in a production deployment.
"""

from __future__ import annotations

import numpy as np

from . import config
from .datasets import load_tabular_dataset


def run_detection() -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    ds = load_tabular_dataset()
    X, y = ds["X"], ds["y"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    # unsupervised anomaly detector (no attack labels needed at train time)
    iso = IsolationForest(n_estimators=100, random_state=config.RANDOM_SEED,
                          contamination="auto")
    iso.fit(X_tr_s)
    iso_pred = (iso.predict(X_te_s) == -1).astype(int)   # -1 -> anomaly

    result = {
        "dataset": ds["name"],
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "anomaly_rate_true": float(np.mean(y_te)),
        "anomaly_rate_pred": float(np.mean(iso_pred)),
    }

    # if labels exist, report supervised + unsupervised quality
    if len(np.unique(y_te)) > 1:
        result["iso_precision"] = round(float(precision_score(y_te, iso_pred, zero_division=0)), 4)
        result["iso_recall"] = round(float(recall_score(y_te, iso_pred, zero_division=0)), 4)
        clf = LogisticRegression(max_iter=200).fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_te_s)[:, 1]
        result["supervised_auc"] = round(float(roc_auc_score(y_te, proba)), 4)
    return result
