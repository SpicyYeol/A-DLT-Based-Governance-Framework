"""Open-dataset loaders with offline fallbacks.

Two open datasets, matching the two layers the deck cares about:

* Image / federated-learning layer  -> MNIST (OpenML, id 554).
  Fallback: scikit-learn ``load_digits`` (bundled, 8x8, real dataset).
  Drives FL Quality Assessment + Behavioral Consistency + Reputation (DL-oriented).

* Tabular / network-security layer   -> KDDCup'99 (scikit-learn fetch, 10% subset).
  Fallback: deterministic synthetic telemetry.
  Drives transaction/anomaly trust for the satellite/IoT use case (ML-oriented).

All downloads are cached under ``config.DATASET_CACHE`` so repeated DAG runs are
offline after the first fetch.
"""

from __future__ import annotations

import os
import numpy as np

from . import config

os.environ.setdefault("SCIKIT_LEARN_DATA", str(config.DATASET_CACHE))


# ---------------------------------------------------------------------------
# Image dataset (federated learning)
# ---------------------------------------------------------------------------
# OpenML image datasets usable as drop-in federated tasks (784 features, 10 cls)
_OPENML_IMAGE = {
    "mnist": ("mnist_784", "MNIST (OpenML id=554)"),
    "fashion_mnist": ("Fashion-MNIST", "Fashion-MNIST (OpenML id=40996)"),
}


def load_image_dataset(name: str | None = None) -> dict:
    """Return {X, y, n_classes, name}. X in [0,1], shape (n, features).

    ``name`` selects the open image dataset (mnist | fashion_mnist) for the C8
    cross-dataset generalization experiment; defaults to config.IMAGE_DATASET.
    """
    name = (name or getattr(config, "IMAGE_DATASET", "mnist")).lower()
    cap = config.IMAGE_SAMPLE_CAP
    rng = np.random.RandomState(config.RANDOM_SEED)
    openml_id, label = _OPENML_IMAGE.get(name, _OPENML_IMAGE["mnist"])
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(openml_id, version=1, as_frame=False,
                          data_home=str(config.DATASET_CACHE), parser="auto")
        X = np.asarray(ds.data, dtype=np.float32) / 255.0
        y = np.asarray(ds.target, dtype=int)
        if len(X) > cap:
            idx = rng.choice(len(X), cap, replace=False)
            X, y = X[idx], y[idx]
        return {"X": X, "y": y, "n_classes": 10, "name": label}
    except Exception as exc:  # offline / fetch failure -> bundled digits
        from sklearn.datasets import load_digits
        d = load_digits()
        X = np.asarray(d.data, dtype=np.float32) / 16.0
        y = np.asarray(d.target, dtype=int)
        return {"X": X, "y": y, "n_classes": 10,
                "name": f"sklearn digits ({name} fallback: {type(exc).__name__})"}


# ---------------------------------------------------------------------------
# Tabular dataset (network / IoT security)
# ---------------------------------------------------------------------------
def load_tabular_dataset() -> dict:
    """Return {X, y, feature_names, name} for anomaly/security scoring.

    Label: 0 = normal, 1 = anomalous/attack.
    """
    cap = config.TABULAR_SAMPLE_CAP
    rng = np.random.RandomState(config.RANDOM_SEED)
    try:
        from sklearn.datasets import fetch_kddcup99
        import pandas as pd
        kdd = fetch_kddcup99(percent10=True, as_frame=True, download_if_missing=True,
                             data_home=str(config.DATASET_CACHE))
        df = kdd.frame.copy()
        # fetch_kddcup99 returns every column as object (bytes); coerce the
        # numeric ones and drop the 3 categorical string columns.
        labels = df["labels"].apply(
            lambda v: v.decode() if isinstance(v, bytes) else str(v))
        y = (~labels.str.startswith("normal")).astype(int).to_numpy()
        feat_df = df.drop(columns=["labels"])
        num = feat_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        num = num.fillna(0.0)
        X = num.to_numpy(dtype=np.float32)
        feats = list(num.columns)
        if len(X) > cap:
            idx = rng.choice(len(X), cap, replace=False)
            X, y = X[idx], y[idx]
        return {"X": X, "y": y, "feature_names": feats,
                "name": "KDDCup'99 10% (scikit-learn)"}
    except Exception as exc:
        return _synthetic_telemetry(rng, cap, exc)


def _synthetic_telemetry(rng, n, exc) -> dict:
    """Deterministic satellite/IoT-style telemetry with injected anomalies."""
    n = min(n, 6000)
    n_anom = n // 10
    normal = rng.normal(
        loc=[0.5, 20.0, 100.0, 0.1], scale=[0.05, 1.5, 5.0, 0.02],
        size=(n - n_anom, 4))
    anom = rng.normal(
        loc=[0.9, 35.0, 180.0, 0.6], scale=[0.2, 6.0, 30.0, 0.2],
        size=(n_anom, 4))
    X = np.vstack([normal, anom]).astype(np.float32)
    y = np.concatenate([np.zeros(len(normal)), np.ones(len(anom))]).astype(int)
    order = rng.permutation(len(X))
    return {"X": X[order], "y": y[order],
            "feature_names": ["signal_ratio", "temp_c", "latency_ms", "err_rate"],
            "name": f"synthetic telemetry (KDD fallback: {type(exc).__name__})"}


# ---------------------------------------------------------------------------
# Federated partitioning across agents
# ---------------------------------------------------------------------------
def partition_for_agents(X, y, num_agents: int, seed: int) -> list[dict]:
    """IID-ish random shard split so agents differ but overlap in classes."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    shards = np.array_split(idx, num_agents)
    return [{"agent_id": i, "X": X[s], "y": y[s]} for i, s in enumerate(shards)]


def partition_dirichlet(X, y, num_agents: int, alpha: float, seed: int,
                        n_classes: int) -> list[dict]:
    """Non-IID label partition via Dirichlet(alpha) per class (Hsu et al. 2019).

    Small alpha -> highly skewed per-agent label distributions (each agent sees
    few classes); large alpha -> approaches IID. Used by the B3 severity sweep.
    """
    rng = np.random.RandomState(seed)
    agent_idx: list[list[int]] = [[] for _ in range(num_agents)]
    for c in range(n_classes):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        props = rng.dirichlet([alpha] * num_agents)
        cuts = (np.cumsum(props) * len(cls_idx)).astype(int)[:-1]
        for a, part in enumerate(np.split(cls_idx, cuts)):
            agent_idx[a].extend(part.tolist())
    # guard against empty shards at very small alpha
    all_idx = rng.permutation(len(X))
    for a in range(num_agents):
        if len(agent_idx[a]) < 10:
            agent_idx[a].extend(all_idx[:10].tolist())
    return [{"agent_id": a, "X": X[np.array(ix)], "y": y[np.array(ix)]}
            for a, ix in enumerate(agent_idx)]
