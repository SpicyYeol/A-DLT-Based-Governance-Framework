"""DL/ML manifold & distribution checks (proposed enhanced trust signals).

Two complementary, learning-theoretic controls that go beyond the update-*norm*
anomaly (M-BC-01):

1. Update-direction manifold analysis  (per-agent, pre/at aggregation)
   Honest local updates lie on a shared low-dimensional manifold and point in
   similar directions; poisoned updates that manipulate *direction* (label-flip,
   backdoor) leave that manifold even when their *norm* is normal. We score each
   agent by (a) mean cosine similarity to the cohort and (b) residual off the
   top-k principal subspace (a spectral-signature style test). Colluders form
   their own off-manifold cluster and are caught too.

   Complementarity: norm-based M-BC-01 catches magnitude attacks (scaling);
   manifold catches directional attacks. Neither alone is sufficient.

2. Post-training output-distribution health  (system-level, at completion)
   After aggregation, a compromised global model shows a degenerate output
   manifold — e.g. label-flip-to-0 poisoning collapses predictions toward one
   class. We compare the predicted class distribution to the expected prior
   (KL divergence), the mean predictive entropy, and per-class coverage. This is
   the "manifold distribution check at training completion" — a governance gate
   on the trained model itself, not just on individual agents.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. update-direction manifold
# ---------------------------------------------------------------------------
def update_direction_scores(updates: dict[int, list[float]]) -> dict:
    """updates: {agent_id: flattened mean update vector}. Returns per-agent
    mean cosine similarity to the cohort and spectral residual, plus a robust
    outlier flag on the cosine statistic."""
    ids = sorted(updates)
    M = np.asarray([updates[i] for i in ids], dtype=float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    U = M / np.clip(norms, 1e-12, None)                 # unit directions
    cos = U @ U.T                                       # pairwise cosine
    n = len(ids)
    # mean cosine to *others*
    mean_cos = (cos.sum(axis=1) - 1.0) / max(n - 1, 1)

    # spectral residual: distance off the top-k principal subspace of directions
    k = min(3, n - 1) if n > 1 else 1
    Uc = U - U.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
        basis = Vt[:k]
        proj = Uc @ basis.T @ basis
        residual = np.linalg.norm(Uc - proj, axis=1)
    except np.linalg.LinAlgError:
        residual = np.zeros(n)

    # robust outlier flag on (low) mean cosine
    med = np.median(mean_cos)
    mad = np.median(np.abs(mean_cos - med)) or 1e-9
    z = 0.6745 * (mean_cos - med) / mad
    return {
        "mean_cosine": {i: round(float(c), 4) for i, c in zip(ids, mean_cos)},
        "spectral_residual": {i: round(float(r), 4) for i, r in zip(ids, residual)},
        "cosine_z": {i: round(float(zi), 3) for i, zi in zip(ids, z)},
    }


def manifold_flagged(scores: dict, z_tau: float = 2.0) -> set[int]:
    """Agents whose direction sits off the honest manifold (low cosine outlier)."""
    return {i for i, z in scores["cosine_z"].items() if z < -z_tau}


# ---------------------------------------------------------------------------
# 2. post-training output-distribution health
# ---------------------------------------------------------------------------
def model_output_health(model, X, n_classes: int,
                        expected_prior: np.ndarray | None = None) -> dict:
    """Distribution check on the trained global model's outputs.

    Returns predicted class-distribution KL vs the expected prior, mean
    predictive entropy, the largest single-class share, and a boolean 'healthy'
    (predictions not collapsed onto few classes)."""
    preds = model.predict(X)
    counts = np.bincount(preds, minlength=n_classes).astype(float)
    p = counts / max(counts.sum(), 1)
    q = (expected_prior if expected_prior is not None
         else np.full(n_classes, 1.0 / n_classes))
    eps = 1e-9
    kl = float(np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0.0)))
    entropy = float(-np.sum(np.where(p > 0, p * np.log(p + eps), 0.0)))
    max_share = float(p.max())
    # try a soft-prediction entropy if available (confidence health)
    try:
        proba = model.predict_proba(X)
        mean_conf_entropy = float(np.mean(
            -np.sum(proba * np.log(proba + eps), axis=1)))
    except Exception:
        mean_conf_entropy = None
    healthy = (kl < 0.20) and (max_share < 2.0 / n_classes)
    return {"pred_class_kl": round(kl, 4),
            "pred_class_entropy": round(entropy, 4),
            "max_class_share": round(max_share, 4),
            "mean_confidence_entropy": (round(mean_conf_entropy, 4)
                                        if mean_conf_entropy is not None else None),
            "healthy": bool(healthy)}
