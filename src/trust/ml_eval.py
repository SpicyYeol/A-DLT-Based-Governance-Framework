"""Deeper DL/ML evaluation beyond detection/accuracy.

Two learning-centric analyses that a DL/ML audience expects:

1. Contribution valuation (leave-one-out marginal accuracy) — data-valuation
   style. Each agent's marginal effect on the global model quantifies whether it
   *helps* or *harms*; a poisoner has strongly negative marginal, an honest agent
   ~zero/positive. Validates the reputation notion with a model-based ground truth.

2. Model-quality suite beyond top-1 accuracy — macro-F1, per-class and
   worst-class recall, and calibration (Expected Calibration Error). Poisoning
   (esp. targeted label-flip) collapses specific classes and de-calibrates
   confidence; governance gating should restore both.
"""

from __future__ import annotations

import numpy as np

from . import config
from . import fl_simulation as fl
from .engine import build_ctx, evaluate_controls
from .experiments import _flagged_set, _agg


def _label(aid: int, profile: dict) -> str:
    if aid == config.REVOKED_AGENT:
        return "revoked"
    return profile.get(aid, "honest")


# ---------------------------------------------------------------------------
# 1. leave-one-out contribution valuation
# ---------------------------------------------------------------------------
def run_contribution(seeds: list[int] | None = None,
                     profile: dict | None = None) -> dict:
    seeds = seeds or config.exp_seeds()[:3]
    profile = profile or config.ADVERSARY_PROFILE
    per = {}
    for s in seeds:
        prep = fl.prepare(s)
        base = fl.train_trajectory(prep, profile=profile)["accuracy"]  # all agents
        for sh in prep["shards"]:
            a = sh["agent_id"]
            wo = fl.train_trajectory(prep, exclude={a}, profile=profile)["accuracy"]
            per.setdefault(a, []).append(base - wo)   # +: helps, -: harms
    gt = set(profile) | {config.REVOKED_AGENT}
    rows = [{"agent": a, "behaviour": _label(a, profile), "is_adversary": a in gt,
             "marginal_contribution": _agg(v)} for a, v in sorted(per.items())]
    return {"seeds": seeds, "rows": rows}


# ---------------------------------------------------------------------------
# 2. model-quality suite (naive vs governance-gated)
# ---------------------------------------------------------------------------
def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    n = len(conf)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _quality(model, X, y, ncls: int) -> dict:
    from sklearn.metrics import f1_score, recall_score
    pred = model.predict(X)
    macro_f1 = float(f1_score(y, pred, average="macro", labels=range(ncls),
                              zero_division=0))
    rec = recall_score(y, pred, average=None, labels=list(range(ncls)),
                       zero_division=0)
    worst = float(np.min(rec))
    try:
        proba = model.predict_proba(X)
        conf = proba.max(axis=1)
        ece = _ece(conf, (pred == y).astype(float))
    except Exception:
        ece = None
    return {"macro_f1": round(macro_f1, 4), "worst_class_recall": round(worst, 4),
            "ece": round(ece, 4) if ece is not None else None}


def run_model_quality(seeds: list[int] | None = None,
                      profile: dict | None = None) -> dict:
    seeds = seeds or config.exp_seeds()
    profile = profile or config.ADVERSARY_PROFILE
    agg = {"naive": {"macro_f1": [], "worst_class_recall": [], "ece": []},
           "gated": {"macro_f1": [], "worst_class_recall": [], "ece": []}}
    for s in seeds:
        prep = fl.prepare(s)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                    profile=profile, return_model=True)
        ctx = build_ctx(naive["signals"])
        results = evaluate_controls(ctx)
        flagged = _flagged_set(results, ctx.agent_ids())
        gated = fl.train_trajectory(prep, exclude=flagged, profile=profile,
                                    return_model=True)
        ncls = prep["n_classes"]
        qn = _quality(naive["model"], prep["X_te"], prep["y_te"], ncls)
        qg = _quality(gated["model"], prep["X_te"], prep["y_te"], ncls)
        for k in agg["naive"]:
            if qn[k] is not None:
                agg["naive"][k].append(qn[k])
            if qg[k] is not None:
                agg["gated"][k].append(qg[k])
    return {"seeds": seeds,
            "naive": {k: _agg(v) for k, v in agg["naive"].items()},
            "gated": {k: _agg(v) for k, v in agg["gated"].items()}}
