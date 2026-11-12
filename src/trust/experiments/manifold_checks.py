# -*- coding: utf-8 -*-
"""Manifold and distribution checks after training.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _flagged_set
from .thresholds import _anomaly_z


# ---------------------------------------------------------------------------
_UPDATE_ATTACKS = {"scale_attack", "stealth_scale", "label_flip", "collude_flip"}


def run_manifold(seed: int | None = None, profile: dict | None = None) -> dict:
    """Compare the update-*norm* detector (M-BC-01) with an update-*direction*
    manifold detector, and run a post-training output-distribution health check
    on the naive (poisoned) vs governance-gated global model."""
    import numpy as np
    from .. import manifold as mf
    seed = config.EXP_BASE_SEED if seed is None else seed
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)

    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile, return_updates=True,
                                return_model=True)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()

    # detectors
    znorm = _anomaly_z(signals)                     # |z| of update norm
    mscores = mf.update_direction_scores(naive["updates"])
    norm_flagged = {i for i in ids if znorm[i] > config.ANOMALY_Z_TAU}
    man_flagged = mf.manifold_flagged(mscores, z_tau=2.0)
    combined = norm_flagged | man_flagged

    # ground truth: only the *update-manipulating* attackers are in scope here
    update_adv = {a for a, b in profile.items() if b in _UPDATE_ATTACKS}
    honest = set(ids) - set(profile) - {config.REVOKED_AGENT}

    def rec_fpr(flagged):
        tp = len(flagged & update_adv)
        rec = tp / max(len(update_adv), 1)
        fp = len(flagged & honest)
        fpr = fp / max(len(honest), 1)
        return round(rec, 3), round(fpr, 3)

    n_rec, n_fpr = rec_fpr(norm_flagged)
    m_rec, m_fpr = rec_fpr(man_flagged)
    c_rec, c_fpr = rec_fpr(combined)

    per_agent = []
    for i in ids:
        beh = ctx.agents[i].get("behaviour")
        caught = []
        if i in norm_flagged:
            caught.append("norm")
        if i in man_flagged:
            caught.append("manifold")
        per_agent.append({"agent": i, "behaviour": beh,
                          "norm_z": round(znorm[i], 2),
                          "cosine_z": mscores["cosine_z"][i],
                          "is_update_adversary": i in update_adv,
                          "caught_by": caught})

    # post-training distribution health: naive (poisoned) vs governance-gated
    flagged_all = _flagged_set(results, ids)
    gated = fl.train_trajectory(prep, exclude=flagged_all, profile=profile,
                                return_model=True)
    prior = prep["base_dist"]
    ncls = prep["n_classes"]
    health_naive = mf.model_output_health(naive["model"], prep["X_te"], ncls, prior)
    health_gated = mf.model_output_health(gated["model"], prep["X_te"], ncls, prior)

    return {
        "seed": seed,
        "detectors": {
            "norm_only": {"recall_update_attacks": n_rec, "fpr": n_fpr},
            "manifold_only": {"recall_update_attacks": m_rec, "fpr": m_fpr},
            "combined": {"recall_update_attacks": c_rec, "fpr": c_fpr},
        },
        "update_adversaries": sorted(update_adv),
        "per_agent": per_agent,
        "post_training_health": {"naive_model": health_naive,
                                 "governance_gated_model": health_gated},
    }
