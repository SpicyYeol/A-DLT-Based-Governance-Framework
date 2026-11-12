# -*- coding: utf-8 -*-
"""The full 36-control ablation: the marginal value of every control.

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


def run_control_ablation_full(seed: int | None = None,
                              profile: dict | None = None) -> dict:
    from ..taxonomy import ALL_CONTROLS
    seed = config.EXP_BASE_SEED if seed is None else seed
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    flagged_full = _flagged_set(results, ids)
    gov_acc = fl.train_trajectory(prep, exclude=flagged_full, profile=profile)["accuracy"]

    rows = []
    for cp in ALL_CONTROLS:
        flagged_ab = _flagged_set(results, ids, disabled=frozenset({cp.id}))
        # adversaries detected with the full control set that now escape
        newly = sorted((gt_bad & flagged_full) - flagged_ab)
        load_bearing = len(newly) > 0
        acc = (fl.train_trajectory(prep, exclude=flagged_ab, profile=profile)["accuracy"]
               if load_bearing else gov_acc)
        rows.append({
            "id": cp.id, "level": cp.level, "category": cp.category,
            "subcategory": cp.subcategory, "severity": cp.severity,
            "method": cp.method, "orient": cp.orient,
            "load_bearing": load_bearing,
            "n_newly_undetected": len(newly), "newly_undetected": newly,
            "accuracy_under_ablation": round(acc, 4),
            "delta_vs_governance": round(acc - gov_acc, 4),
        })
    n_load = sum(1 for r in rows if r["load_bearing"])
    return {"seed": seed, "governance_accuracy": round(gov_acc, 4),
            "flagged_full": sorted(flagged_full), "gt_bad": sorted(gt_bad),
            "n_load_bearing": n_load, "n_redundant": len(rows) - n_load,
            "rows": rows}
