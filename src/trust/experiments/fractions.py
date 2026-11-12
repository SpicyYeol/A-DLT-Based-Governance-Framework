# -*- coding: utf-8 -*-
"""Adversary-fraction sweep.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _agg, _conf, _flagged_set


_ATTACK_ORDER = [1, 3, 5, 6, 2, 4, 0]          # slots to fill as k grows
_ATTACK_CYCLE = ["scale_attack", "label_flip", "stale", "tamper"]


def make_profile(k: int) -> dict:
    return {_ATTACK_ORDER[i]: _ATTACK_CYCLE[i % len(_ATTACK_CYCLE)]
            for i in range(min(k, len(_ATTACK_ORDER)))}


def run_fraction_sweep(ks: list[int] | None = None,
                       seeds: list[int] | None = None) -> dict:
    """Vary the number of malicious agents k and measure accuracy + detection.
    Disables the standalone revoked-agent during the sweep so ground truth is
    exactly the k injected attackers."""
    n = config.NUM_AGENTS
    ks = ks or list(range(1, n))            # 1 .. n-1 malicious
    seeds = seeds or config.exp_seeds()[:2]  # keep the sweep bounded
    saved_revoked = config.REVOKED_AGENT
    config.REVOKED_AGENT = -1               # no standalone revocation here
    try:
        rows = []
        for k in ks:
            profile = make_profile(k)
            gt_bad = set(profile)
            per = {"naive": [], "gov": [], "oracle": [],
                   "recall": [], "precision": [], "fpr": []}
            for s in seeds:
                prep = fl.prepare(s)
                naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                            profile=profile)
                signals = naive["signals"]
                ctx = build_ctx(signals)
                results = evaluate_controls(ctx)
                ids = ctx.agent_ids()
                flagged = _flagged_set(results, ids)
                det = _conf(flagged, ids, gt_bad)
                per["naive"].append(naive["accuracy"])
                per["gov"].append(fl.train_trajectory(
                    prep, exclude=flagged, profile=profile)["accuracy"])
                per["oracle"].append(fl.train_trajectory(
                    prep, exclude=gt_bad, profile=profile)["accuracy"])
                per["recall"].append(det["recall"])
                per["precision"].append(det["precision"])
                per["fpr"].append(det["false_positive_rate"])
            rows.append({
                "k": k, "malicious_fraction": round(k / n, 3),
                "naive_accuracy": _agg(per["naive"]),
                "governance_accuracy": _agg(per["gov"]),
                "oracle_accuracy": _agg(per["oracle"]),
                "detection_recall": _agg(per["recall"]),
                "detection_precision": _agg(per["precision"]),
                "detection_fpr": _agg(per["fpr"]),
            })
    finally:
        config.REVOKED_AGENT = saved_revoked
    return {"seeds": seeds, "num_agents": n, "rows": rows}
